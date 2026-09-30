import argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from PIL import Image

NUM_CLASSES = 10 # digits 0-9
NULL_CLASS = NUM_CLASSES # extra class index meaning 'no class', used for CFG

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h

        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dh) # [B, N, 3, h, dh]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0) # 3 x [B, h, N, dh]
        a = (q @ k.transpose(-1, -2)) * self.dh ** -.5 # [B, h, N, N]
        a = a.softmax(-1) @ v # [B, h, N, dh]
        # a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))

class Model(nn.Module):
    def __init__(self, embedding_dim, depth, heads, patch_size, patch_stride):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.depth = depth
        self.g = (28 - patch_size) // patch_stride + 1

        self.proj = nn.Conv2d(1, embedding_dim, patch_size, stride=patch_stride)
        self.pos = nn.Parameter(torch.randn(1, self.g * self.g, embedding_dim) * .02)
        self.cls = nn.Embedding(NUM_CLASSES + 1, embedding_dim) # + 1 for NULL_CLASS
        self.block = Block(embedding_dim, heads)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, y):
        h = self.proj(x) # [B, 1, H, W] -> [B, D, g, g]
        h = h.flatten(2).transpose(1, 2) # [B, D, g, g] -> [B, N, D]
        h = h + self.pos
        c = self.cls(y).unsqueeze(1) # [B] -> [B, 1, D]
        h = torch.cat([c, h], 1) # prepend class token -> [B, 1 + N, D]
        for i in range(self.depth):
            h = self.block(h)
        h = self.norm(h[:, 1:]) # drop class token -> [B, N, D]
        h = h.transpose(1, 2).unflatten(2, (self.g, self.g)).contiguous() # [B, N, D] -> [B, D, g, g]
        return F.conv_transpose2d(h, self.proj.weight, stride=self.patch_stride) # [B, D, g, g] -> [B, 1, H, W]

def train(model, steps, batch_size, lr, warmup, ema_decay, class_dropout):
    device = next(model.parameters()).device
    mnist = MNIST('./data', train=True, download=True)
    X = (mnist.data.float() / 127.5 - 1.0).unsqueeze(1).to(device)
    Y = mnist.targets.to(device)

    params = list(model.parameters())
    ema = [torch.zeros_like(q) for q in params] # ema of parameters
    loss_ema = torch.zeros((), device=device) # ema of loss
    fwd = torch.compile(model, mode='max-autotune') if device.type == 'cuda' else model
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95), fused=device.type == 'cuda')
    W = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / W))

    it = 0
    try:
        for it in range(steps):
            idx = torch.randint(0, X.shape[0], (batch_size,), device=device)
            x1, y = X[idx], Y[idx]
            drop = torch.rand(batch_size, device=device) < class_dropout # drop class to learn unconditional too
            y = torch.where(drop, NULL_CLASS, y)
            u = torch.arange(batch_size, device=device) + torch.rand(1, device=device)
            u = u / batch_size # even spread over (0, 1) to reduce variance
            t = torch.sigmoid(torch.special.ndtri(u)).view(-1, 1, 1, 1)
            x0 = torch.randn_like(x1)
            xt = (1 - t) * x0 + t * x1
            with torch.autocast(device.type, torch.bfloat16):
                x1_pred = fwd(xt, y)
                loss = (((x1_pred - x1) / (1 - t).clamp(min=0.05)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema, params, 1 - ema_decay)
                loss_ema.lerp_(loss.float(), 1 - ema_decay)
            if (it + 1) % 1000 == 0:
                loss_avg = loss_ema.item() / (1 - ema_decay ** (it + 1)) # ema bias correction
                print(f'{it+1}/{steps}  loss {loss_avg:.5f}  lr {sched.get_last_lr()[0]:.2e}')
    except KeyboardInterrupt:
        steps = it + 1
        print(f'\ninterrupted after {steps} steps')

    with torch.no_grad():
        for q, e in zip(params, ema):
            q.copy_(e / (1 - ema_decay ** steps)) # ema bias correction

def predict(model, xt, y, cfg): # classifier-free guidance
    xx = torch.cat([xt, xt])
    yy = torch.cat([y, torch.full_like(y, NULL_CLASS)]) # conditional and unconditional
    c, u = model(xx, yy).chunk(2)
    return u + cfg * (c - u)

@torch.no_grad()
def make_sampling_gif(model, path, steps, cfg, size=10):
    model.eval()
    gif = []
    device = next(model.parameters()).device
    xt = torch.randn(size * size, 1, 28, 28, device=device)
    y = torch.arange(size * size, device=device) % NUM_CLASSES # class labels 0, 1, ..., 9 arranged
    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        x1 = predict(model, xt, y, cfg)
        g = make_grid(x1.clamp(-1, 1) * .5 + .5, nrow=size, padding=2)[0]
        gif.append(Image.fromarray((g * 255).byte().cpu().numpy()).quantize(16))
        xt = xt + (x1 - xt) / (1 - t) * dt
    gif[0].save(path, save_all=True, append_images=gif[1:], loop=0, duration=max(1,1200//steps))
    print(f'wrote {path}')

def frechet(f1, f2): # frechet distance between gaussians fitted to two sets of features
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = torch.cov(f1.T), torch.cov(f2.T)
    e, v = torch.linalg.eigh(c1)
    r = v @ torch.diag(e.clamp(min=0).sqrt()) @ v.T # c1^(1/2)
    s = torch.linalg.eigvalsh(r @ c2 @ r).clamp(min=0).sqrt().sum() # tr((c1 c2)^(1/2))
    return ((m1 - m2) ** 2).sum() + c1.trace() + c2.trace() - 2 * s

@torch.no_grad()
def sample(model, n, steps, cfg, batch=500):
    model.eval()
    out = []
    device = next(model.parameters()).device
    dt = 1.0 / steps
    for i in range(0, n, batch):
        xt = torch.randn(min(batch, n - i), 1, 28, 28, device=device)
        y = torch.randint(0, NUM_CLASSES, (len(xt),), device=device) # random class labels 0, 1, ..., 9
        for j in range(steps):
            t = j * dt
            x1 = predict(model, xt, y, cfg)
            xt = xt + (x1 - xt) / (1 - t) * dt
        out.append(xt)
    return torch.cat(out)

def calculate_fid(model, n, steps, cfg):
    from torchvision.models import inception_v3, Inception_V3_Weights
    device = next(model.parameters()).device
    inc = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
    inc.fc = nn.Identity() # [B, 3, 299, 299] -> [B, 2048] pool features
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(x, batch=250):
        f = []
        for i in range(0, len(x), batch):
            y = x[i:i + batch].to(device).clamp(-1, 1) * .5 + .5 # [-1, 1] -> [0, 1]
            y = y.expand(-1, 3, -1, -1) # grayscale -> rgb
            y = F.interpolate(y, 299, mode='bilinear', align_corners=False)
            f.append(inc((y - mean) / std).double())
        return torch.cat(f)

    test = (MNIST('./data', train=False, download=True).data.float() / 127.5 - 1.0).unsqueeze(1)
    train = (MNIST('./data', train=True, download=True).data.float() / 127.5 - 1.0).unsqueeze(1)
    samples = sample(model, n, steps, cfg)
    test_ft = features(test)
    train_ft = features(train[:n])
    sampled_ft = features(samples)
    print(f'fid train {frechet(train_ft, test_ft):.2f}')
    print(f'fid model {frechet(sampled_ft, test_ft):.2f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embedding-dim', type=int, default=96)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--patch-size', type=int, default=4)
    p.add_argument('--patch-stride', type=int, default=4)
    p.add_argument('--train-steps', type=int, default=64000)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=float, default=0.05)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--class-dropout', type=float, default=0.1)
    p.add_argument('--train-seed', type=int, default=0)
    p.add_argument('--gif-seed', type=int, default=0)
    p.add_argument('--sampling-steps', type=int, default=15)
    p.add_argument('--cfg', type=float, default=1.0)
    p.add_argument('--gif', default=None)
    p.add_argument('--ckpt', default='model.pt')
    p.add_argument('--skip-train', action='store_true')
    p.add_argument('--fid', type=int, default=0)
    p.add_argument('--fid-seed', type=int, default=0)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.skip_train:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_args = ckpt['model_args']
    else:
        model_args = dict(
            embedding_dim=args.embedding_dim, 
            depth=args.depth, 
            heads=args.heads, 
            patch_size=args.patch_size, 
            patch_stride=args.patch_stride
        )
    model = Model(**model_args).to(device)
    param_count = sum(q.numel() for q in model.parameters())
    print(f'{param_count} params')

    if args.skip_train:
        model.load_state_dict(ckpt['model'])
    else:
        torch.manual_seed(args.train_seed)
        train(model, args.train_steps, args.batch_size, args.lr, args.warmup, args.ema, args.class_dropout)
        torch.save({'model_args': model_args, 'model': model.state_dict()}, args.ckpt)
        print(f'wrote {args.ckpt}')

    if args.gif:
        torch.manual_seed(args.gif_seed)
        make_sampling_gif(model, args.gif, args.sampling_steps, args.cfg)

    if args.fid:
        torch.manual_seed(args.fid_seed)
        calculate_fid(model, args.fid, args.sampling_steps, args.cfg)
