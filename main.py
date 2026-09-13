import argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from PIL import Image

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
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        a = ((q @ k.transpose(-1, -2)) * self.dh ** -.5).softmax(-1) @ v
        x = x + self.o(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))

class Model(nn.Module):
    def __init__(self, embedding_dim, depth, heads, patch_size):
        super().__init__()
        self.patch_size, self.depth, self.g = patch_size, depth, 28 // patch_size
        self.proj = nn.Linear(patch_size * patch_size, embedding_dim)
        self.pos = nn.Parameter(torch.randn(1, self.g * self.g, embedding_dim) * .02)
        self.block = Block(embedding_dim, heads)
        self.depth_emb = nn.Parameter(torch.zeros(depth, 1, 1, embedding_dim))
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, patch_size * patch_size)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        k, g, B = self.patch_size, self.g, x.shape[0]
        h = self.proj(x.view(B, 1, g, k, g, k).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, k * k))
        h = h + self.pos
        for i in range(self.depth):
            h = self.block(h + self.depth_emb[i])
        v = self.head(self.norm(h))
        return v.view(B, g, g, k, k).permute(0, 1, 3, 2, 4).reshape(B, 1, g * k, g * k)

p = argparse.ArgumentParser()
p.add_argument('--embedding-dim', type=int, default=96)
p.add_argument('--depth', type=int, default=8)
p.add_argument('--heads', type=int, default=8)
p.add_argument('--patch-size', type=int, default=4)
p.add_argument('--train-steps', type=int, default=32000)
p.add_argument('--batch-size', type=int, default=512)
p.add_argument('--lr', type=float, default=3e-3)
p.add_argument('--warmup', type=float, default=0.05)
p.add_argument('--ema', type=float, default=0.999)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--sample-steps', type=int, default=200)
p.add_argument('--gif', default='sampling.gif')
p.add_argument('--ckpt', default='model.pt')
p.add_argument('--skip-train', action='store_true')
args = p.parse_args()

torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Model(args.embedding_dim, args.depth, args.heads, args.patch_size).to(device)
params = list(net.parameters())
param_count = sum(q.numel() for q in params)
print(f'{param_count} params')

if args.skip_train:
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
else:
    X = (MNIST('./data', train=True, download=True).data.float() / 127.5 - 1.0).unsqueeze(1).to(device)

    ema = [torch.zeros_like(q) for q in params]
    fwd = torch.compile(net, mode='max-autotune') if device == 'cuda' else net
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=device == 'cuda')
    W = max(1, int(args.warmup * args.train_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / W))

    for it in range(args.train_steps):
        x1 = X[torch.randint(0, X.shape[0], (args.batch_size,), device=device)]
        t = torch.sigmoid(torch.randn(args.batch_size, 1, 1, 1, device=device))
        xt = (1 - t) * torch.randn_like(x1) + t * x1
        with torch.autocast(device, torch.bfloat16):
            loss = F.mse_loss(fwd(xt), x1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema, params, 1 - args.ema)
        if (it + 1) % 1000 == 0:
            print(f'{it+1}/{args.train_steps}  loss {loss.item():.5f}  lr {sched.get_last_lr()[0]:.2e}')
    print()

    with torch.no_grad():
        for q, e in zip(params, ema):
            q.copy_(e / (1 - args.ema ** args.train_steps))
    torch.save(net.state_dict(), args.ckpt)
    print(f'wrote {args.ckpt}')
net.eval()

gif = []

torch.manual_seed(3)
x = torch.randn(64, 1, 28, 28, device=device)

dt = 1.0 / args.sample_steps
with torch.no_grad():
    for i in range(args.sample_steps):
        x1_hat = net(x)
        g = make_grid(x1_hat.clamp(-1, 1) * .5 + .5, nrow=8, padding=2)[0]
        gif.append(Image.fromarray((g * 255).byte().cpu().numpy()).quantize(16))
        x = x + (x1_hat - x) / max(1 - i * dt, dt) * dt
gif[0].save(args.gif, save_all=True, append_images=gif[1:], loop=0, duration=60)
print(f'wrote {args.gif}')
