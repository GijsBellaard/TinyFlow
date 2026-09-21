# TinyFlow

A tiny (120k parameters) flow matching model for generating MNIST digits.

![sampling](sampling.gif)

## Dependencies

The only dependencies are [PyTorch](https://pytorch.org/) (including Torchvision) and [Pillow](https://python-pillow.org/).

## How to run

To generate samples without training a new model run:
```bash
python main.py --skip-train  --ckpt model.pt --gif sampling.gif
```
This will load the model weights `model.pt`.\
The file `sampling.gif` will be created/overwritten, showing how the model generates MNIST digits.

To train the model and then generate some samples run:
```bash
python main.py --ckpt model.pt --gif sampling.gif
```
It will download the MNIST dataset if it's not already present.\
It will save the model weights to `model.pt`.

To check the FID of the model `model.pt` run:
```bash
python main.py --skip-train --ckpt model.pt --fid 10000
```
This will download 100MB of Inception-v3 weights.\
The train set has a FID of 1.76.\
The included model has a FID of 9.01.\
This might seem like a big difference, however, the train set plus some noise with σ=0.02 already has a FID of 21.45....

## Settings

The following command line arguments are available:
- `--embedding-dim`: Token embedding dimension (default: 96)
- `--depth`: Number of times the shared transformer block is applied (default: 8)
- `--heads`: Number of attention heads, must divide `--embedding-dim` (default: 8)
- `--patch-size`: Patch size, must divide 28 (default: 4)
- `--train-steps`: Number of training steps (default: 64000)
- `--batch-size`: Batch size (default: 512)
- `--lr`: Learning rate (default: 3e-3)
- `--warmup`: Warmup fraction (default: 0.05)
- `--ema`: Exponential moving average decay (default: 0.999)
- `--train-seed`: RNG seed for training (default: 0)
- `--sampling-steps`: Number of sampling steps (default: 50)
- `--gif-seed`: RNG seed for the samples shown in the GIF (default: 3)
- `--gif`: Output GIF filename, no GIF is made unless this is set (default: none)
- `--ckpt`: Checkpoint filename (default: `model.pt`)
- `--fid`: Number of samples used to compute FID, 0 disables (default: 0)
- `--fid-seed`: RNG seed for the samples used to compute FID (default: 0)
- `--skip-train`: Skip training and only sample from the model (default: False). The model options are read from the checkpoint, so `--embedding-dim`, `--depth`, `--heads` and `--patch-size` are ignored.

So, for example, you can run:
```bash
python main.py --train-steps 10000 --batch-size 256 --lr 1e-3
```
or
```bash
python main.py --skip-train --ckpt model.pt --gif my_sampling.gif --sampling-steps 10 --gif-seed 42
```

## Architecture

Some notable architectural choices:
- The model is a vision transformer.
- Given the input $x_t$, the model is trained to directly predict $x_1$, rather than predicting the velocity field $v_t$ of the flow.
- The model does _not_ see $t$, it only sees $x_t$. The noise level has to be inferred from the input itself.
- A single transformer block is applied 8 times with shared weights.
- $t$ is drawn from a logit-normal distribution, $t = \sigma(z)$ where $\sigma$ is the sigmoid function and $z \sim \mathcal{N}(0, 1)$ is standard normally distributed.
- An exponential moving average of the model weights is maintained during training and used for sampling.
- Self rolled attention is used as this is faster than the built in `scaled_dot_product_attention` in this minimal example.
- The output head has no weights itself; it is the transpose of the patch projection.