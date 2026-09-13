# TinyFlow

A tiny (120k parameters) flow matching model for generating MNIST digits.

![sampling](sampling.gif)

## Dependencies

The only dependencies are [PyTorch](https://pytorch.org/) and [Pillow](https://python-pillow.org/).

## How to run

To run the model:
```bash
python main.py
```
It will download the MNIST dataset if it's not already present.\
Once the model is trained, it will save the model weights to `model.pt`.\
After training `sampling.gif` will be generated, showing how the model generates MNIST digits.

## Settings

The following command line arguments are available:
- `--train-steps`: Number of training steps (default: 32000)
- `--batch-size`: Batch size (default: 512)
- `--lr`: Learning rate (default: 3e-3)
- `--warmup`: Warmup fraction (default: 0.05)
- `--ema`: Exponential moving average decay (default: 0.999)
- `--seed`: Random seed (default: 0)
- `--sample-steps`: Number of sampling steps (default: 200)
- `--gif`: Output GIF filename (default: sampling.gif)
- `--ckpt`: Checkpoint filename (default: model.pt)
- `--skip-train`: Skip training and only sample from the model (default: False)

So, for example, you can run:
```bash
python main.py --train-steps 10000 --batch-size 256 --lr 1e-3
```