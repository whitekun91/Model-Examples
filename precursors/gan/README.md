<div align="center">

![GAN](https://img.shields.io/badge/GAN-adversarial-8957e5?style=for-the-badge)

## Generative adversarial network (MNIST)

*Generator and discriminator trained together — synthetic digits vs. real.*

[![Notebook](https://img.shields.io/badge/📓-gan__torch__model__app.ipynb-F37626?style=flat-square)](gan_torch_model_app.ipynb)

</div>

---

## Code map

| | |
|--|--|
| **[`extractor/data_loader.py`](extractor/data_loader.py)** | MNIST → tensors / image dump |
| **[`Model/GAN.py`](Model/GAN.py)** | Generator & discriminator (PyTorch) |
| **[`config/torch_config.json`](config/torch_config.json)** | Training knobs |

---

## Requirements

| | |
|--|--|
| **Python** | 3.10.x |
| **Notebook** | Jupyter 1.0.0 |
| **Core** | numpy · pandas |
| **Deep learning** | torch · torchvision |
