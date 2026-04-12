<div align="center">

![AE](https://img.shields.io/badge/AE-reconstruction-1a7f37?style=for-the-badge)

## Autoencoder (MNIST)

*Encoder → bottleneck → decoder; reconstruction loss (e.g. MSE).*

[![Notebook](https://img.shields.io/badge/📓-autoencoder__mnist__app.ipynb-F37626?style=flat-square)](autoencoder_mnist_app.ipynb)

</div>

---

## Code map

| | |
|--|--|
| **[`extractor/data_loader.py`](extractor/data_loader.py)** | MNIST loading helpers |
| **[`Model/AutoEncoder.py`](Model/AutoEncoder.py)** | FC encoder / decoder (PyTorch) |
| **[`config/torch_config.json`](config/torch_config.json)** | Training settings |

---

## Requirements

| | |
|--|--|
| **Python** | 3.10.x |
| **Notebook** | Jupyter 1.0.0 |
| **Core** | numpy · pandas |
| **Deep learning** | torch · torchvision |
