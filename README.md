<div align="center">

![Python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-generative-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)

# Generative AI — model walkthroughs

*Small notebooks and helpers — patterns that reappear in diffusion & sequence models.*

[![Browse precursors](https://img.shields.io/badge/📂%20precursors-explore-24292f?style=flat-square)](precursors/)

</div>

Roadmap in three steps: **precursors** (what exists today) → **diffusion** → **Transformers**.

> **Run notebooks** from **inside** that model’s folder (e.g. `precursors/gan/`) so imports like `Model....` resolve.

---

## Part 1 — Precursors *(in repo)*

Short examples under **[`precursors/`](precursors/)** — patterns that show up again in diffusion and sequence models.

### Models

| | Topic | Idea | Folder | Notebook |
|---|--------|------|--------|----------|
| ![AR](https://img.shields.io/badge/AR-time%20series-0366d6?style=flat-square) | Auto-regressive | Next step from past lags | [`precursors/ar/`](precursors/ar/) | [`autoregressive_model_app.ipynb`](precursors/ar/autoregressive_model_app.ipynb) |
| ![GAN](https://img.shields.io/badge/GAN-adversarial-8957e5?style=flat-square) | GAN | \(p(x)\) via generator vs. discriminator | [`precursors/gan/`](precursors/gan/) | [`gan_torch_model_app.ipynb`](precursors/gan/gan_torch_model_app.ipynb) |
| ![AE](https://img.shields.io/badge/AE-reconstruction-1a7f37?style=flat-square) | Autoencoder | Encode → latent → decode | [`precursors/ae/`](precursors/ae/) | [`autoencoder_mnist_app.ipynb`](precursors/ae/autoencoder_mnist_app.ipynb) |

### Background reading

| Model | Articles |
|--------|----------|
| **AR** | [Wikipedia (EN)](https://en.wikipedia.org/wiki/Autoregressive_model) · [위키백과 (KO)](https://ko.wikipedia.org/wiki/%EC%9E%90%EA%B8%B0%ED%9A%8C%EA%B7%80%EB%AA%A8%ED%98%95) |
| **GAN** | [Original paper (PDF)](https://arxiv.org/pdf/1406.2661) · [위키백과 (KO)](https://ko.wikipedia.org/wiki/%EC%83%9D%EC%84%B1%EC%A0%81_%EC%A0%81%EB%8C%80_%EC%8B%A0%EA%B2%BD%EB%A7%9D) |
| **AE** | [Wikipedia (EN)](https://en.wikipedia.org/wiki/Autoencoder) · [위키백과 (KO)](https://ko.wikipedia.org/wiki/%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94) |

---

## Part 2 — Diffusion models *(planned)*

Notebook(s) and notes on **diffusion** — forward noise process, denoising / score-based views, sampling. Links and paths TBD.

---

## Part 3 — Transformer models *(planned)*

Notebook(s) and notes on **Transformers** — attention, sequence modeling, and ties to modern LLMs. Links and paths TBD.

---

## Requirements

| | |
|--|--|
| **Runtime** | Python **3.10.x**, Jupyter **1.0.0** |
| **Common** | numpy · pandas · scikit-learn · statsmodels · scipy |
| **Neural** | **torch**, **torchvision** (GAN & AE notebooks) |
