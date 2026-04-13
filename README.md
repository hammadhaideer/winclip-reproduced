# Anomaly Detection Playground

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Type-Research%20%2B%20Applied-blueviolet?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*My primary research domain - statistical, ML, and deep learning methods for detecting the unusual.*

## About This Repository

Anomaly Detection is my **core research specialization** at Xinjiang University. This repository is both a learning resource and a reflection of my active research — covering tabular, time-series, text, and visual anomaly detection.

> "An anomaly is a data point that differs significantly from the majority — detecting it is a matter of safety, quality, and intelligence."

## Repository Structure

```
anomaly-detection-playground/
│
├── 01_fundamentals/
│   ├── 01_what_is_anomaly_detection.ipynb
│   ├── 02_types_of_anomalies.ipynb
│   └── 03_evaluation_metrics.ipynb
│
├── 02_statistical_methods/
│   ├── 01_zscore_iqr.ipynb
│   ├── 02_grubbs_test.ipynb
│   └── 03_gaussian_mixture_models.ipynb
│
├── 03_ml_methods/
│   ├── 01_isolation_forest.ipynb
│   ├── 02_one_class_svm.ipynb
│   ├── 03_local_outlier_factor.ipynb
│   └── 04_dbscan_for_anomalies.ipynb
│
├── 04_deep_learning_methods/
│   ├── 01_autoencoder_anomaly.ipynb
│   ├── 02_variational_autoencoder.ipynb
│   └── 03_lstm_time_series_anomaly.ipynb
│
├── 05_nlp_anomaly/
│   ├── 01_text_anomaly_detection.ipynb
│   └── 02_log_anomaly_detection.ipynb
│
├── 06_visual_anomaly/
├── 07_benchmarks_and_datasets/
├── 08_paper_implementations/
└── resources.md
```

## Notebooks

| # | Method | Domain | Dataset | Status |
|---|--------|--------|---------|--------|
| 01 | Isolation Forest | Tabular | KDD Cup | ⏳ |
| 02 | Autoencoder | Tabular | Credit Card Fraud | ⏳ |
| 03 | LSTM Anomaly | Time Series | NASA SMAP | ⏳ |
| 04 | Text Anomaly | NLP | Custom Logs | ⏳ |

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/anomaly-detection-playground.git
cd anomaly-detection-playground
pip install -r requirements.txt
jupyter notebook
```

## Tech Stack

`Python` · `scikit-learn` · `PyTorch` · `PyOD` · `NumPy` · `Pandas` · `Matplotlib`

## Connect

[LinkedIn](https://www.linkedin.com/in/hammadhaideer)
[GitHub](https://github.com/hammadhaideer)

⭐ Star this repo if you find it useful!
