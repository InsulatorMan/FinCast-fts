# FinCast: A Foundation Model for Financial Time-Series Forecasting

[![Paper](https://img.shields.io/badge/Paper-CIKM%202025-blue)](paper.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange)]()

This repository contains the official implementation of **FinCast**, introduced in our paper:

> **FinCast: A Foundation Model for Financial Time-Series Forecasting**  
> Zhuohang Zhu, Haodong Chen, Qiang Qu, Vera Chung  
> *CIKM 2025* (Accepted)  
> Arxiv link: https://arxiv.org/abs/2508.19609

FinCast is a **decoder-only transformer** trained on over **20B financial time points** across diverse domains and temporal resolutions.  
Technical Highlights:
- **PQ-Loss**: Joint point + probabilistic forecasting.
- **Mixture-of-Experts (MoE)**: Specialization across domains.

---

## 🔥 Features
- Foundation model for **financial time-series forecasting**, flexible input and output length.
- Strong performance in **zero-shot**, **supervised**, and **few-shot** settings.
- Modular architecture with **MoE** and **quantile-aware loss**.
- Scalable to **billions of parameters** with efficient inference.

---

## 📦 Installation

- The model weight can be found on 🤗 https://huggingface.co/Vincent05R/FinCast
- The model code can be found on https://github.com/vincent05r/FinCast-fts
- The corresponding datasets to reproduce the results can be found on https://huggingface.co/datasets/Vincent05R/FinCast-Paper-test

Run the env_setup.sh first then run the dep_install.sh. We use conda as the venv management.

## 📊 Experiments

- run the corresponding scripts in the scripts directory to reproduce the results in the paper. The result summary can be generate using the result summary notebook in the notebook directory.



## ⚡ Future Updates

- PEFT finetune(LORA/DORA) is done, just need to do some testing
- Package together for easy inference
- Covariate Inference(currently implemented the same code as timesfm)