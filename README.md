---
tags:
- autotrain
- text-generation
- finance
widget:
- text: 'I love AutoTrain because '
license: apache-2.0
datasets:
- AdiOO7/llama-2-finance
---

# Model Trained Using AutoTrain

This repository contains the code for training an advanced language model using the autotrain library from Hugging Face. The goal of this project is to fine-tune a pre-trained language model on financial data to improve its performance on downstream tasks such as sentiment analysis and named entity recognition.

## Installation


To use this repository, you will need to have Python installed with the following packages:
`
pip install autotrain-advanced
pip install huggingface_hub
`

## Training Data


The training data for this project consists of financial news articles scraped from various sources. These articles were selected based on their relevance to the stock market and other financial topics. The data was then cleaned and processed into a format suitable for training a language model.

## Model Architecture


The model architecture used for this project is a variant of the BERT (Bidirectional Encoder Representations from Transformers) model. Specifically, we used the TinyPixel/Llama-2-7B-bf16-sharded model, which has been optimized for finance-related tasks. This model uses a combination of wordpiece tokenization and positional encoding to represent input sequences.

## Hyperparameters


The hyperparameters adjusted for this project are listed below:

- Learning rate: 0.0002
- Train batch size: 4
- Number of epochs: 1
- Trainer: SFT (Stochastic Fine-Tuning)
- FP16 precision: True
- Maximum sequence length: 512

Other settings were set to default. These hyperparameters were chosen based on experience with similar projects and the characteristics of the training data. However, feel free to experiment with different values to see if they produce better results.

## Conclusion


In conclusion, this project demonstrates how to fine-tune a pre-trained language model on financial data using the autotrain library from Hugging Face. By modifying and adjusting the hyperparameters, you can tailor the training process to suit your specific needs.