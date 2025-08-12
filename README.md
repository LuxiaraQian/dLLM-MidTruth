



# <img src="assets/logo.png" width="8%" alt="" align=center /> Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-green)](#)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://aim-uofa.github.io/dLLM-MidTruth/)
[![Code](https://img.shields.io/badge/Code-GitHub-orange)](#)
[![License](https://img.shields.io/badge/License-BSD%202--clause-lightgrey)](https://opensource.org/license/bsd-2-clause)

## 📣 News

- [2025-08-13] Paper Released!

## 🚀 Overview

<div align="center" >
<img src="assets/demo.png"/>
</div>

## 📖 Description

Diffusion large language models (dLLMs) generate text through iterative denoising, yet current decoding strategies discard rich intermediate predictions in favor of the final output. Our work here reveals a critical phenomenon---<b>temporal oscillation</b>---where correct answers often emerge in the middle process, but are overwritten in later denoising steps.
To address this issue, we introduce two complementary methods that exploit temporal consistency: 
- **Temporal Self-Consistency Voting**, a training-free, test-time decoding strategy that aggregates predictions across denoising steps to select the most consistent output; 
- **Temporal Consistency Reinforcement**, a post-training method that uses Temporal Semantic Entropy (TSE), a measure of semantic stability across intermediate predictions, as a reward signal to encourage stable generations. 


<!-- ## Getting Started

```
todo: file structure
```

### Setup

### Temporal Majority Voting

### Temporal Consistency Reinforcement

### Evaluation -->

## 🚩 Plan
- [ ] source code of temporal self-consistency voting and evaluation
- [ ] source code of temporal consistency reinforcement

## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).

## 🖊️ Citation

If you find this work useful, please consider citing:

```bibtex 
todo
```