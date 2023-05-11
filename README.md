# README

## Introduction

### Theory and general introduction

- Refer to this [Alex Honchar's article](https://towardsdatascience.com/neural-odes-breakdown-of-another-deep-learning-breakthrough-3e78c7213795) or `NeuralODE_Alex_Honchar.htlm`

### Tutorial

- Refer to `NeuralODE_tutorial.ipynb`

## Installation

### Jupyter notebook

- Install jupyter notebook: `pip install jupyter`
- Run jupyter notebook: `jupyter notebook`

### Pytorch and torchdyn with a GPU

    1. Create new env: conda create --name py38torch python=3.8
    2. Active env: conda activate py38torch
    4. Install torchdyn: pip install torchdyn==1.0.3
    5. Install torch: conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    ==> Install 4, 5 in the end to avoid packages overwritten

### Note

- Recommend to use [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to install packages for convenience

- If you dont have GPU, please refer to [Torch website](https://pytorch.org/get-started/locally/) for more details

## Recommendations for further reading

### NeuralODEs and its applications

- [[Journal paper] Neural Ordinary Differential Equations](<https://arxiv.org/abs/1806.07366>)
- [[Presentation video] Neural Ordinary Differential Equations](<https://www.youtube.com/watch?v=V6nGT0Gakyg&ab_channel=AIPursuitbyTAIR>)
- [[Github] Tutorials and NeuralODE variants](<https://github.com/DiffEqML/torchdyn/tree/master/tutorials>)
- [[Journal paper] Neural Controlled Differential Equations for Irregular Time Series](<https://arxiv.org/abs/2005.08926>)
- [[Presentation video] Neural Controlled Differential Equations for Irregular Time Series](<https://www.youtube.com/watch?v=sbcIKugElZ4&ab_channel=PatrickKidger>)

### Other Time series forecasting models

- [[Article] The Best Deep Learning Models for Time Series Forecasting](<https://towardsdatascience.comthe-best-deep-learning-models-for-time-series-forecasting-690767bc63f0>)
- [[Article] DeepAR: Mastering Time-Series Forecasting with Deep Learning](<https://towardsdatascience.com/deepar-mastering-time-series-forecasting-with-deep-learning-bc717771ce85>)
- [[Article] DeepAR: Mastering Time-Series Forecasting with Deep Learning](<https://towardsdatascience.com/deepar-mastering-time-series-forecasting-with-deep-learning-bc717771ce85>)
- [[Python library] PyTorch Forecasting](<https://pytorch-forecasting.readthedocs.io/en/stable/>)
- [[Article] Time Series Forecasting with Neural Ordinary Differential Equations](<https://towardsdatascience.com/time-series-forecasting-with-neural-ordinary-differential-equations-ff3c7a90a75e>)

### Marine vehicles hydrodynamics and related machine learning applications

- [[Book] HANDBOOK OF MARINE CRAFT HYDRODYNAMICS AND MOTION CONTROL](<https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>)
- [[Book] Modelling and Control of Dynamic Systems Using Gaussian Process Model](<https://link.springer.com/book/10.1007/978-3-319-21021-6>)
- [[Book chapter] Autonomous Underwater Vehicle Dynamics](<https://www.researchgate.net/publication/290363811_Autonomous_Underwater_Vehicle_Dynamics>): Top priority
- [[Journal paper] Nonparametric modeling of ship maneuvering motion based on self-designed fully connected neural network](<https://doi.org/10.1016/j.oceaneng.2022.111113>)
- [[Journal paper] Identification and Prediction of Ship Maneuvering Motion Based on a Gaussian Process with Uncertainty Propagation](<https://doi.org/10.3390/jmse9080804>)
- [[Journal paper] System identification of ship dynamic model based on Gaussian process regression with input noise](<https://doi.org/10.1016/j.oceaneng.2020.107862>)


### Tips for reading articles on `Medium` or `towardsdatascience` without subscription

- Clear the cookies of the website when you reach the limit of reading articles
