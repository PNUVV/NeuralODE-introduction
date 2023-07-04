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

- [[Journal paper] Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- [[Presentation video] Neural Ordinary Differential Equations](https://www.youtube.com/watch?v=V6nGT0Gakyg&ab_channel=AIPursuitbyTAIR)
- [[Github] Tutorials and NeuralODE variants](https://github.com/DiffEqML/torchdyn/tree/master/tutorials)
- [[Journal paper] Neural Controlled Differential Equations for Irregular Time Series](https://arxiv.org/abs/2005.08926)
- [[Presentation video] Neural Controlled Differential Equations for Irregular Time Series](https://www.youtube.com/watch?v=sbcIKugElZ4&ab_channel=PatrickKidger)

### Other Time series forecasting models

- [[Article] The Best Deep Learning Models for Time Series Forecasting](https://towardsdatascience.comthe-best-deep-learning-models-for-time-series-forecasting-690767bc63f0)
- [[Article] DeepAR: Mastering Time-Series Forecasting with Deep Learning](https://towardsdatascience.com/deepar-mastering-time-series-forecasting-with-deep-learning-bc717771ce85)
- [[Article] DeepAR: Mastering Time-Series Forecasting with Deep Learning](https://towardsdatascience.com/deepar-mastering-time-series-forecasting-with-deep-learning-bc717771ce85)
- [[Python library] PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/)
- [[Article] Time Series Forecasting with Neural Ordinary Differential Equations](https://towardsdatascience.com/time-series-forecasting-with-neural-ordinary-differential-equations-ff3c7a90a75e)

### Marine vehicles hydrodynamics and related machine learning applications

- [[Book chapter] Autonomous Underwater Vehicle Dynamics](https://www.researchgate.net/publication/290363811_Autonomous_Underwater_Vehicle_Dynamics): Top priority
- [[Book] HANDBOOK OF MARINE CRAFT HYDRODYNAMICS AND MOTION CONTROL](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138)
- [[Book] Modelling and Control of Dynamic Systems Using Gaussian Process Model](https://link.springer.com/book/10.1007/978-3-319-21021-6)
- [[Journal paper] Nonparametric modeling of ship maneuvering motion based on self-designed fully connected neural network](https://doi.org/10.1016/j.oceaneng.2022.111113)
- [[Journal paper] Identification and Prediction of Ship Maneuvering Motion Based on a Gaussian Process with Uncertainty Propagation](https://doi.org/10.3390/jmse9080804)
- [[Journal paper] System identification of ship dynamic model based on Gaussian process regression with input noise](https://doi.org/10.1016/j.oceaneng.2020.107862)

### Tips for reading articles on `Medium` or `towardsdatascience` without subscription

- Clear the cookies of your tab browser when you reach the limit of reading articles

## When using venv in WSL2 or Linux

- [issue] Expected numpy array as input but given PyTorchTensors causing error

plot_2D_space_depth이 텐서를 받아들이지만 int 함수를 적용하려고 하여 발생하는것 같습니다.
따라서 아래 수식으로 lib의 torchdyn/utils.py 코드를 변경하였습니다.

### Usage

Here is how to use the `plot_2D_space_depth` function to plot 2D trajectories in a 3D space:

```python
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_2D_space_depth(s_span, trajectory, yn, n_lines):
    "Plots 2D trajectories in a 3D space (2 dimensions of the system + time)."
    # Convert the Tensors to numpy arrays if necessary
    s_span = s_span.detach().cpu().numpy() if torch.is_tensor(s_span) else s_span
    trajectory = trajectory.detach().cpu().numpy() if torch.is_tensor(trajectory) else trajectory
    yn = yn.detach().cpu().numpy() if torch.is_tensor(yn) else yn

    colors = ['orange', 'blue']
    fig = plt.figure(figsize=(6,3))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    for i in range(n_lines):
        ax.plot(s_span, trajectory[:,i,0], trajectory[:,i,1], color=colors[yn[i].astype(int)], alpha = .1)
        ax.view_init(30, -110)

    ax.set_xlabel(r"$s$ [Depth]")
    ax.set_ylabel(r"$h_0$")
    ax.set_zlabel(r"$h_1$")
    ax.set_title("Flows in the space-depth")
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

# Create some example data
s_span = torch.linspace(0, 10, 100)
trajectory = torch.rand((100, 5, 2))  # 100 time steps, 5 lines, 2 dimensions
yn = torch.randint(0, 2, (5,))  # 5 lines

# Call the function
plot_2D_space_depth(s_span, trajectory, yn, len(trajectory[0]))

```
