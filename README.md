# Quantum Temporal Convolutional Network (QTCN)

This repository contains a quantum implementation of [the Temporal Convolutional Network (TCN)](https://arxiv.org/abs/1803.01271) using PennyLane and PyTorch.
The quantum model implements the temporal convolutional component of TCNs with quantum dilated convolutional neural network (QDCNN).  

## Benchmarking Datasets

The benchmarking datasets used in this work are selected to represent different levels of data complexity : 

  - **The Adding Problem** with various T (we evaluated on T=200, 400, 600)
  - **Sequential MNIST** digit classification
  - **JSB Chorales** polyphonic music
  - **Nottingham** polyphonic music


## Usage

The repository respects the **same file organization as the original TCN [repository](https://github.com/locuslab/TCN)** to make exploration and comparison easier :

```
[TASK_NAME] /
    data/
    experiments.py
    utils.py
QTCN.py
```


