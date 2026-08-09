# Quantum Temporal Convolutional Network (QTCN)

This repository contains the implementation of [the Quantum Temporal Convolutional Network (QTCN)](https://link.springer.com/chapter/10.1007/978-3-032-32335-4_20) using PennyLane and PyTorch.
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

To cite this work:

@inproceedings{hoceini2026quantum,
  title={Quantum Temporal Convolution Network},
  author={Hoceini, Rihab and Bouida, Ahmed},
  booktitle={German Conference on Artificial Intelligence (K{\"u}nstliche Intelligenz)},
  pages={244--251},
  year={2026},
  organization={Springer}
}
