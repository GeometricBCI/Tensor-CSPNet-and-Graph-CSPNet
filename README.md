# Tensor-CSPNet-and-Graph-CSPNet

In this repository, I implement two motor imagery-electroencephalography classifiers using geometric deep learning on symmetric positive definite manifolds. In essence, it is a deep learning-based MI-EEG classifier on the second-order statistics of EEG signals. 

We would like to call these novel classes of approaches geometric methods. Thanks for geometers' great contributions in modern geometry and engineering disciplins so that we can formulate the world in a geometric perspective. 


## Introduction

This is the python implementation of Tensor-CSPNet and Graph-CSPNet.

1. Tensor-CSPNet: Tensor-CSPNet is the first geometric deep learning approach for the motor imagery-electroencephalography classification. It exploits the patterns from the time, spatial, and frequency domains sequentially. This is implementation of our paper [**Tensor-CSPNet: A Novel Geometric Deep Learning Framework for Motor Imagery Classification**](https://ieeexplore.ieee.org/document/9805775) accepted by IEEE Transactions on Neural Networks and Learning Systems (IEEE TNNLS). 


    If you want to cite Tensor-CSPNet, please kindly add this bibtex entry in references and cite. It is now early accessed in IEEE TNNLS.
    
    ~~~~~~~~
        @ARTICLE{9805775,
            author={Ju, Ce and Guan, Cuntai},
            journal={IEEE Transactions on Neural Networks and Learning Systems}, 
            title={Tensor-CSPNet: A Novel Geometric Deep Learning Framework for Motor Imagery Classification}, 
            year={2022},
            volume={},
            number={},
            pages={1-15},
            doi={10.1109/TNNLS.2022.3172108}
          }
2. Graph-CSPNet: Graph-CSPNet uses graph-based techniques to simultaneously characterize the EEG signals in both the time and frequency domains. It exploits the time-frequency domain simultaneously, and then in the space domain. 


## Architecture and Usage

The mainstream of an effective MI-EEG classifier will exploit information from the time, spatial, and frequency domain. For spatial information, they both use BiMap-structure as the BiMap transofrmation in the CSP methods. For temporal and frequency information, their architectures vary on two approaches. Tensor-CSPNet uses CNNs for capturing the temporal dynamics, while Graph-CSPNet uses graph-based techniques for capturing information behind the time-frequency domains. 

We provide the models under `/utils/model/` inside there we have 9 subdirectories `/S1` to `/S9` each representing each subject. Inside each subdirectory there are 6 files. `model.h5` is the saved keras model of variable EEG-TCNet and `model_fixed.h5` is the saved keras model of fixed EEG-TCNet. Then there are two pipeline files in each subdirectory which vary depending on if data normalization was used or not. Please refer to  `Accuracy_and_kappa_scores.ipynb` in the main directory to see how these pipelines are produces. There you also find the accuracy score and kappa score verification of EEG-TCNet.

Under `/utils` you find the data loading and model making files. Then also a small sample of how to train is given with `sample_train.py`, please note that because of the stochastic nature of training with GPUs it's very hard to fix every random variable in the backend. Therefore to reproduce the same or similar models one might need to train a couple of times in order to get the same highly accurate models we present.


### Related Repositoroes

We list the repositoires of several modules of our model and baseslines in the experiments as follows, 

+ FBCSP:https://fbcsptoolbox.github.io/
+ FBCNet:https://github.com/ravikiran-mane/FBCNet
+ EEGNet:https://github.com/vlawhern/arl-eegmodels
+ pyRiemann:https://github.com/pyRiemann/pyRiemann
+ SPDNet:https://github.com/zhiwu-huang/SPDNet
+ torchspdnet:https://gitlab.lip6.fr/schwander/torchspdnet
+ geoopt:https://github.com/geoopt/geoopt

### Dataset Availability

KU Dataset (a.k.a., OpenBMI dataset) can be downloaded in the following link:
[**GIGADB**](http://gigadb.org/dataset/100542)
with the dataset discription [**EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy**](https://academic.oup.com/gigascience/article/8/5/giz002/5304369).

BCIC-IV-2a can be downloaded in the following link:
[**BNCI-Horizon-2020**](http://bnci-horizon-2020.eu/database/data-sets)
with the dataset discription [**BCI Competition 2008 – Graz data set A**](https://www.bbci.de/competition/iv/desc_2a.pdf) and the introduction to [**the BCI competition**](https://www.bbci.de/competition/iv/).

### License and Attribution
Please refer to the LICENSE file for the licensing of our code.

