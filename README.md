We would like to call the following novel path the geometric method in motor imagery-electroencephalography (MI-EEG) classification. Because we hope to keep it alive by involving and sowing the seeds of more techniques and perspectives from differential geometry, information geometry, Riemannian optimization, geometric statistics, geometric control theory, manifold learning, geometric deep learning, etc. 

# Tensor-CSPNet and Graph-CSPNet

In this repository, two MI-EEG classifiers were implemented using geometric deep learning on symmetric positive definite manifolds. In essence, it is a deep learning-based MI-EEG classifier on the second-order statistics of EEG signals. In contrast to first-order statistics, using these second-order statistics is the classical treatment in MI-EEG classification, and the discriminative information contained in these second-order statistics is adequate for classification.

## Introduction

The mainstream of an effective MI-EEG classifier will exploit information from the time, spatial, and frequency domains. For spatial information, they both use BiMap-structure as the BiMap transformation in the CSP methods. For temporal and frequency information, their architectures vary in two approaches. Tensor-CSPNet uses CNNs for capturing the temporal dynamics, while Graph-CSPNet uses graph-based techniques for capturing information behind the time-frequency domains. In the following table, we have a detailed comparision of two methods. 

| Geometric Methods     | Tensor-CSPNet       |Graph-CSPNet   |
| ---------------------- | ------------- | ------------- |
| 1.Network Input:          | Tensorized Spatial Covariance Matrices         | Time-frequency Graph  |
| 2.Architecture:           | (Mixed) Geometric Deep Learning:         | Geometric Deep Learning:  |
|                        | BiMaps; CNNs                           | Graph-BiMaps |
| 3.Optimizer:             | Riemannian Adaptive Optimization     | Riemannian Adaptive Optimization|
|4.Underlying Space:|SPD Manifolds| SPD Manifolds|
|5.Heritage:|Common Spatial Patterns|Common Spatial Patterns; Riemannian-based Approaches|
|6.Principle:|The Time-Space-Frequency Principle: Exploitation in the frequency, space, and time domains sequentially.|The Time-Space-Frequency Principle: Exploitation in the time-frequency domain simultaneously, and then in the space domain.|
|7.Mechanism:|Trainable-parameter CNNs for temporal dynamics|Preset graph weights in spectral clustering for time-frequency distributions|

### Tensor-CSPNet

[<img src="https://img.shields.io/badge/IEEE-9805775-b31b1b"></img>](https://ieeexplore.ieee.org/document/9805775)
[<img src="https://img.shields.io/badge/arXiv-2202.02472-b31b1b"></img>](https://arxiv.org/abs/2202.02472)

Tensor-CSPNet is the first geometric deep learning approach for the motor imagery-electroencephalography classification. It exploits the patterns from the time, spatial, and frequency domains sequentially. 

![Illustration of Tensor-CSPNet](img/Tensor_CSPNet_v100.png)

If you want to cite Tensor-CSPNet, please kindly add this bibtex entry in references and cite. 
        
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
          
### Graph-CSPNet

Graph-CSPNet uses graph-based techniques to simultaneously characterize the EEG signals in both the time and frequency domains. It exploits the time-frequency domain simultaneously, and then in the space domain. 

![Illustration of Graph-CSPNet](img/graph_CSPNet.png)

    If you want to cite Graph-CSPNet, please kindly add this bibtex entry in references and cite. 
    
   
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

### Timeline of Related Works

#### Before 2020

Great thanks to all of the poineers for techniques in SPD manifolds and Riemannian-based classifiers in BCIs, including Xavier Pennec, Alexandre Barachant, Marco Congedo, Zhiwu Huang, et al. for their great contributions in developing the fundamental tools and perspectives for this path. 

#### 2020
1. C. Ju, D. Gao, R. Mane, B. Tan, Y. Liu and C. Guan, "Federated Transfer Learning for EEG Signal Classification," 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society, 2020, pp. 3040-3045, doi: 10.1109/EMBC44109.2020.9175344. (**EMBC2020**)
#### 2022
1. C. Ju and C. Guan, “Tensor-cspnet: A novel geometric deep learning framework for motor imagery classification,” IEEE Transactions on Neural Networks and Learning Systems, 2022, pp. 1–15, doi: 10.1109/TNNLS.2022.3172108.(**TNNLS2022**)
2. C. Ju and C. Guan, “Deep optimal transport for domain adaptation on spd manifolds,” arXiv preprint arXiv:2201.05745, 2022.
3. R. J. Kobler, J.-i. Hirayama, and M. Kawanabe, “Controlling the fréchet variance improves batch normalization on the symmetric positive definite manifold,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2022, pp. 3863–3867. (**ICASSP2022**)
4. R. J. Kobler, J.-i. Hirayama, Q. Zhao, and M. Kawanabe, "SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG," accepted by **NeurIPS2022**. [<img src="https://img.shields.io/badge/GitHub-TSMNet-b31b1b"></img>](https://github.com/rkobler/TSMNet)
5. Y.-T.Pan, J.-L.Chou, and C.-S.Wei, “Matt:Amanifoldattentionnetwork for eeg decoding,” accpeted by **NeurIPS2022**.
6. ...


### Related Repositories

We list the repositories of related python packages as follows: [<img src="https://img.shields.io/badge/GitHub-FBCSP-b31b1b"></img>](https://fbcsptoolbox.github.io/), [<img src="https://img.shields.io/badge/GitHub-FBCNet-b31b1b"></img>](https://github.com/ravikiran-mane/FBCNet), [<img src="https://img.shields.io/badge/GitHub-pyRiemann-b31b1b"></img>](https://github.com/pyRiemann/pyRiemann), [<img src="https://img.shields.io/badge/GitHub-SPDNet(Z.W.Huang)-b31b1b"></img>](https://github.com/zhiwu-huang/SPDNet), [<img src="https://img.shields.io/badge/GitHub-SPDNet(LIP6)-b31b1b"></img>](https://gitlab.lip6.fr/schwander/torchspdnet), and [<img src="https://img.shields.io/badge/GitHub-geoopt-b31b1b"></img>](https://github.com/geoopt/geoopt).

## Usages

The models are provided under `/utils/model` inside there we have Tensor-CSPNet and Graph-CSPNet. The modules of the two models can be found in `/utils/modules`. There are two scenarios, i.e., cross-validation and holdout, in the experiments. The train-test split index sets for the cross-validation scenario are put in the `/index/KU_index` and `/index/BCIC_index`. Since the validation set is not always necessary in a relatively small size of training trials, we will not divide the validation set from the whole dataset in the beginning. In the training file, we code a train-validation-test option which will be directly subdivided from the training set. This yields a little randomness in the result. I also wrote all of the training files for two scenarios, two algorithms, and two datasets in both .py files in `/training_files/`. Please build and put your downloaded data into the folder `/dataset/`, take the training file from the folder out, and then train and test your model. You need also build the folder `/model_paras/` and `/results/` to keep model/optimizer parameters and results, respectively.

Keep in mind that with the given network architecture, training parameters, and training-test indices in the folder, Tensor-CSPNet and Graph-CSPNet should achieve around 76% on the CV scenario and around 72% on the holdout scenario on the BCIC-IV-2a dataset, and around 73% on the CV scenario and around 69% on the holdout scenario on the KU dataset using my local computer. It is normal to have a bit of randomness in each run because of many computational issues. The classification performance of Tensor-CSPNet and Graph-CSPNet is near optimal on the given scenarios. For better performance, please try more hyperparamerter combinations in each run, i.e., signal segmentation, training parameters, etc. Both Tensor-CSPNet and Graph-CSPNet use matrix backpropagation for updating weights in each layer, which runs a little slower in each epoch than that of typical one, but 50 epochs in total in a run probably yields relatively good performance. For other MI-BCI datasets, it is suggested to try a novel segmentation plan that characterizes the region of interest associated with the task. Please modify the related classes in `/utils/load_data`. I have tested several runs with the given network and training parameters and strategies. The results look very close to the reported ones. Keep in mind that the reported results are the best that I achieve. Please try slightly modified hyperparameters to adapt your device if you cannot reproduce them. There are several tips in training probably helpful to follow:
Tip 1. Set the learning rate = 1e-3;
Tip 2. Pick a not so big batch size for training;
Tip 3. The epoch of each run should at least 50 in either scenario. So, don't use the early stopping before 50 epoch;
Tip 4. For the CV scenario, I sometimes to ban the validation process in order to increase the number of trials for training. 

There are two kinds of optimizers that we provide in this folder. One is called Class MixOptimizer given in `/utils/functional`; The other is from the python package geoopt. In my implementation, we found the best classification performance is initialized parameters from the BiMap layer with nn.Parameter and parameters from Riemannian Batch Normalization with geoopt. This might not always be a useful hand-on experience for other tasks. 

### Data Availability

The KU dataset (a.k.a., the OpenBMI dataset) can be downloaded in the following link:
[**GIGADB**](http://gigadb.org/dataset/100542)
with the dataset discription [**EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy**](https://academic.oup.com/gigascience/article/8/5/giz002/5304369); The BCIC-IV-2a dataset can be downloaded in the following link:
[**BNCI-Horizon-2020**](http://bnci-horizon-2020.eu/database/data-sets)
with the dataset discription [**BCI Competition 2008 – Graz data set A**](https://www.bbci.de/competition/iv/desc_2a.pdf) and the introduction to [**the BCI competition**](https://www.bbci.de/competition/iv/).

### License and Attribution

Copyright 2022 S-Lab, Nanyang Technological University. All rights reserved.

Please refer to the LICENSE file for the licensing of our code.

