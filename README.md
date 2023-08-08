This repository includes a Python implementation of Tensor-CSPNet and Graph-CSPNet, which are two classifiers for motor imagery-electroencephalography (MI-EEG). 

My current academic goal is to further combine the decoding of neural signals with mainstream techniques in theoretical mathematics. Grateful for my background in theoretical mathematics and physics, as well as years of training, I have chosen this path to pay tribute to the contributions of theoretical and applied geometers in the past century. We refer to this category of approaches as the **Geometric MI-BCI Classifier** and aim to continue its development by incorporating additional techniques and perspectives from fields such as differential geometry, information geometry, Riemannian optimization, geometric statistics, geometric control theory, manifold learning, and geometric deep learning. What is exciting is that these classifiers have indeed performed well in this small engineering task. 


# Tensor-CSPNet and Graph-CSPNet

## Introduction

In line with the event-related desynchronization and event-related synchronization phenomenon, the commonly used motor imagery-electroencephalography (MI-EEG) classifiers rely on extracting discriminative information from time, spatial, and frequency domains. In this regard, both Tensor-CSPNet and Graph-CSPNet employ BiMap network structures, akin to CSP methods, to capture spatial information. However, their architectures differ in the approaches used to capture temporal and frequency information. Tensor-CSPNet utilizes convolutional neural networks to capture temporal dynamics, while Graph-CSPNet leverages graph-based techniques to capture information underlying the time-frequency domains. A detailed comparison of the two methods is presented in the following table.


| Geometric Methods     | Tensor-CSPNet       |Graph-CSPNet   |
| ---------------------- | ------------- | ------------- |
| Network Input:          | Tensorized Spatial Covariance Matrices.         | Time-frequency Graph.  |
| Architecture:           | Geometric Deep Learning:BiMaps; CNNs.  | Geometric Deep Learning: Graph-BiMaps. |
|Distinctive Structure:|CNNs for temporal dynamics.|Spectral clustering for time-frequency distributions.|
|Training Optimizer:             | Riemannian Adaptive Optimization.     | Riemannian Adaptive Optimization.|
|Underlying Space:|SPD Manifolds with Riemannian metric AIRM<sup>*</sup>.| SPD Manifolds with Riemannian metric AIRM.|
|Methodology Heritage:|Common Spatial Patterns.|Common Spatial Patterns; Riemannian-based Approaches.|
|Design Principle:|The Time-Space-Frequency Principle: Exploitation in the frequency, space, and time domains sequentially.|The Time-Space-Frequency Principle and the Principle of Time-Frequency: Exploitation in the time-frequency domain simultaneously, and then in the space domain.|

AIRM<sup>*</sup> stands for Affine Invariant Riemannian Metric. It is the most commonly used metric for SPD manifolds and was also the Riemannian metric first used by Alexandre Barachant in his EEG-BCI classifier. In our implementation, Class BatchNormSPD involves parallel transportation, which requires a specific metric to be determined. LogEig also require a metric, but we have simplified the process that we compute it at identity since SPD manifolds are geodesically complete. Different metrics can have some impact on the results, but the degree of impact is much lower than the impact of different network structures using the same metric. Essentially, no metric is specifically designed for this classification problem, so we choose the most commonly used one.

### Graph-CSPNet

[<img src="https://img.shields.io/badge/arXiv-2211.02641-b31b1b"></img>](https://arxiv.org/abs/2211.02641)

Graph-CSPNet uses graph-based techniques to simultaneously characterize the EEG signals in both the time and frequency domains. It exploits the time-frequency domain simultaneously, and then in the space domain. 

![Illustration of Graph-CSPNet](graph_CSPNet.pdf)

    If you want to cite Graph-CSPNet, please kindly add this bibtex entry in references and cite. 
    
        @article{ju2022graph,
          title={Graph Neural Networks on SPD Manifolds for Motor Imagery Classification: A Perspective from the Time-Frequency Analysis},
          author={Ju, Ce and Guan, Cuntai},
          journal={arXiv preprint arXiv:2211.02641},
          year={2022}
        }


### Tensor-CSPNet

[<img src="https://img.shields.io/badge/IEEE-9805775-b31b1b"></img>](https://ieeexplore.ieee.org/document/9805775)
[<img src="https://img.shields.io/badge/arXiv-2202.02472-b31b1b"></img>](https://arxiv.org/abs/2202.02472)

Tensor-CSPNet is the first geometric deep learning approach for the motor imagery-electroencephalography classification. It exploits the patterns from the time, spatial, and frequency domains sequentially. 

![Illustration of Tensor-CSPNet](Tensor_CSPNet.png)

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
          

### Timeline of Related Works

#### Before 2020

Tensor-CSPNet and Graph-CSPNet are deep learning classifiers that operate on the second-order statistics of EEG signals. In contrast to the utilization of first-order statistics, the use of second-order statistics is a classical treatment in MI-EEG classification. The discriminative information present in these second-order statistics is considered sufficient for effective classification. These classifiers are based on the Riemannian geometry perspective in BCIs that has been developed over the past decade by prominent researchers such as Alexandre Barachant, Marco Congedo, Christian Jutten, Florian Yger, among others. Tensor-CSPNet and Graph-CSPNet leverage modern second-order neural networks on SPD manifolds that were developed by Zhiwu Huang, Luc Van Gool, Daniel Brooks, Olivier Schwander, Frederic Barbaresco, and others. We extend our sincere gratitude to these pioneers for their remarkable contributions in the development of the fundamental tools and contributing perspectives for this path.


#### 2020
1. C. Ju, D. Gao, R. Mane, B. Tan, Y. Liu and C. Guan, "Federated Transfer Learning for EEG Signal Classification," 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society, 2020, pp. 3040-3045, doi: 10.1109/EMBC44109.2020.9175344. (**EMBC2020**)
#### 2021
1. Suh, Yoon-Je, and Byung Hyung Kim. "Riemannian embedding banks for common spatial patterns with eeg-based spd neural networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 1. 2021. (**AAAI2021**)
#### 2022
1. C. Ju and C. Guan, “Tensor-cspnet: A novel geometric deep learning framework for motor imagery classification,” IEEE Transactions on Neural Networks and Learning Systems, 2022, pp. 1–15, doi: 10.1109/TNNLS.2022.3172108.(**TNNLS2022**)
2. C. Ju and C. Guan, “Deep optimal transport for domain adaptation on spd manifolds,” arXiv preprint arXiv:2201.05745, 2022, in review.
3. R. J. Kobler, J.-i. Hirayama, and M. Kawanabe, “Controlling the fréchet variance improves batch normalization on the symmetric positive definite manifold,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2022, pp. 3863–3867. (**ICASSP2022**)
4. R. J. Kobler, J.-i. Hirayama, Q. Zhao, and M. Kawanabe, "SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG," accepted by **NeurIPS2022**. 
5. Y.-T.Pan, J.-L.Chou, and C.-S.Wei, “Matt:Amanifoldattentionnetwork for eeg decoding,” accpeted by **NeurIPS2022**.
6. C. Ju and C. Guan, "Graph Neural Networks on SPD Manifolds for Motor Imagery Classification: A Perspective from the Time-Frequency Analysis," arXiv preprint arXiv:2211.02641, 2022, in review.
7. Daniel Wilson, Robin Tibor Schirrmeister, Lukas Alexander Wilhelm Gemein, Tonio Ball, "Deep Riemannian Networks for EEG Decoding", arXiv preprint arXiv:2212.10426, 2022.

#### 2023
1. C. Ju, R.J. Kobler, & C. Guan, Score-based Data Generation for EEG Spatial Covariance Matrices: Towards Boosting BCI Performance. 2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society, 2023, arXiv preprint arXiv:2302.11410 (**EMBC2023**) 
2. C. Bonet, B. Malézieux, A. Rakotomamonjy, L. Drumetz, T. Moreau, M. Kowalski, & N. Courty, Sliced-Wasserstein on Symmetric Positive Definite Matrices for M/EEG Signals. arXiv preprint arXiv:2303.05798, 2023, ICML. (**ICML2023**) 
3. Mostajeran, C., Da Costa, N., Van Goffrier, G., & Sepulchre, R. Differential geometry with extreme eigenvalues in the positive semidefinite cone. arXiv preprint arXiv:2304.07347, 2023, in review. 



### Related Repositories

We extend our gratitude to the open-source community, which facilitates the wider dissemination of the work of other researchers as well as our own. The coding style in this repo is relatively rough. We welcome anyone to refactor it to make it more effective. The codebase for our models builds heavily on the following repositories: 
[<img src="https://img.shields.io/badge/GitHub-FBCSP-b31b1b"></img>](https://fbcsptoolbox.github.io/)
[<img src="https://img.shields.io/badge/GitHub-FBCNet-b31b1b"></img>](https://github.com/ravikiran-mane/FBCNet)
[<img src="https://img.shields.io/badge/GitHub-pyRiemann-b31b1b"></img>](https://github.com/pyRiemann/pyRiemann)
[<img src="https://img.shields.io/badge/GitHub-SPDNet(Z.W.Huang)-b31b1b"></img>](https://github.com/zhiwu-huang/SPDNet)
[<img src="https://img.shields.io/badge/GitHub-SPDNet(LIP6)-b31b1b"></img>](https://gitlab.lip6.fr/schwander/torchspdnet)
[<img src="https://img.shields.io/badge/GitHub-geoopt-b31b1b"></img>](https://github.com/geoopt/geoopt)

Another implementation of SPD manifold-valued neural networks refers to Kobler's TSMNet [<img src="https://img.shields.io/badge/GitHub-TSMNet-b31b1b"></img>](https://github.com/rkobler/TSMNet). 

## Usages

The Tensor-CSPNet and Graph-CSPNet models are included in the `/utils/model` directory, with their respective modules located in `/utils/modules`. The experiments are conducted using two scenarios, namely, cross-validation and holdout, with two training files for Graph-CSPNet available for the BCIC dataset. After downloading the data, place it in the `/dataset/` folder, extract the training file, and then proceed to train and test your model. It is also necessary to create the `/model_paras/` and `/results/` folders to store the model/optimizer parameters and results, respectively.

Note that with the provided network architecture and training parameters in the folder, Tensor-CSPNet and Graph-CSPNet achieved approximately 76% accuracy on the CV scenario and 72% accuracy on the holdout scenario for the BCIC-IV-2a dataset, and approximately 73% accuracy on the CV scenario and 69% accuracy on the holdout scenario for the KU dataset on my local computer. These results represent the best performance after several runs, although some randomness may be present due to various computational issues. Overall, the classification performance of Tensor-CSPNet and Graph-CSPNet is near-optimal on the given scenarios. To achieve better performance, users are encouraged to try different hyperparameter combinations, network architectures, and training strategies.

Both Tensor-CSPNet and Graph-CSPNet use matrix backpropagation to update weights in each layer, which runs slightly slower than typical methods, but 50 epochs per run generally yields relatively good performance. For other MI-BCI datasets, it is recommended to develop a novel segmentation plan that characterizes the region of interest associated with the task. Users can modify the related classes in `/utils/load_data` accordingly. Several tips for training that may be helpful (or not) include: setting the initial learning rate to 1e-3; choosing a reasonably sized batch for training; running each epoch for at least 50 iterations in either scenario, and avoiding early stopping before 50 epochs. This is because EEG signals typically have high signal-to-noise ratios, and noise can force early stopping before patterns have been learned. In some cases, the validation process can be disabled to increase the number of trials for training in the CV scenario.

In particular, we recommend applying shrinkage regularization to the input matrices. Shrinkage regularization is a method used to estimate a positive definite matrix, which is a matrix that is symmetric and has all positive eigenvalues. This method is particularly useful when the number of observations is small relative to the number of variables, as this can lead to an unstable estimate of the SPD matrix.

We provide two optimizers in this folder: Class MixOptimizer in `/utils/functional`, and the geoopt package. In my implementation, initializing parameters from the BiMap layer with nn.Parameter and parameters from Riemannian Batch Normalization with geoopt has yielded the best classification performance. However, this may not always be the best approach for other tasks.

This code has been made available subsequent to the completion of the Graph-CSPNet paper. A few modifications have been introduced to the functional functions following the completion of the Tensor-CSPNet paper. To replicate the results presented in Table IV of the Graph-CSPNet paper, kindly adhere to the hyperparameters outlined in the same paper. The provided code corresponds to the particular hyperparameters detailed in the Graph-CSPNet paper.




### Data Availability

The KU dataset (a.k.a., the OpenBMI dataset) can be downloaded in the following link:
[**GIGADB**](http://gigadb.org/dataset/100542)
with the dataset discription [**EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy**](https://academic.oup.com/gigascience/article/8/5/giz002/5304369); The BCIC-IV-2a dataset can be downloaded in the following link:
[**BNCI-Horizon-2020**](http://bnci-horizon-2020.eu/database/data-sets)
with the dataset discription [**BCI Competition 2008 – Graz data set A**](https://www.bbci.de/competition/iv/desc_2a.pdf) and the introduction to [**the BCI competition**](https://www.bbci.de/competition/iv/).
All of this data can be accessed through the [**MOABB**](https://github.com/NeuroTechX/moabb). This package includes a benchmark dataset for advanced decoding algorithms, which comprises 12 open-access datasets and covers over 250 subjects.


### License and Attribution

Copyright 2022 S-Lab. All rights reserved.

Please refer to the LICENSE file for the licensing of our code.

