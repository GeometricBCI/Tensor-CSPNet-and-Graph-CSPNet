I would like to call the following novel direction the geometric method in MI-EEG classification. Thanks for the geometers' great contributions so that we can formulate the world from a geometric perspective. I am also thankful for their great contributions in engineering disciplines, including geometric control theory, Riemannian optimization, geometric statistics, etc.

# Geometric Method: Tensor-CSPNet and Graph-CSPNet

In this repository, I implement two motor imagery-electroencephalography (MI-EEG) classifiers using geometric deep learning on symmetric positive definite manifolds. In essence, it is a deep learning-based MI-EEG classifier on the second-order statistics of EEG signals. In contrast to first-order statistics, using these second-order statistics is the classical treatment in MI-EEG classification, and the discriminative information contained in these second-order statistics is adequate for classification.




## Introduction

The mainstream of an effective MI-EEG classifier will exploit information from the time, spatial, and frequency domains. For spatial information, they both use BiMap-structure as the BiMap transformation in the CSP methods. For temporal and frequency information, their architectures vary in two approaches. Tensor-CSPNet uses CNNs for capturing the temporal dynamics, while Graph-CSPNet uses graph-based techniques for capturing information behind the time-frequency domains. In the following table, we have a detailed comparision of two methods. 



| Geometric Methods     | Tensor-CSPNet       |Graph-CSPNet   |
| ---------------------- | ------------- | ------------- |
| 1.Network Input:          | Tensorized Spatial Covariance Matrices         | Time-frequency Graph  |
| 2.Architecture:           | Mixed Geometric Deep Learning:         | Geometric Deep Learning:  |
|                        | BiMaps; CNNs                           | Graph-BiMaps |
| 3.Optimizer:             | Riemannian (Adaptive) Optimization     | Riemannian (Adaptive) Optimization|
|4.Underlying Space:|SPD Manifolds| SPD Manifolds|
|5.Heritage:|Common Spatial Patterns|Common Spatial Patterns; Riemannian-based Approaches|
|6.Principle:|The Time-Space-Frequency Principle: Exploitation in the frequency, space, and time domains sequentially.|The Time-Space-Frequency Principle: Exploitation in the time-frequency domain simultaneously, and then in the space domain.|
|7.Mechanism:|Trainable-parameter CNNs for temporal dynamics|Preset graph weights in spectral clustering for time-frequency distributions|
   

### Tensor-CSPNet

Tensor-CSPNet is the first geometric deep learning approach for the motor imagery-electroencephalography classification. It exploits the patterns from the time, spatial, and frequency domains sequentially. This is implementation of my paper [**Tensor-CSPNet: A Novel Geometric Deep Learning Framework for Motor Imagery Classification**](https://ieeexplore.ieee.org/document/9805775) accepted by IEEE Transactions on Neural Networks and Learning Systems (IEEE TNNLS). 

![Illustration of Tensor-CSPNet](img/Tensor_CSPNet_v100.png)


If you want to cite Tensor-CSPNet, please kindly add this bibtex entry in references and cite. It is now early accessed in IEEE TNNLS.
        
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


    If you want to cite Graph-CSPNet, please kindly add this bibtex entry in references and cite. It has been submitted. 
    
   
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


## Usages

I provide the models under `/utils/model` inside there we have Tensor-CSPNet and Graph-CSPNet. The modules of the two models can be found in `/utils/modules`. There are two scenarios, i.e., cross-validation and holdout, in the experiments. The train-test split index sets for the cross-validation scenario are put in the `/index/KU_index` and `/index/BCIC_index`. Since the validation set is not always necessary in a relatively small size of training trials, we will not divide the validation set from the whole dataset in the beginning. In the training file, we code a train-validation-test option which will be directly subdivided from the training set. This yields a little randomness in the result. I also wrote all of the training files for two scenarios, two algorithms, and two datasets in both .py and .ipynb files in `/training_files/`. Please put your downloaded data into the folder `/dataset/`, take the training file from the folder out, and then train and test your model. 

Keep in mind that with the given network architecture, training parameters, and training-test indices in the folder, Tensor-CSPNet and Graph-CSPNet should achieve around 76% on the CV scenario and around 72% on the holdout scenario on the BCIC-IV-2a dataset, and around 73% on the CV scenario and around 69% on the holdout scenario on the KU dataset using my local computer with CPU. It is normal to have a bit of randomness in each run. The classification performance of Tensor-CSPNet and Graph-CSPNet is near optimal during the deep learning age. Please try more combinations (signal segmentation, training parameters, etc.) for better performance in each scenario. Both Tensor-CSPNet and Graph-CSPNet use matrix backpropagation for updating weights in each layer, which runs a little slower in each epoch than that of typical CNNs, but 50 epochs in total in a run probably yields relatively good performance. For other MI-BCI datasets, it is suggested to try a novel segmentation plan that characterizes the region of interest associated with the task. Please modify the related classes in `/utils/load_data`.

There are two kinds of optimizers that we provide in this folder. One is called Class MixOptimizer given in `/utils/functional`; The other is from the python package geoopt. In the implementation, I found the best classification performance is initialized parameters from the BiMap layer with nn.Parameter and parameters from Riemannian Batch Normalization with geoopt.ManifoldParameter(..., manifold=geoopt.SymmetricPositiveDefinite()) using Riemannian adaptive optimizer in geoopt for our experiments. This might not always be right in other tasks.   


### Related Repositories

We list the repositories of related python packages and baseslines as follows, 

+ FBCSP:https://fbcsptoolbox.github.io/
+ FBCNet:https://github.com/ravikiran-mane/FBCNet
+ EEGNet:https://github.com/vlawhern/arl-eegmodels
+ pyRiemann:https://github.com/pyRiemann/pyRiemann
+ SPDNet:https://github.com/zhiwu-huang/SPDNet
+ torchspdnet:https://gitlab.lip6.fr/schwander/torchspdnet
+ geoopt:https://github.com/geoopt/geoopt

### Data Availability

The KU dataset (a.k.a., the OpenBMI dataset) can be downloaded in the following link:
[**GIGADB**](http://gigadb.org/dataset/100542)
with the dataset discription [**EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy**](https://academic.oup.com/gigascience/article/8/5/giz002/5304369).

The BCIC-IV-2a dataset can be downloaded in the following link:
[**BNCI-Horizon-2020**](http://bnci-horizon-2020.eu/database/data-sets)
with the dataset discription [**BCI Competition 2008 – Graz data set A**](https://www.bbci.de/competition/iv/desc_2a.pdf) and the introduction to [**the BCI competition**](https://www.bbci.de/competition/iv/).

### License and Attribution

Please refer to the LICENSE file for the licensing of our code.

