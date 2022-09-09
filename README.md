# Tensor-CSPNet-and-Graph-CSPNet

In this repository, I implement two motor imagery-electroencephalography classifiers using geometric deep learning on symmetric positive definite manifolds. In essence, it is a deep learning-based MI-EEG classifier on the second-order statistics of EEG signals. 

We call these classes of approaches geometric methods. Thanks for geometers' great contributions in modern geometry and engineering disciplins so that we can formulate the world in a geometric perspective. 


Introduction
------------
This is the python implementation of Tensor-CSPNet and Graph-CSPNet.

1. Tensor-CSPNet: Tensor-CSPNet is the first geometric deep learning approach for the motor imagery-electroencephalography classification. It exploits the patterns from the time, spatial, and frequency domains sequentially. 
https://ieeexplore.ieee.org/document/9805775


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


Architecture 
------------

The mainstream of an effective MI-EEG classifier will exploit information from the time, spatial, and frequency domain. For spatial information, they both use BiMap-structure as the BiMap transofrmation in the CSP methods. For temporal and frequency information, their architectures vary on two approaches. Tensor-CSPNet uses CNNs for capturing the temporal dynamics, while Graph-CSPNet uses graph-based techniques for capturing information behind the time-frequency domains. 


Performance
------------



