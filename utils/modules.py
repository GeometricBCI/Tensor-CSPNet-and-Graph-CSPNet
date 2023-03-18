'''
#####################################################################################################################
Discription: 

This file constructs fundamental modules, encompassing the BiMap layer, the graph BiMap layer, the Riemannian Batch Normalization, 
the ReEig layer, and LogEig layer. 

These layers constitute the two geometric models (Tensor-CSPNet and Graph-CSPNet) found in model.py. 

There are two types of weight parameters for initialization as follows:

1. functional.StiefelParameter(th.empty(self._h, self._ni, self._no, dtype = dtype, device = device))

   In this instance, the parameter class is typically nn.Parameter. The backpropagations originate from the subsequent sources:

        BiMap/Graph BiMap <------  nn.Parameter; 

        BatchNormSPD      <------  nn.Parameter; 

2. geoopt.ManifoldParameter(th.empty(self._h, self._ni, self._no), manifold = ..)

     In this case, the parameter class is invoked from the "geoopt" package. Since the weights in BiMap and Riemannian Batch Normalization 
     reside on Stiefel and SPD manifolds, the backpropagations stem from the following sources:

        BiMap/Graph BiMap <------ geoopt.ManifoldParameter(,manifold=geoopt.CanonicalStiefel()); 

        BatchNormSPD      <------ geoopt.ManifoldParameter(,manifold=geoopt.SymmetricPositiveDefinite()) 

3. The objective of the SPDIncreaseDim() class is to augment the dimension of weights W, for instance, from (3, 3) to (5, 5), with the 
    expanding dimension being the identity matrix I_2.

#######################################################################################################################
'''

import torch as th
import torch.nn as nn
from torch.autograd import Function as F
from . import functional
import numpy as np
from . import geoopt

dtype =th.double
device=th.device('cpu')


class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):

        super(SPDIncreaseDim, self).__init__()

        self.register_buffer('eye', th.eye(output_size, input_size))

        add = np.asarray([0] * input_size + [1] * (output_size-input_size), dtype=np.float32)

        self.register_buffer('add', th.from_numpy(np.diag(add)))

    def forward(self, input):

        eye    = self.eye.unsqueeze(0).unsqueeze(0).double()

        eye    = eye.expand(input.size(0), input.size(1), -1, -1)

        add    = self.add.unsqueeze(0).unsqueeze(0).double()

        add    = add.expand(input.size(0), input.size(1), -1, -1)

        output = th.add(add, th.matmul(eye, th.matmul(input, eye.transpose(2,3))))

        return output


class BiMap(nn.Module):

    def __init__(self,h,ni,no):
        super(BiMap, self).__init__()

        self._h = h

        self.increase_dim = None

        if no <= ni:
            self._ni, self._no = ni, no
        else:
            self._ni, self._no = no, no
            self.increase_dim  = SPDIncreaseDim(ni,no)

        self._W =functional.StiefelParameter(th.empty(self._h, self._ni, self._no, dtype=dtype, device=device))
        #self._W = geoopt.ManifoldParameter(th.empty(self._h, self._ni, self._no), manifold=geoopt.CanonicalStiefel())

        self._init_bimap_parameter()

    def _init_bimap_parameter(self):

        for i in range(self._h):
            v  = th.empty(self._ni, self._ni, dtype = self._W.dtype, device = self._W.device).uniform_(0., 1.)
            vv = th.svd(v.matmul(v.t()))[0][:, :self._no]
            self._W.data[i] = vv

    def _bimap_multiplication(self, X):

        batch_size, channels_in, n_in, _ = X.shape

        P = th.zeros(batch_size, self._h, self._no, self._no, dtype = X.dtype, device = X.device)

        for c in range(self._h):
            P[:,c,:,:] = self._W[c, :, :].t().matmul(X[:,c,:,:]).matmul(self._W[c, :, :])

        return P

    def forward(self, X):

        if self.increase_dim:
            return self._bimap_multiplication(self.increase_dim(X))
        else:
            return self._bimap_multiplication(X)


class Graph_BiMap(nn.Module):

    def __init__(self,h,ni,no,P):
        super(Graph_BiMap, self).__init__()

        self._h = h
        self.increase_dim = None
        self._P = P

        if no <= ni:
            self._ni, self._no = ni, no
        else:
            self._ni, self._no = no, no
            self.increase_dim = SPDIncreaseDim(ni,no)

        self._W =functional.StiefelParameter(th.empty(self._h, self._ni, self._no, dtype=dtype,device=device))
        #self._W = geoopt.ManifoldParameter(th.empty(self._h, self._ni, self._no), manifold=geoopt.CanonicalStiefel())

        self._init_bimap_parameter()

    def _init_bimap_parameter(self):

        for i in range(self._h):
            v  = th.empty(self._ni, self._ni, dtype = self._W.dtype, device = self._W.device).uniform_(0.,1.)
            vv = th.svd(v.matmul(v.t()))[0][:, :self._no]
            self._W.data[i] = vv

    def _bimap_multiplication(self, X):

        batch_size, channels_in, n_in, _ = X.shape

        P = th.zeros(batch_size, self._h, self._no, self._no, dtype = X.dtype, device = X.device)

        for c in range(self._h):
            P[:,c,:,:] = self._W[c, :, :].t().matmul(X[:,c,:,:]).matmul(self._W[c, :, :])

        return P

    def forward(self, X):
        batch_size, channel_num, dim = X.shape[0], X.shape[1], X.shape[-1]

        if self.increase_dim:
            return self._bimap_multiplication(self.increase_dim(th.matmul(self._P, X.reshape((batch_size, channel_num, -1))).reshape((batch_size, channel_num, dim, dim))))
        else:
            return self._bimap_multiplication(th.matmul(self._P, X.reshape((batch_size, channel_num, -1))).reshape((batch_size, channel_num, dim, dim)))


class BatchNormSPD(nn.Module):

    def __init__(self, momentum, n):
        super(__class__, self).__init__()

        self.momentum     = momentum

        self.running_mean = geoopt.ManifoldParameter(th.eye(n, dtype=th.double),
                                               manifold=geoopt.SymmetricPositiveDefinite(),
                                               requires_grad=False
                                               )
        self.weight       = geoopt.ManifoldParameter(th.eye(n, dtype=th.double),
                                               manifold=geoopt.SymmetricPositiveDefinite(),
                                               )
        
    def forward(self,X):

        N, h, n, n  = X.shape

        X_batched   = X.permute(2, 3, 0, 1).contiguous().view(n, n, N*h, 1).permute(2, 3, 0, 1).contiguous()

        if(self.training):

            mean = functional.BaryGeom(X_batched)

            with th.no_grad():
                self.running_mean.data = functional.geodesic(self.running_mean, mean, self.momentum)

            X_centered = functional.CongrG(X_batched, mean, 'neg')

        else:
            X_centered = functional.CongrG(X_batched, self.running_mean, 'neg')

        X_normalized   = functional.CongrG(X_centered, self.weight, 'pos')

        return X_normalized.permute(2,3,0,1).contiguous().view(n,n,N,h).permute(2,3,0,1).contiguous()

class ReEig(nn.Module):
    def forward(self,P):
        return functional.ReEig.apply(P)

class LogEig(nn.Module):
    def forward(self,P):
        return functional.LogEig.apply(P)


