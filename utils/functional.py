'''
#####################################################################################################################
Discription: 

The utility functions in this file offer the forward function for the ReEig layer, the LogEig layer, and Riemannian Batch 
Normalization in geometric models (Tensor-CSPNet and Graph-CSPNet). Additionally, they provide an optimizer for network 
architecture. The primary functions and classes are mainly derived from the following repository:

https://gitlab.lip6.fr/schwander/torchspdnet
https://github.com/adavoudi/spdnet
https://github.com/zhiwu-huang/SPDNet
https://github.com/YirongMao/SPDNet

#######################################################################################################################
'''
import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Function as F

import torch.optim
from . import functional


class MixOptimizer():
    """ Optimizer with mixed constraints """

    def __init__(self, parameters, optimizer=torch.optim.Adam, lr=1e-2, *args, **kwargs):
        parameters = list(parameters)
        parameters =[param for param in parameters if param.requires_grad]
        self.lr = lr
        self.stiefel_parameters = [param for param in parameters if param.__class__.__name__=='StiefelParameter']
        self.spd_parameters     = [param for param in parameters if param.__class__.__name__=='SPDParameter']
        self.other_parameters   = [param for param in parameters if param.__class__.__name__=='Parameter']
        self.optim              = optimizer(self.other_parameters, lr, *args, **kwargs)

    def _StiefelOptim_step(self):
        for W in self.stiefel_parameters:
            dir_tan=proj_tanX_stiefel(W.grad.data,W.data)
            W_new  =ExpX_stiefel(-self.lr*dir_tan.data, W.data)
            W.data =W_new

    def _SPDOptim_step(self):
        for W in self.spd_parameters:
            dir_tan=proj_tanX_spd(W.grad.data,W.data)
            W.data=functional.ExpG(-self.lr*dir_tan.data,W.data)[0,0]

    def step(self):
        self.optim.step()
        self._StiefelOptim_step()
        self._SPDOptim_step()

    def _StiefelOptim_zero_grad(self):
        for p in self.stiefel_parameters:
            if p.grad is not None:
                p.grad.data.zero_()
    def _SPDOptim_zero_grad(self):
        for p in self.spd_parameters:
            if p.grad is not None:
                p.grad.data.zero_()
    def zero_grad(self):
        self.optim.zero_grad()
        self._StiefelOptim_zero_grad()
        self._SPDOptim_zero_grad()
        

def proj_tanX_stiefel(x,X):
    """ Projection of x in the Stiefel manifold's tangent space at X """
    return x-X.matmul(x.transpose(-2,-1)).matmul(X)

def ExpX_stiefel(x,X):
    """ Exponential mapping of x on the Stiefel manifold at X (retraction operation) """
    a=X+x
    Q=th.zeros_like(a)
    for i in range(a.shape[0]):
        q,_=gram_schmidt(a[i])
        Q[i]=q
    return Q

def proj_tanX_spd(x,X):
    """ Projection of x in the SPD manifold's tangent space at X """
    return X.matmul(sym(x)).matmul(X)

#V is a in M(n,N); output W an semi-orthonormal free family of Rn; we consider n >= N
#also returns R such that WR is the QR decomposition
def gram_schmidt(V):
    n,N=V.shape #dimension, cardinal
    W=th.zeros_like(V)
    R=th.zeros((N,N)).double().to(V.device)
    W[:,0]=V[:,0]/th.norm(V[:,0])
    R[0,0]=W[:,0].dot(V[:,0])
    for i in range(1,N):
        proj=th.zeros(n).double().to(V.device)
        for j in range(i):
            proj=proj+V[:,i].dot(W[:,j])*W[:,j]
            R[j,i]=W[:,j].dot(V[:,i])
        if(isclose(th.norm(V[:,i]-proj),th.DoubleTensor([0]).to(V.device))):
            W[:,i]=V[:,i]/th.norm(V[:,i])
        else:
            W[:,i]=(V[:,i]-proj)/th.norm(V[:,i]-proj)
        R[i,i]=W[:,i].dot(V[:,i])
    return W,R

def isclose(a,b,rtol=1e-05, atol=1e-08):
    return ((a - b).abs() <= (atol + rtol * b.abs())).all()

def sym(X):
    if(len(X.shape)==2):
        if isinstance(X,np.ndarray):
            return 0.5*(X+X.T.conj())
        else:
            return 0.5*(X+X.t())
    elif(len(X.shape)==3):
        if isinstance(X,np.ndarray):
            return 0.5*(X+X.transpose([0,2,1]))
        else:
            return 0.5*(X+X.transpose(1,2))
    elif(len(X.shape)==4):
        if isinstance(X,np.ndarray):
            return 0.5*(X+X.transpose([0,1,3,2]))
        else:
            return 0.5*(X+X.transpose(2,3))


class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass

class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass

def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape #batch size,channel depth,dimension
    U, S = th.zeros_like(P, device=P.device), th.zeros(batch_size, channels, n, dtype=P.dtype, device=P.device)

    for i in range(batch_size):
        for j in range(channels):
            if(eig_mode=='eig'):
                #This is for v_pytorch >= 1.9;
                s, U[i, j] = th.linalg.eig(P[i,j][None,:])
                S[i, j]    = s[:,0]
            elif(eig_mode=='svd'):
                U[i,j], S[i,j], _ = th.svd(P[i,j])

    S_fn = op.fn(S, param)
    X    = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2,3))
    return X, U, S, S_fn

def modeig_backward(dx,U,S,S_fn,op,param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    S_fn_deriv    = BatchDiag(op.fn_deriv(S,param))
    SS            = S[...,None].repeat(1,1,1,S.shape[-1])
    SS_fn         = S_fn[...,None].repeat(1,1,1,S_fn.shape[-1])
    L             = (SS_fn-SS_fn.transpose(2,3))/(SS-SS.transpose(2,3))
    L[L==-np.inf] = 0 
    L[L==np.inf]  = 0 
    L[th.isnan(L)]= 0
    L             = L + S_fn_deriv
    dp            = L * (U.transpose(2,3).matmul(dx).matmul(U))
    dp            = U.matmul(dp).matmul(U.transpose(2,3))
    return dp

class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Log_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Log_op)

class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Re_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Re_op)


class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Exp_op,eig_mode='eig')
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Exp_op)

class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Sqm_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Sqm_op)


class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Sqminv_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Sqminv_op)


class PowerEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P,power):
        Power_op._power=power
        X,U,S,S_fn=modeig_forward(P,Power_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Power_op), None


def geodesic(A,B,t):
    """
    Geodesic from A to B at step t
    :param A: SPD matrix (n,n) to start from
    :param B: SPD matrix (n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPD matrix (n,n) along the geodesic
    """
    M=CongrG(PowerEig.apply(CongrG(B,A,'neg'),t),A,'pos')[0,0]
    return M

def dist_riemann(x,y):
    """
    Riemannian distance between SPD matrices x and SPD matrix y
    :param x: batch of SPD matrices (batch_size,1,n,n)
    :param y: single SPD matrix (n,n)
    :return:
    """
    return LogEig.apply(CongrG(x,y,'neg')).view(x.shape[0],x.shape[1],-1).norm(p=2,dim=-1)

def CongrG(P,G,mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if(mode=='pos'):
        GG=SqmEig.apply(G[None,None,:,:])
    elif(mode=='neg'):
        GG=SqminvEig.apply(G[None,None,:,:])
    PP=GG.matmul(P).matmul(GG)
    return PP

def LogG(x,X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    return CongrG(LogEig.apply(CongrG(x,X,'neg')),X,'pos')

def ExpG(x,X):
    """ Exponential mapping of x on the SPD manifold at X """
    return CongrG(ExpEig.apply(CongrG(x,X,'neg')),X,'pos')

def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size,channels,n=P.shape #batch size,channel depth,dimension
    Q=th.zeros(batch_size,channels,n,n,dtype=P.dtype,device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i,j]=P[i,j].diag()
    return Q

def karcher_step(x,G,alpha):
    """
    One step in the Karcher flow
    """
    x_log=LogG(x,G)
    G_tan=x_log.mean(dim=0)[None,...]
    G=ExpG(alpha*G_tan,G)[0,0]
    return G
    
def BaryGeom(x):
    """
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    """
    k=1
    alpha=1
    with th.no_grad():
        G=th.mean(x,dim=0)[0,:,:]
        for _ in range(k):
            G=karcher_step(x,G,alpha)
        return G

class Log_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.log(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 1/S

class Re_op():
    """ Log function and its derivative """
    _threshold=1e-4
    @classmethod
    def fn(cls,S,param=None):
        return nn.Threshold(cls._threshold,cls._threshold)(S)
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (S>cls._threshold).double()

class Sqm_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 0.5/th.sqrt(S)

class Sqminv_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return 1/th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return -0.5/th.sqrt(S)**3


class Power_op():
    """ Power function and its derivative """
    _power=1
    @classmethod
    def fn(cls,S,param=None):
        return S**cls._power
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (cls._power)*S**(cls._power-1)


class Exp_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.exp(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return th.exp(S)
