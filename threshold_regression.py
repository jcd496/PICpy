import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pickle
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LassoThresholdRegressor():
    """
    This implementation of Lasso Regression achieves the same result for sparse identification of dynamic systems as the
    sklearn implementation sklearn.linear_model.Lasso. It is unfortnuately much slower, but for the sake of interpretability
    this is much more straight forward.   Tested on basic linear regression, lorentz attractors, and our data.  
    """
    def __init__(self, lr = 0.01,
                 lambda_=1.0,
                 tolerance = 1e-4,
                 withBias = False,
                 verbose = False,
                 maxIter = 1e4,
                 scaling = False,
                 lthreshold = 0.0,
                 uthreshold = 1e6):
        """ Linear Regressor Object
            will use L0-norm, sklearn styled.
            Data dim:
                x [m,n]
                y [m]
            1/m * [\sum_i^m (y - [w*x + b])^2 + \lambda \sum_i^m w!=0]
        """
        self.lr = lr
        self.lambda_ = lambda_
        self.W = None
        self.B = None
        self.m = None
        self.n = None
        self.withBias = withBias
        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()
        self.verbose = verbose
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.scaling = scaling
        self.lthreshold = lthreshold
        self.uthreshold = uthreshold
        self.dws = []
    def fit(self, x, y):
        
        if y.ndim == 1:
            y = y[:, np.newaxis]
            
        self.m, self.n = x.shape
        self.W = np.ones((self.n, 1))
        self.B = np.ones((y.shape[1])) 
        
        #scale inputs
        xscaled = x if not self.scaling else self.xscaler.fit_transform(x)
        yscaled = y if not self.scaling else self.yscaler.fit_transform(y)

        previousError = 0
        mse = self.__loss(yscaled, xscaled)

        print('Initial MSE: ', mse)
        iterations = 0
        div_cnt = 0
        while(iterations < self.maxIter and abs(mse - previousError) > self.tolerance and div_cnt < 2):
            
            previousError = mse
            self.__updateParameters(xscaled, yscaled)
            mse = self.__loss(yscaled, xscaled) ## loss = mse + lambda*L1
            
            if self.verbose:
                print('MSE: ', mse)
                
            ## early stop condition
            if mse > previousError:
                div_cnt += 1
            else:
                div_cnt = 0
            iterations += 1
            
        zeroMask = np.where((np.abs(self.W) < self.lthreshold) | (np.abs(self.W) > self.uthreshold))
        self.W[zeroMask] = 0
        
        ## Report Results ##
        plt.plot(yscaled, 'o', label='data')
        plt.plot(self.__fit_predict(xscaled), label='prediction')
        plt.legend()
        print(f"\t final error: {mse} after {iterations} iterations")
        print(f'Weights:')
        for i in range(len(self.W)):
            print(f'\t Term {i}: {self.W[i]}')
        if self.withBias:
            print(f"\t B: {self.B}")
        #####################
            
    def __loss(self, y, x):
        yt = self.__fit_predict(x)
        return np.mean( (y - yt)**2 )  + self.lambda_*np.sum(np.abs(self.W)) 

    def __updateParameters(self, x, y):
        """
        w_k = w_{k-1} - lr*dw
        b_k = b_{k-1} - lr*db
        """
        yt = self.__fit_predict(x)
        residuals = yt - y
        dW = 2/self.m * (x.T).dot(residuals) + self.lambda_ * (self.W != 0).astype(int)*np.sign(self.W)
        self.dws.append(dW)
        self.W = self.W - self.lr*dW

        if self.withBias:
            dB = 2/self.m * np.sum( residuals )
            self.B = self.B - self.lr*dB

    def __fit_predict(self, x):
        yt = np.dot( x, self.W )
        if self.withBias:
            yt = yt + self.B
        return yt
    
    def predict(self, x):
        xscaled = x if not self.scaling else self.xscaler.transform(x)
        yt = np.dot(xscaled, self.W )
        if self.withBias:
            yt = yt + self.B
        return yt if not self.scaling else self.yscaler.inverse_transform(yt)
    
    def getWeights(self, x):
        #scl = self.xscaler.transform(x)
        #invScl = self.xscaler.inverse_transform(self.W*scl)
        for i in range(self.n):
            #print(f'{i} {np.mean(invScl[-1:,i]/(x[-1:,i]))}' )
            print(f'{i} {self.xscaler.var_[i]*self.W[i]}')
            
            
class LstsqThresholdRegressor():
    """
    This implementation of thresholded least squares uses the numpy linalg.lstsqr
    and iteratively thresholds result.  Tested on basic linear regression, lorentz attractors, and our data.
    Abstracts away more hyperparameters than the LassoThresholdRegressor.
    """
    def __init__(self, 
                 scaling = False,
                 lthreshold = 0.0,
                 uthreshold = 1e6):
        """ Linear Regressor Object
            Data dim:
                x [m,n]
                y [m]
            1/m * [\sum_i^m (y - [w*x + b])^2 + \lambda \sum_i^m w!=0]
        """
        self.W = None
        self.B = None
        self.m = None
        self.n = None
 
        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()
        self.scaling = scaling
        self.lthreshold = lthreshold
        self.uthreshold = uthreshold

        self.dws = []
    def fit(self, x, y):
        
        if y.ndim == 1:
            y = y[:, np.newaxis]

        self.m, self.n = x.shape
        self.W = np.ones((self.n, 1))
        self.B = np.ones((y.shape[1])) 

        #scale inputs
        xscaled = x if not self.scaling else self.xscaler.fit_transform(x)
        yscaled = y if not self.scaling else self.yscaler.fit_transform(y)

        self.W, residuals, rank, s = np.linalg.lstsq(xscaled, yscaled)
        for i in range(self.n):
            #print(self.W)
            oneMask = ((np.abs(self.W) > self.lthreshold) * (np.abs(self.W) < self.uthreshold)).flatten()
            self.W[~oneMask] = 0
            self.W[oneMask], residuals, rank, s = np.linalg.lstsq(xscaled[:,oneMask], yscaled)
            
        ## Report Results ##
        plt.plot(yscaled, 'o', label='data')
        plt.plot(self.__fit_predict(xscaled), label='prediction')
        plt.legend()
        print(f"final SE:\n\t {residuals.flatten()}")
#         print(f"final MSE:\n\t {residuals.flatten()/self.m}")
        print(f'Weights:')
        for i in range(self.n):
            print(f'\t Term {i}: {self.W[i]}')
        #####################
        
    def __fit_predict(self, x):
        yt = np.dot( x, self.W )
        return yt
    
    def predict(self, x):
        xscaled = x if not self.scaling else self.xscaler.transform(x)
        yt = np.dot(xscaled, self.W )
        return yt if not self.scaling else self.yscaler.inverse_transform(yt)
    
    