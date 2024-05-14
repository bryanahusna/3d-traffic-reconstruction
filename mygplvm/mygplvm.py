from matplotlib import pyplot as plt
import numpy as np
# from numpy.core.umath_tests import inner1d  # Fast trace multiplication
from scipy.optimize import fmin_cg  # Non-linear SCG
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data
from sklearn.gaussian_process import kernels
from sklearn.datasets import load_wine
import time
from tqdm import tqdm
# from fake_dataset import generate_observations, plot
from datetime import datetime

def kernel(X, Y, alpha, beta, gamma):
    kernel = kernels.RBF(length_scale=(1./gamma**2))
    return np.matrix(alpha*kernel(X, Y) + np.eye(X.shape[0])/(beta**2))

def likelihood(var, *args):
    YYT, N, D, latent_dimension, = args

    X = np.array(var[:-3]).reshape((N, latent_dimension))
    alpha = var[-3]
    beta = var[-2]
    gamma = var[-1]
    K = kernel(X, X, alpha, beta, gamma)


    # return -log likelihood
    # trace = np.sum(inner1d(K.I, YYT))
    trace = np.trace(K.I * YYT)
    return D*np.linalg.slogdet(K).logabsdet/2 + trace/2

class MyGPLVM:
    def __init__(self):
        self.iteration = 0
        self.name = None

        # parameters
        self.X = None
        self.Y = None
        self.alpha = None
        self.beta = None
        self.gamma = None

    def save_vars(self, var):
        self.iteration += 1
        if self.iteration%10 == 0:
            timestamp = str(datetime.now()).replace(" ", "_")
            np.save("results/" + str(self.name) + "_" + timestamp + ".npy", var)

    def simple_gplvm(self, Y, experiment_name="experiment", latent_dimension=2, epsilon = 0.001, maxiter = 10):
        ''' Implementation of GPLVM algorithm, returns data in latent space
        '''
        global name
        name = experiment_name
        Y = np.matrix(Y)
        # TODO(oleguer): Should we center observations?
        
        # Initialize X through PCA
        # First X approximation
        X = PCA(n_components=latent_dimension).fit_transform(np.asarray(Y))
        kernel_params = np.ones(3)  # (alpha, beta, gamma) TODO(oleguer): Should we rewrite those at each iteration? I dont thinkn so

        var = list(X.flatten()) + list(kernel_params)
        YYT = Y*Y.T
        N = Y.shape[0]
        D = Y.shape[1]

        # Optimization
        t1 = time.time()
        var = fmin_cg(likelihood, var, args=tuple((YYT,N,D,latent_dimension,)), epsilon = epsilon, disp=True, callback=self.save_vars, maxiter=maxiter)
        print("time:", time.time() - t1)

        var = list(var)

        np.save("mygplvm/results/" + str(name) + "_final.npy", var)

        N = Y.shape[0]
        X = np.array(var[:-3]).reshape((N, latent_dimension))
        alpha = var[-3]
        beta = var[-2]
        gamma = var[-1]

        print("alpha", alpha)
        print("beta", beta)
        print("gamma", gamma)

        # save to property
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        return X
    
    def recall(self, x):
        def kernel(X, Y, alpha, beta, gamma):
            kernel = kernels.RBF(length_scale=(1./gamma**2))
            return np.matrix(alpha*kernel(X, Y))
            # if X.shape == Y.shape:
            #     return np.matrix(alpha*kernel(X, Y) + np.eye(Y.shape[0])/(beta**2))
            # else:
            #     return np.matrix(alpha*kernel(X, Y) + np.ones(Y.shape[0])/(beta**2))

        k = kernel(x, self.X, self.alpha, self.beta, self.gamma)
        K = kernel(self.X, self.X, self.alpha, self.beta, self.gamma)
        KI = K.I
        return k * KI * self.Y


if __name__ == "__main__":
    X = load_wine();
    print(X);