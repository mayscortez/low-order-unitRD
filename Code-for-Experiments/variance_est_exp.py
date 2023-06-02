import numpy as np
import nci_linear_setup as ncls
import time

startTime = time.time()
n = 10000
r = 2
diag = 1
offdiag = r*diag
p = 0.2

# Create Weighted Adjcency matrix
deg = 10
A = ncls.erdos_renyi(n,deg/n)
rand_wts = np.random.rand(n,3)
alpha = rand_wts[:,0].flatten() # baseline effects
C = ncls.simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())

# Potential outcomes function
fy = lambda z: ncls.linear_pom(C,alpha,z)

TTE = 1/n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))
print("Ground-Truth TTE: {}\n".format(TTE))

T = 200
TTE_hat, TTE_var_hat = np.zeros(T), np.zeros(T)

for i in range(T):
    z = ncls.bernoulli(n,p)
    y = fy(z)

    TTE_hat[i] = ncls.SNIPE_deg1(n, p, y, A, z)
    TTE_var_hat[i] = ncls.var_est(n, p, y, A, z)

bound = ncls.var_bound(n, p, A, C, alpha)

endTime = time.time()

print("SNIPE: {}".format(np.sum(TTE_hat)/T))
print("SNIPE bias: {}\n".format(((np.sum(TTE_hat)/T) - TTE)/TTE))

exp_var = np.sum((TTE_hat-TTE)**2)/T
print("MSE (Experimental Variance): {}".format(exp_var))
print("Variance Bound: {}".format(bound))
print("Variance Estimate: {}\n".format(np.sum(TTE_var_hat)/T))
print("Variance Estimator bias: {}\n".format(((np.sum(TTE_var_hat)/T) - exp_var)))

print("Runtime (in minutes): {}".format((endTime-startTime)/60))