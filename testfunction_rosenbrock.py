import numpy as np

def testfunction_rosenbrock_d(X, r = 1):
    mu, sigma = 0, 0
    noise = np.random.normal(mu, sigma, r)

    d = len(X)
    f_val = 0
    for i in range(d-1):
        f_val += (1 - X[i])**2 + 100*(X[i+1] - X[i]**2)**2 

    return (f_val + noise- 633.085).tolist()

#9.93098 2d. # 92.9442 3d. 