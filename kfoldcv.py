# Input: number of folds k
#         numpy matrix X of features, with n rows (samples), d columns (features)
#         numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
import numpy as np
import math
import cart
import probcpredict
def run(k,X,y):
    n = len(X)
    d = len(X[0])
    #print(n,d)
    z = np.zeros((k,1))
    for i in range(0,k):
       
       
        lower = math.floor((n*i)/k)
        upper = math.floor((n*(i+1))/(k))
        T = np.array(range(lower,upper))
        S = np.array(range(0,n))
        S = np.setdiff1d(S, T)
        
  
        X_train = np.zeros((len(S),d))
        Y_train = np.zeros(len(S))
        
        for t in range(0,len(S)):
            X_train[t] = X[S[t]]
            Y_train[t] = y[S[t]]
        
        q,mu_pos,mu_neg,sigma2_pos,sigma2_neg = probclearn.run(X_train, Y_train)
        z[i] = 0
        for t in T:
           
           
           if (y[t] != probcpredict.run(q,mu_pos,mu_neg,sigma2_pos,sigma2_neg, np.transpose([X[t]]))):
               
               z[i] = z[i] + 1
               
        z[i] = z[i]/len(T)
    return z