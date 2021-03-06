# Input: number of folds k
#         numpy matrix X of features, with n rows (samples), d columns (features)
#         numpy vector y of scalar values, with n rows (samples), 1 column
#        (optional) number threshold: either information or gini gain threshold to be used by algo
# Output: numpy vector z of k rows, 1 column
import math

import graphviz
import numpy as np
import sklearn


def run(k, X, y, algo, algo_name, threshold=None):
    n, d = X.shape
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
            X_train[t] = X.iloc[S[t]]
            Y_train[t] = y.iloc[S[t]]
        
        # q,mu_pos,mu_neg,sigma2_pos,sigma2_neg = probclearn.run(X_train, Y_train)
        clf = algo.train(X_train, Y_train, threshold) 
        # dot_data = sklearn.tree.export_graphviz(clf, feature_names = X.columns, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render("visualizations/visualization_"+algo_name+"_"+str(i))
        z[i] = 0

        for t in T:
            y_pred = algo.test([X.iloc[t]], clf)
            pred_value = False
            if y_pred[0] == 1.0:
                pred_value = True

           
            if (y.iloc[t]['is_pork'] != pred_value):
               z[i] = z[i] + 1
               
        z[i] = z[i]/len(T)
    return z
