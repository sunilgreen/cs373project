import c45_tree
import cart
import kfoldcv
from sklearn import metrics
import numpy as np
import pandas as pd
import random
import csv
NUM_ITERS = 30


def show_graphs():
    ##Make sure code in cart.train is uncommented 
    X, y = cart.load_data()
    kfoldcv.run(4, X, y, cart, "cart", threshold=0.026)

# Observe error when using different Gini Threshold values
def tune_cart():
    X, y = cart.load_data()
    for i in range(0, 300):
        thresh = float(i/1000)
        print("Threshold: "+str(thresh))
        print(kfoldcv.run(4, X, y, cart, "cart", threshold=float(i/1000)))


# Observe error when using different Information Gain Threshold values
def tune_C45():
    #Note: the code for loading data is in cart.py but the process is the same for both algorithms 
    X, y = cart.load_data() 
    for i in range(1,40):
        print("Threshold: "+str(i))
        print(kfoldcv.run(4, X, y, c45_tree, "c45", threshold=float(i)))

# Test accuracy when using different values of the hyperparameter Information Gain Threshold
def thresholds(C45):

    if (C45):
        # Open CSV to which findings will be written
        with open("thresholds_C45.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Information Gain Threshold", "Accuracy"])


            # Load bill data
            X, y = cart.load_data()

            # Check accuracy when information gain threshold is t = 1, 2, 3,...30
            for t in range(1,200):

                # The running sum of accuracies, to be divided to find the average
                t_tot = 0

                for iters in range(NUM_ITERS):

                    # Select 20 random indices to use for test data, without replacement
                    indices = random.sample(range(200), 20)

                    # All possible indices that weren't randomly selected
                    unselected_indices = [i for i in range(200) if i not in indices]

                    # Use every sample that's not test data as training data
                    X_sub = np.array([X.iloc[i] for i in unselected_indices])
                    y_sub = np.array([y.iloc[i] for i in unselected_indices])

                    # Get testing set using randomly selected indices
                    X_test = np.array([X.iloc[i] for i in indices])
                    y_test = np.array([y.iloc[i] for i in indices])

                    clf = c45_tree.train(X_sub, np.hstack(y_sub), threshold=t)

                    # Increment running accuracy total
                    t_tot += clf.score(X_test, y_test)

                t_avg = t_tot/NUM_ITERS

                print("Average accuracy for threshold = "+str(t)+": "+str(t_avg))

                # Write threshold and corresponding average accuracy to CSV file
                writer.writerow([t, t_avg])
    else:
        # Open CSV to which findings will be written
        with open("thresholds_cart.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Gini Threshold", "Accuracy"])


            # Load bill data
            X, y = cart.load_data()

            # Check accuracy when information gain threshold is t = 1, 2, 3,...30
            for t in range(0,31):

                # The running sum of accuracies, to be divided to find the average
                t_tot = 0

                for iters in range(NUM_ITERS):

                    # Select 20 random indices to use for test data, without replacement
                    indices = random.sample(range(200), 20)

                    # All possible indices that weren't randomly selected
                    unselected_indices = [i for i in range(200) if i not in indices]

                    # Use every sample that's not test data as training data
                    X_sub = np.array([X.iloc[i] for i in unselected_indices])
                    y_sub = np.array([y.iloc[i] for i in unselected_indices])

                    # Get testing set using randomly selected indices
                    X_test = np.array([X.iloc[i] for i in indices])
                    y_test = np.array([y.iloc[i] for i in indices])

                    clf = cart.train(X_sub, np.hstack(y_sub), threshold=float(t/1000))

                    # Increment running accuracy total
                    t_tot += clf.score(X_test, y_test)

                t_avg = t_tot/NUM_ITERS

                print("Average accuracy for threshold = "+str(t/1000)+": "+str(t_avg))

                # Write threshold and corresponding average accuracy to CSV file
                writer.writerow([t, t_avg])
        
            


# Test accuracy when training with different sized subsets
def subsets(C45):
    if (C45):
        # Open CSV to which findings will be written
        with open("subsets_C45.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Number of Samples", "Accuracy"])
            
            # Load bill data
            X, y = cart.load_data()


            # Check accuracy with subsets of size n = 10, 20, 30, ..., 180
            for n in range(5, 181, 5):

                # The running sum of accuracies, to be divided to find the average
                n_tot = 0

                for iters in range(NUM_ITERS):
                    # Select n random indices plus 20 to use as test data, without replacement
                    indices = random.sample(range(200), n + 20)

                    # Get subset to train on
                    X_sub = np.array([X.iloc[i] for i in indices[:n]])
                    y_sub = np.array([y.iloc[i] for i in indices[:n]])

                    # Get testing subset
                    X_test = np.array([X.iloc[i] for i in indices[n:]])
                    y_test = np.array([y.iloc[i] for i in indices[n:]])

                    
                    clf = c45_tree.train(X_sub, np.hstack(y_sub), threshold=1.0)

                    # Increment running accuracy total
                    n_tot += clf.score(X_test, y_test)

                n_avg = n_tot/NUM_ITERS

                print("Average accuracy for n = "+str(n)+": "+str(n_avg))
                
                # Write n and corresponding average accuracy to CSV file
                writer.writerow([n, n_avg])
    else:
        with open("subsets_cart.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Number of Samples", "Accuracy"])
            
            # Load bill data
            X, y = cart.load_data()


            # Check accuracy with subsets of size n = 10, 20, 30, ..., 180
            for n in range(5, 181, 5):

                # The running sum of accuracies, to be divided to find the average
                n_tot = 0

                for iters in range(NUM_ITERS):
                    # Select n random indices plus 20 to use as test data, without replacement
                    indices = random.sample(range(200), n + 20)

                    # Get subset to train on
                    X_sub = np.array([X.iloc[i] for i in indices[:n]])
                    y_sub = np.array([y.iloc[i] for i in indices[:n]])

                    # Get testing subset
                    X_test = np.array([X.iloc[i] for i in indices[n:]])
                    y_test = np.array([y.iloc[i] for i in indices[n:]])

                    
                    clf = cart.train(X_sub, np.hstack(y_sub), threshold=0.026)

                    # Increment running accuracy total
                    n_tot += clf.score(X_test, y_test)

                n_avg = n_tot/NUM_ITERS

                print("Average accuracy for n = "+str(n)+": "+str(n_avg))
                
                # Write n and corresponding average accuracy to CSV file
                writer.writerow([n, n_avg])


thresholds(False)