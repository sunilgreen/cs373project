import ast
import csv
import sys

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

np.set_printoptions(threshold=sys.maxsize)

def load_data():

    # Binarize committe and subcommittee codes
    mlb = MultiLabelBinarizer()
    df = pd.read_csv("billdata.csv")

    # In csv file, column values are strings, but we must parse them as lists
    df["committee_codes"] = df["committee_codes"].transform(lambda x: ast.literal_eval(x))
    df["subcommittee_codes"] = df["subcommittee_codes"].transform(lambda x: ast.literal_eval(x))

    # Transform committe_codes and subcommittee_codes into several columns, such that e.g. HSFA, HSGO, etc. have their own binary column.
    # So if a particular bill has HSFA as a committee, it has a 1 in the HSFA column. Otherwise it has a 0.
    df = df.join(pd.DataFrame(mlb.fit_transform(df["committee_codes"]), columns=mlb.classes_, index=df.index))
    df = df.join(pd.DataFrame(mlb.fit_transform(df["subcommittee_codes"]), columns=mlb.classes_, index=df.index))

    # Remove columns since they are now redundant
    df = df.drop(["committee_codes", "subcommittee_codes"], axis=1)


    one_hot_encoder_columns = ['sponsor_party', 'sponsor_state', 'primary_subject']
    # Transform sponsor columns into one-hot arrays so decision tree can use categorical variables
    enc = OneHotEncoder(handle_unknown='ignore')
    X = enc.fit_transform(df[one_hot_encoder_columns])

    # Create dataframe with sponsor columns and their corresponding feature titles
    # The new features will be labled as sponsor_party_R, sponsor_party_D, sponsor_state_CA, sponsor_state_AZ, etc.
    #   and will take on binary values
    Xdf = pd.DataFrame(X.toarray(), columns=enc.get_feature_names(one_hot_encoder_columns))

    # Combine sponsor dataframe with that of the rest of the data
    df = df.join(Xdf).drop(['bill_id'], axis=1)
    
    # Remove columns made redundant
    df = df.drop(one_hot_encoder_columns, axis=1)

    X_data = df.drop(['is_pork'], axis=1) 
    y_data = df[['is_pork']]
    return X_data, y_data

def train(X_train, y_train):

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


    # sklearn.tree.plot_tree(clf)

    # plt.figure()
    # plot_tree(clf, filled=True)
    # plt.show()
    print(X_train)

    return clf

def test(X_test, clf):
    return clf.predict(X_test)
