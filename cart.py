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

#print(__doc__)



np.set_printoptions(threshold=sys.maxsize)

def load_data():
    mlb = MultiLabelBinarizer()
    # header_names = ["bill_id", "cosponsors", "sponsor_party", "sponsor_state", "committee_codes", "subcommittee_codes", "primary_subject", "is_pork"]
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

    # Transform sponsor columns into one-hot arrays
    enc = OneHotEncoder(handle_unknown='ignore')
    X = df[['sponsor_party', 'sponsor_state']]
    enc.fit(X)
    X = enc.transform(X).toarray()
    df = df.join(pd.DataFrame(X))
    # print(df.columns)
    df = df.drop(['bill_id'], axis=1)
    

    # Remove redundant columns
    df = df.drop(["sponsor_party", "sponsor_state", "primary_subject"], axis=1)
    #print(df.to_string())

    # X_train, X_test, y_train, y_test = train_test_split(df.to_numpy(), y, test_size=test_size, random_state=1)

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

    return clf

def test(X_test, clf):
    return clf.predict(X_test)
