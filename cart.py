import ast
import csv
import sys

import graphviz
import numpy
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

numpy.set_printoptions(threshold=sys.maxsize)

mlb = MultiLabelBinarizer()
header_names = ["bill_id", "cosponsors", "sponsor_party", "sponsor_state", "committee_codes", "subcommittee_codes", "primary_subject", "is_pork"]
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

# Remove redundant columns
#df = df.drop(["sponsor_party", "sponsor_state"])
#print(df.to_string())



feature_names = ["cosponsors", "sponsor_party", "sponsor_state", "committee_codes", "subcommittee_codes", "primary_subject", "is_pork"]

y = df.is_pork

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# sklearn.tree.plot_tree(clf)

dot_data = sklearn.tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("wee")
