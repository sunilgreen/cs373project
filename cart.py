import csv
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

header_names = ["bill_id", "cosponsors", "sponsor_party", "sponsor_state", "committee_codes", "subcommittee_codes", "primary_subject", "is_pork"]
df = pd.read_csv("billdata.csv")

enc = OneHotEncoder(handle_unknown='ignore')
X = df[['sponsor_party', 'sponsor_state', 'committee_codes']]
enc.fit(X)
X = enc.transform(X).toarray()
print(X.shape)



feature_names = ["cosponsors", "sponsor_party", "sponsor_state", "committee_codes", "subcommittee_codes", "primary_subject", "is_pork"]

y = df.is_pork

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
