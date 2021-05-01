from c45 import C45

# Credit: https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier

def train(X_train, y_train, threshold=None):
    clf = C45(threshold=threshold)
    clf = clf.fit(X_train, y_train)

    return clf

def test(X_test, clf):
    return clf.predict(X_test)
