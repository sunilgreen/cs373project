from c45 import C45


def train(X_train, y_train, threshold=None):
    clf = C45(threshold=threshold)
    clf = clf.fit(X_train, y_train)

    return clf

def test(X_test, clf):
    return clf.predict(X_test)
