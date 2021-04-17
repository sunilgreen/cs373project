import graphviz
import sklearn

import cart

# Produce vizualization of the decision tree produced by the model

X, y = cart.load_data()
clf = cart.train(X,y)

dot_data = sklearn.tree.export_graphviz(clf, feature_names = X.columns, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("visualization")
