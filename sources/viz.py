import graphviz
import sklearn

import cart

# Produce vizualization of the decision tree produced by the model

X, y = cart.load_data()
clf = cart.train(X,y)

dot_data = sklearn.tree.export_graphviz(clf, feature_names=list(X.columns), class_names=["Is Pork", "Is Not Pork"], out_file=None, filled=True)
graph = graphviz.Source(dot_data, format="pdf")
graph.render("visualization", "visualizations/")
