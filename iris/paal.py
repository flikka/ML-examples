
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

mlp = MLPClassifier()
mlp.fit(X, y)
print("Training set score: %.2f" % mlp.score(X, y))

mlp.predict( (5.0, 3.5, 1.5, 0.5) )  # 5.0 3.4 1.5 0.2 -> I.setosa 
# 0 setosa

mlp.predict( (6.0, 3.0, 4.0, 1.5) )  # 5.8 2.7 3.9 1.2 -> I.versicolor
# 1 versicolor

mlp.predict( (6.0, 3.0, 5.0, 2.0) )  # 5.9 3.0 5.1 1.8 -> I.virginica
# 2 virginica
