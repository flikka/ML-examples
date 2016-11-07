from sklearn import datasets, neighbors, decomposition

from matplotlib import pyplot

def plot_pca_iris():
    iris_data = datasets.load_iris()
    X = iris_data.data
    Y = iris_data.target

    pca = decomposition.PCA()


    X_transformed = pca.fit_transform(X)

    print(X_transformed.shape)
    pyplot.scatter(X_transformed[:,0], X_transformed[:, 1], c=Y, s=70, label=Y)
    pyplot.show()

if __name__ == '__main__':
    plot_pca_iris()
