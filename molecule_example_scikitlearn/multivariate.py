import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.cross_decomposition import PLSRegression
import sys

def read_X_Y(filepath):
    df = pandas.read_csv(filepath_or_buffer=filepath, sep=',')
    print(df.shape)
    target = df.ix[:,0].values
    x = df.ix[:,1:].values

    return x, target

def do_pca(X, y):
    print (X.shape)
    pca = decomposition.IncrementalPCA(n_components=2)
    comps = pca.fit(X).transform(X)
    print(comps.shape)

    #plt.scatter(X[:, 0], X[:, 1], cmap='viridis')
    plt.scatter(comps[:,0], comps[:,1], c=y, cmap='viridis', s=70)

    plt.savefig('pca.png', dpi=125)
    plt.show()


if __name__ == '__main__':
    X, y = read_X_Y('Data/train_500.csv')

    do_pca(X, y)
    #do_pls(X, y




















def do_pls(X, Y):
    pls2 = PLSRegression(n_components=2)
    pls2.fit(X,Y)
    out = pls2.transform(X)
    print(out)
    print(out.shape)

    plt.title("PLS2")
    plt.xlabel("PL1")
    plt.ylabel("PL2")
    plt.grid();
    plt.scatter(out[:, 0], out[:, 1], c=Y, cmap='viridis')
    plt.savefig('pls.png', dpi=125)
    #plt.show()
