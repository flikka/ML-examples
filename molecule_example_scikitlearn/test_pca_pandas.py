import pandas

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

df = pandas.read_csv(filepath_or_buffer='Data/train_500.csv', sep=',')

X = df.ix[:,1:].values
y = df.ix[:,0].values

#X = StandardScaler().fit_transform(X)
tran_ne = X
pca = decomposition.PCA(n_components=5)
comps = pca.fit(tran_ne).transform(tran_ne)

print(comps.shape)


plt.scatter(comps.T[1,:], comps.T[2,:], c=y, cmap='viridis')

plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid();
plt.savefig('molecule.png', dpi=125)

