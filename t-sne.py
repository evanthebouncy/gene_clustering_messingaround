import pickle
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt

X = pickle.load(open( "embedded_np.p", "rb" ))
X_embedded = TSNE(n_components=2).fit_transform(X)

print (X_embedded)
x = [x[0] for x in X_embedded]
y = [x[1] for x in X_embedded]

pickle.dump((x,y), open( "embedded_tsne.p", "wb" ) )

colors = (0,0,0)
area = np.pi*3
 
# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('gene_2D_vis.png')

