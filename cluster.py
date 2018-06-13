import pickle
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = pickle.load(open( "embedded_np.p", "rb" ))

kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
asmt = kmeans.predict(X)
print (asmt)

pickle.dump(asmt, open("row_asmt_np.p", "wb"))

# import matplotlib
# matplotlib.use("svg")
# import matplotlib.pyplot as plt
# x = range(2, 16)
# plt.scatter(x, scores)
# plt.savefig('cluster_score.png')
