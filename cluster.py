import pickle
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = pickle.load(open( "embedded_np.p", "rb" ))

def get_score(i):
  kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
  return kmeans.score(X)

scores = [get_score(i) for i in range(2, 16)]

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt

x = range(2, 16)
plt.scatter(x, scores)
plt.savefig('cluster_score.png')
