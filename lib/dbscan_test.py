from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
data = np.random.rand(500,3)

db = DBSCAN(eps=0.13, min_samples=1).fit(data)
labels = db.labels_
print labels
print Counter(labels)

