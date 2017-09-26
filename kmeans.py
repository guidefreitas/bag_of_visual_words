from sklearn.cluster import MiniBatchKMeans
import numpy as np

class Kmeans:
  kmeans_batch_size = 45
  kmeans = None
  k = None
  def __init__(self, k=10, centers=None):
    self.k = k
    if(centers!=None):
      init_centers = centers
    else:
      init_centers = 'k-means++'

    self.kmeans = MiniBatchKMeans(init=init_centers, n_clusters=self.k, batch_size=self.kmeans_batch_size,
                       n_init=10, max_no_improvement=10, verbose=0)
  
  def fit(self, X):
    self.kmeans.fit(X)

  def partial_fit(self, X):
    self.kmeans.partial_fit(X)

  def predict(self, X):
    return self.kmeans.predict(X)

  def get_centers(self):
    return self.kmeans.cluster_centers_

  def set_centers(self, centers):
    self.kmeans.cluster_centers_ = centers    

  def predict_hist(self, X):
    labels = self.predict(X)
    bins = range(self.k)
    histogram = np.histogram(labels, bins=bins, density=True)[0]
    #histogram = histogram/X.shape[0]
    return histogram

  def get_params(self):
    return self.kmeans.get_params(deep=True)

  def set_params(self, kmeans_params):
    self.kmeans.set_params(**kmeans_params)