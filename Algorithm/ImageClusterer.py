import numpy as np
import cv2
import json
from Utilities import *

import os
import sys

'''from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle'''

#sys.path.append(os.path.join('vocpy-master', 'tools'))
#import generatecodebook

class Image:
  def __init__(self, image_path, metadata_path, features):
    self.image_path = image_path
    self.load_metadata(metadata_path)
    self.features = features
    
  def load_metadata(self, metadata_path):
    with open(metadata_path, 'r') as f:
      self.metadata = json.load(f)

def image_distance(img1, img2):
  pass

def cluster_images(images):
  #images = images[:30]
  descriptors = []
  for image in images[]: # ota vain osa
    try:
      img = cv2.imread(image.image_path, cv2.IMREAD_COLOR)
      surf = cv2.SURF(400)
      kp, des = surf.detectAndCompute(img, None)
    except:
      print "fuufuu"
      break
    try:
      image.features = (kp, des)
      descriptors.extend(des)
    except:
      print "failed to add descriptors", len(kp)
    #print des
    #print '---'
    
  # Cluster descriptors for codebook  (borrowed from codebook.py)
  from sklearn.cluster import MiniBatchKMeans
  print "descriptors:", len(descriptors)
  return
  #n_clusters = len(descriptors) / 400
  n_clusters = 10
  print "codes:", n_clusters
  mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters,
      batch_size=n_clusters, n_init=3, max_iter=50,
      max_no_improvement=3, verbose=0, compute_labels=False)
  mbk.fit(descriptors)
  codebook = mbk.cluster_centers_
  print "codebook:", codebook
  
  '''visual_distances = np.zeros((len(images), len(images)))
  for i in range(len(images)):
    for j in range(i+1, len(images)):
      d = 0.0
      visual_distances[i, j] = d'''
  
'''# This is the actual clustering algorithm
def cluster_images_test(images):
  print "Clustering images"
  centers = [[1,1], [-1,-1], [1,-1]]
  
  X = [[0.0,0.0],[0.2,0.1],[1.0,1.0],[-2.0,-0.5],[-1.0,-2.0]]
  X = []
  for img in images:
    gps = img.metadata['gps']
    X.append([float(gps[0]), float(gps[1])])
  X = np.array(X)
  print X
  
  #X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
  print X
  #bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
  #print bandwidth
  bandwidth = 0.5
  ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
  ms.fit(X)
  labels = ms.labels_
  cluster_centers = ms.cluster_centers_
  labels_unique = np.unique(labels)
  n_clusters_ = len(labels_unique)
  print "number of estimated clusters:", n_clusters_
  
  plt.figure(1)
  plt.clf()
  colors = cycle('bgrcmyk')
  for k, col in zip(range(n_clusters_), colors):
    my_members = labels==k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members,0], X[my_members,1], col+'.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
  plt.show()'''
  

def main():
  folder = get_folder_argument()
  base_folder = '../' + folder + '/'
  
  # Load features
  all_features = load_features(base_folder)
  
  # Read metadata from files
  (image_paths, metadata_paths) = get_image_paths(base_folder)
  images = []
  for i in range(len(image_paths)):
    features = ['todo']
    images.append(Image(image_paths[i], metadata_paths[i], features))
  
  # Start the main algorithm
  cluster_images(images)

if __name__ == '__main__':
  main()