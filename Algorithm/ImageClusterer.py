import numpy as np
import math
import random
import cv2
import json
from Utilities import *

import os
import sys

class Image:
  image_path = None
  metadata = None
  features = []
  
  def __init__(self, image_path, metadata_path):
    self.image_path = image_path
    self.load_metadata(metadata_path)
    
  def load_metadata(self, metadata_path):
    with open(metadata_path, 'r') as f:
      self.metadata = json.load(f)

def image_distance(img1, img2):
  pass

def cluster_images(images):
  random.shuffle(images)
  ### Create bag of features
  # Extract features from images
  descriptors = []
  maxd = 0
  for image in images:
    img = cv2.imread(image.image_path, cv2.IMREAD_COLOR)
    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(img, None)
    # todo: limit descriptors?
    descriptors.extend(des)
    
  #todo PCA
  random.shuffle(descriptors)
  
  # Cluster descriptors for codebook
  from sklearn.cluster import MiniBatchKMeans
  codebook_size = 100
  print "Creating codebook of size {} from {} descriptors".format(codebook_size, len(descriptors))
  mbk = MiniBatchKMeans(n_clusters=codebook_size, init='k-means++', n_init=3, max_iter=50)
  '''mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters,
      batch_size=, n_init=3, max_iter=50,
      max_no_improvement=3, verbose=0, compute_labels=False)'''
  mbk.fit(descriptors)
  codebook = mbk.cluster_centers_
  print "Calculating visual words for images"
  def eucl_dist(a, b):
    if len(a) != len(b):
      raise Exception("Vectors are different length: {} and {}".format(len(a), len(b)))
    d = 0
    for i in range(len(a)):
      d += (a[i] - b[i])**2
    return math.sqrt(d)
  for image in images:
    image.features = [0] * len(codebook)
    img = cv2.imread(image.image_path, cv2.IMREAD_COLOR)
    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(img, None)
    for d in des:
      d = np.array(d)
      min_index = 0
      min_dist = eucl_dist(codebook[0, :], d)
      for i in range(1, codebook.shape[0]):
        dist = eucl_dist(codebook[i, :], d)
        if dist < min_dist:
          min_index = i
          min_dist = dist
      image.features[min_index] += 1
  print images[5].features
    
def test(images):
  n_images = len(images)
  n_codes = 800
  # Extract features from images
  features = []
  print "Extracting image features for {} images...".format(n_images)
  for image in images:
    img = cv2.imread(image.image_path, cv2.IMREAD_COLOR)
    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(img, None)
    image.des = [np.array(d) for d in des]
    features.extend(des[:1000])
  # Create the codebook
  print "Creating codebook of size {} from {} features...".format(n_codes, len(features))
  random.shuffle(features)
  features = np.array(features)
  codebook = features[:n_codes]
  for image in images:
    hist = np.zeros(n_codes)
    for des in image.des:
      min_index = 0
      min_dist = np.linalg.norm(codebook[0] - des)
      for i in range(1, n_codes):
        dist = np.linalg.norm(codebook[i] - des)
        if dist < min_dist:
          min_index = i
          min_dist = dist
      threshold = 1.0 #
      if min_dist < threshold:
        hist[min_index] += 1.0
    image.hist = hist / np.linalg.norm(hist, 2)      # todo mieti onko jarkeva
  # Image distances
  print "Computing image distances..."
  distances = np.zeros((n_images, n_images))
  for i in range(n_images):
    img1 = images[i]
    for j in range(i+1, n_images):
      img2 = images[j]
      d = np.linalg.norm(img1.hist - img2.hist)
      distances[i,j] = d
      distances[j,i] = d
  print distances
  print "Clustering images..."
  '''max_dist_ij = np.unravel_index(distances.argmax(), distances.shape)
  distances += 10000.0 * np.identity(n_images)
  min_dist_ij = np.unravel_index(distances.argmin(), distances.shape)
  print images[max_dist_ij[0]].image_path, images[max_dist_ij[1]].image_path
  print images[min_dist_ij[0]].image_path, images[min_dist_ij[1]].image_path'''
  from sklearn.cluster import DBSCAN
  dbs = DBSCAN(eps=0.4, metric='precomputed', min_samples=2)
  dbs.fit(distances)
  print dbs.labels_

def main():
  folder = get_folder_argument()
  base_folder = '../' + folder + '/'
  
  # Load features
  all_features = load_features(base_folder)
  
  # Read metadata from files
  (image_paths, metadata_paths) = get_image_paths(base_folder)
  images = []
  for i in range(len(image_paths)):
    images.append(Image(image_paths[i], metadata_paths[i]))
  
  # Start the main algorithm
  test(images)
  #cluster_images(images)

if __name__ == '__main__':
  main()