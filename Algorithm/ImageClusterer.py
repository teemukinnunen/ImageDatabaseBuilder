import numpy as np
import math
import random
import cv2
import json

import Utilities

import os
import sys

class Image:
  image_path = None
  metadata = None
  features = []
  tags = []
  gps = []
  
  def __init__(self, image_path, metadata_path):
    self.image_path = image_path
    self.load_metadata(metadata_path)
    
  def load_metadata(self, metadata_path):
    with open(metadata_path, 'r') as f:
      self.metadata = json.load(f)
    self.tags = self.metadata['tags']
    gps = self.metadata['gps']
    self.gps = [float(gps[0]), float(gps[1])]

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
    
def test(images, folder):
  n_images = len(images)
  n_codes = 500
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
      threshold = 100000.0 #
      if min_dist < threshold:
        hist[min_index] += 1.0
      else:
        print "threshold miss"
    image.hist = hist / np.linalg.norm(hist, 2)      # todo mieti onko jarkeva
    #print image.hist
  # Image distances
  print "Clustering images..."
  feature_array = np.array([image.hist for image in images])
  
  from sklearn.cluster import KMeans
  ms = KMeans(n_clusters = n_images / 5)
  labels = ms.fit_predict(feature_array)
  #from sklearn.cluster import MeanShift
  #ms = MeanShift()
  #labels = ms.fit_predict(feature_array)
  clusters = {}
  for l in np.unique(labels):
    clusters[l] = []
  for i in range(len(labels)):
    clusters[labels[i]].append(i)
  for c in clusters:
    print "Images in cluster:"
    for index in clusters[c]:
      image = images[index]
      print image.image_path
  # Save clusters for easy viewing
  base_output_folder = './Clusters/' + folder + '/'
  print "Saving clusters to {}".format(base_output_folder)
  for c in clusters:
    input_folder = Utilities.get_folder(images[clusters[c][0]].image_path)
    output_folder = base_output_folder + '{}/'.format(c)
    img_paths = []
    md_paths = []
    for index in clusters[c]:
      image = images[index]
      filename = Utilities.get_filename(image.image_path).split('.')[0] # get rid of folder and extension
      img_paths.append(filename + '.jpg')
      md_paths.append(filename + '.txt')
      print filename
    print "Copying cluster {} files from {} to {}".format(c, input_folder, output_folder)
    Utilities.copy_images(input_folder, output_folder, img_paths, md_paths)
  
# Not clustering, just finding "paths" through distance
def find_views(images, folder):
  pass

  
def cluster_by_tags_and_gps(images, folder):
  # Shuffle images
  n_images = len(images)
  print "n_images:", n_images
  random.shuffle(images)
  
  # Create vocabulary
  vocabulary = []
  tags = []
  for image in images:
    for tag in image.metadata['tags']:
      if tag not in vocabulary:
        vocabulary.append(tag)
    tags.append(' '.join(image.tags))
    
  # todo: make vocabulary smaller here
  
  # Create TF-IDF features from each image's tags
  from sklearn.feature_extraction.text import TfidfVectorizer
  tfidf_vectorizer = TfidfVectorizer()
  tfidf_matrix = tfidf_vectorizer.fit_transform(tags)
  print "tfidf matrix shape:", tfidf_matrix.shape
  print "vocab size: {}, tfidf size: {}".format(len(vocabulary), tfidf_matrix.shape[1])
  
  # Calculate cosine similarity between images using tfidf features
  from sklearn.metrics.pairwise import cosine_similarity
  D_tags = cosine_similarity(tfidf_matrix)
  
  # Calculate gps differences
  #D_gps = 
  
  # gps clustering (just testing)
  '''X_gps = np.array([image.gps for image in images])
  from sklearn.cluster import KMeans
  km = KMeans(n_clusters = n_images/20)
  labels = km.fit_predict(X_gps)
  
  # Plot clusters
  import matplotlib.pyplot as plt
  plt.figure()
  colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'*100])
  colors = np.hstack([colors] * 20)
  plt.scatter(X_gps[:,0], X_gps[:,1], color=colors[labels].tolist())
  plt.show()'''
  

def get_folder_argument():
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  args = parser.parse_args()
  return args.folder_name


def main():
  folder = Utilities.get_folder_argument()
  base_folder = '../' + folder + '/'
  
  # Load features
  #all_features = load_features(base_folder)
  
  # Read metadata from files
  (image_paths, metadata_paths) = Utilities.get_image_paths(base_folder)
  images = []
  for i in range(len(image_paths)):
    images.append(Image(image_paths[i], metadata_paths[i]))
  
  # Start the main algorithm
  #test(images, folder)
  #cluster_images(images)
  #find_views(images, folder)
  cluster_by_tags_and_gps(images, folder)

if __name__ == '__main__':
  main()