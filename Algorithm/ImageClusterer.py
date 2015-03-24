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
  date = None
  
  def __init__(self, image_path, metadata_path):
    self.image_path = image_path
    self.load_metadata(metadata_path)
    
  def load_metadata(self, metadata_path):
    with open(metadata_path, 'r') as f:
      self.metadata = json.load(f)
    self.tags = self.metadata['tags']
    gps = self.metadata['gps']
    self.gps = [float(gps[0]), float(gps[1])]
    date_string = self.metadata['datetaken']
    from dateutil import parser
    from datetime import datetime
    self.date = parser.parse(date_string)

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
  
  # Create visual codebook
  n_descriptors = 100
  n_codebook = 10
  n_maxfeatures = 1000
  n_images = len(images)
  descriptors = []
  surf = cv2.SURF(400)
  failed_images = 0
  print "Extract descriptors from images"
  try:
    for image, i in zip(images, range(n_images)):
        img = cv2.imread(image.image_path)
        kp, des = surf.detectAndCompute(img, None)
        if des == None:
          failed_images += 1
          continue
        des = des[:n_maxfeatures]
        descriptors.extend(des)
        #image.des = des
  except KeyboardInterrupt as e:
    raise e
  except:
    "Failed at image {}/{}".format(i+1, n_images)
      
  
  random.shuffle(descriptors)
  if n_descriptors != None:
    descriptors = descriptors[:n_descriptors]
  
  print "Clustering {} descriptors into codebook of size {}".format(len(descriptors), n_codebook)
  from sklearn.cluster import MiniBatchKMeans
  mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_codebook, n_init=3, max_iter=50, max_no_improvement=3, verbose=0, compute_labels=False) # batch size?
  mbk.fit(descriptors)
  codebook = mbk.cluster_centers_
  
  import scipy
  import scipy.cluster.vq
  
  def generate_histogram(codebook, features):  # from vocpy library
        [N, d] = codebook.shape
        if features.size <= 1:
            return np.zeros((N, 0))

        [hits, d] = scipy.cluster.vq.vq(features, codebook)
        [y, x] = np.histogram(hits, bins=range(0, N + 1))
        return y
  
  print "Generating feature histograms"
  visual_features = []
  for image in images:
    img = cv2.imread(image.image_path)
    _, des = surf.detectAndCompute(img, None)
    if des != None:
      visual_hist = generate_histogram(codebook, des)#image.des)
    else:
      visual_hist = [math.sqrt(1.0 / n_codebook)] * n_codebook # flat histogram
    visual_features.append(visual_hist)
  from sklearn.feature_extraction.text import TfidfTransformer
  visual_tfidf = TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
  visual_features = visual_tfidf.fit_transform(visual_features)
  #for vfs, image in zip(visual_features, images):
  #  image.vfs = vfs
  from sklearn.metrics.pairwise import cosine_similarity
  D_visual = [[0.5]*n_images]*n_images#cosine_similarity(visual_features)
  
  # FLANN matching
  '''FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(descriptors[0], descriptors[1], k=2)'''
  
  # Create tag vocabulary
  print "Creating tag vocabulary"
  vocabulary = []
  tags = []
  for image in images:
    #for tag in image.metadata['tags']:
    for tag in image.tags:
      if tag not in vocabulary:
        vocabulary.append(tag)
    tags.append(' '.join(image.tags))
    
  # Find time-specific tags
  '''hourly_tag_hists = [[0] * len(vocabulary)] * 24
  monthly_tag_hists = [[0] * len(vocabulary)] * 12
  for image in images:
    hour = image.date.hour
    month = image.date.month - 1  # 0-indexing
    for tag, i in zip(vocabulary, range(len(vocabulary))):
      if tag in image.tags:
        monthly_tag_hists[month][i] += 1
        hourly_tag_hists[hour][i] += 1
  from sklearn.feature_extraction.text import TfidfTransformer
  hourly_tag_hists = np.array(hourly_tag_hists).transpose()
  monthly_tag_hists = np.array(monthly_tag_hists).transpose()
  hourly_tfidf = TfidfTransformer().fit_transform(hourly_tag_hists)
  monthly_tfidf = TfidfTransformer().fit_transform(monthly_tag_hists)
  for i in range(24):
    hourly_max = max(hourly_tfidf[i])
    import code
    code.interact(local=locals())
    for j in range(hourly_tfidf.shape[0]):
      if hourly_tfidf[i, j] == hourly_max:
        print "max:", vocabulary[j]'''
  
  # Create TF-IDF features from each image's tags
  print "Computing tf-idf tag features for images"
  from sklearn.feature_extraction.text import TfidfVectorizer
  tfidf_vectorizer = TfidfVectorizer()
  tfidf_matrix = tfidf_vectorizer.fit_transform(tags)
  print "tfidf matrix shape:", tfidf_matrix.shape
  print "vocab size: {}, tfidf size: {}".format(len(vocabulary), tfidf_matrix.shape[1])
  
  # Calculate cosine similarity between images using tfidf features
  D_tags = [[0.5]*n_images]*n_images#cosine_similarity(tfidf_matrix)
  
  return # temp
  D_tot = np.zeros((n_images, n_images))
  for i in range(n_images):
    for j in range(i, n_images):
      d = D_visual[i, j] * D_tags[i, j]
      D_tot[i, j] = d
      #D_tot[j, i] = d
  min_ij = (1, 0)
  max_ij = (1, 0)
  min_d = D_tot[1, 0]
  max_d = D_tot[1, 0]
  print D_visual
  print "----"
  print D_tags
  for i in range(n_images):
    for j in range(i, n_images):
      if i == j:
        continue
      d = D_tot[i, j]
      if d < min_d:
        min_d = d
        min_ij = (i, j)
      elif d > max_d:
        max_d = d
        max_ij = (i, j)
  print "Min:", min_ij, min_d
  print "Max:", max_ij, max_d
  
  import code
  code.interact(local=locals())
  

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