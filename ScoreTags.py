import argparse, json
import math
import string
from os import listdir
from os.path import isfile, join
import pylab as pl

class ImageInfo:
  def __init__(self, filename, rating):
    self.score = None
    self.filename = filename
    self.rating = int(rating)
    if self.rating == 0:
      self.rating = -1
    with open(filename + '.txt', 'r') as f:
      md = json.load(f)
      self.tags = md['tags']
      '''title = md['title'].encode('utf-8').translate(string.maketrans('',''), string.punctuation)
      title_tags = title.split(' ')
      if len(title_tags) != 0:
        self.tags.extend(title_tags)'''
      self.tags = [tag for tag in self.tags if len(tag) > 0]

def get_images_in_folder(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [folder + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    return files

def main(folder):
  # Read the manually created ratings
  with open(folder + 'ratings.txt', 'r') as f:
    ratings = json.load(f)
    
  # Read rated images
  rated_images = []
  for filename in ratings:
    if ratings[filename] == None:
      continue
    img = ImageInfo(filename, ratings[filename])
    rated_images.append(img)
  
  bad_hist = {}
  good_hist = {}
  all_tags = []
  for img in rated_images:
    for tag in img.tags:
      if tag not in all_tags:
        all_tags.append(tag)
      bad_hist[tag] = 0
      good_hist[tag] = 0
  for img in rated_images:
    for tag in img.tags:
      if img.rating == 1:
        good_hist[tag] += 1
      elif img.rating == -1:
        bad_hist[tag] += 1
  
  # Create the world's greatest histogram ever
  n = len(all_tags)
  all_tags.sort(key=lambda tag: good_hist[tag] + bad_hist[tag], reverse=True)
  all_tags = all_tags[:200]
  show_tags = False
  weight_freq = False
  
  fig = pl.figure()
  good_data = [good_hist[tag] for tag in all_tags]
  bad_data = [-bad_hist[tag] for tag in all_tags]
  if weight_freq:
    rated_good = 0
    rated_bad = 0
    for img in rated_images:
      if img.rating == 1:
        rated_good += 1
      elif img.rating == -1:
        rated_bad += 1
    w = float(rated_good) / rated_bad
    print w
    bad_data = [d * w for d in bad_data]
  ax = pl.subplot(111)
  ax.bar(range(len(all_tags)), good_data, width=1, color='b')
  ax.bar(range(len(all_tags)), bad_data, width=1, color='r')
  pl.ylabel('frekvenssi')
  pl.xlabel('avainsanan indeksi')
  if show_tags:
    pl.xticks(range(len(all_tags)), all_tags)
    pl.xticks(rotation=90)
    pl.xlabel('avainsana')
    pl.gcf().subplots_adjust(bottom=0.45)
  pl.ylim(min(bad_data) - 1, max(good_data) + 1) 
  pl.show()
  
  
  
  '''good_tag_vals = []
  bad_tag_vals = []
  for tag in all_tags:
    good_tag_vals.append(good_hist[tag])
    bad_tag_vals.append(bad_hist[tag])
  
  for h in [good_hist, bad_hist]:
    k = h.keys()
    v = [h[key] for key in k]
    h = {}
    z = zip(k, v)
    z.sort(key=lambda x: x[1], reverse=True)
    z = z[:20]
    z.sort(key=lambda x: x[0], reverse=False)
    for k,v in z:
      h[k] = v
    ks = [x[0] for x in z]
    vs = [x[1] for x in z]
    X = range(len(ks))
    pl.bar(X, vs, align='center', width=0.5)
    pl.xticks(rotation=90)
    pl.xticks(X, ks)
    ymax = max(vs) + 1
    pl.ylim(0, ymax)
    pl.show()'''
  
  '''
  # Score the tags (todo: scale down if same photographer uses tag a lot?)
  tag_scores = {}
  for img in rated_images:
    for tag in img.tags:
      if tag not in tag_scores:
        tag_scores[tag] = 0.0
      tag_scores[tag] += float(img.rating) / len(img.tags)
  
  # Read all images
  all_images = []
  image_paths = get_images_in_folder(folder)
  for path in image_paths:
    img = ImageInfo(path, 1000)  # rating doesn't matter for these
    all_images.append(img)
  
  # Score all images
  for img in all_images:
    img.score = 0
    for tag in img.tags:
      if tag in tag_scores:
        img.score += tag_scores[tag]
  
  # Print top 5 images
  print "Best 5"
  all_images.sort(key=lambda x: x.score, reverse=True)
  for img in all_images[:5]:
    pass
    print img.filename, img.score
    #print img.filename, img.score, img.tags, "\n"
  print "Worst 5"
  for img in all_images[-6:-1]:
    pass
    print img.filename, img.score
    #print img.filename, img.score, img.tags, "\n"
  
  print "Best 5 tags"
  tag_score_list = zip(tag_scores.keys(), tag_scores.values())
  tag_score_list.sort(key=lambda x: x[1], reverse=True)
  for tag_score in tag_score_list[:5]:
    print tag_score[0], tag_score[1]
  print "Worst 5 tags"
  for tag_score in tag_score_list[-6:-1]:
    print tag_score[0], tag_score[1]
  
  # Write the ratings to 'img_scores.txt'
  #with open(folder + 'img_scores.txt', 'w') as f:
  '''

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This script analyses tag usefulness')
  parser.add_argument('-f', '--folder', help='Image folder name', required=True)
  args = parser.parse_args()
  
  main('./' + args.folder + '/')