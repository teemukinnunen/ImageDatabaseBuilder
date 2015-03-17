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
  all_tags = all_tags[:20]
  show_tags = True
  weight_freq = True
  use_percentage = False
  
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
  pl.title('Avainsanojen frekvenssi')
  if weight_freq:
    pl.title('Avainsanojen painotettu frekvenssi')
  if use_percentage:
    good_total = 1.0 * sum(good_hist.values())
    bad_total = 1.0 * sum(bad_hist.values())
    good_data = [d / good_total for d in good_data]
    bad_data = [d / bad_total for d in bad_data]
    pl.title('Avainsanojen suhteellinen frekvenssi')
  ax = pl.subplot(111)
  ax.bar(range(len(all_tags)), good_data, width=1, color='b')
  ax.bar(range(len(all_tags)), bad_data, width=1, color='r')
  if use_percentage:
    pl.ylabel('suhteellinen frekvenssi')
  else:
    pl.ylabel('frekvenssi')
  pl.xlabel('avainsanan indeksi')
  if show_tags:
    pl.xticks(range(len(all_tags)), all_tags)
    pl.xticks(rotation=90)
    pl.xlabel('avainsana')
    pl.gcf().subplots_adjust(bottom=0.45)
  if use_percentage:
    pl.ylim(min(bad_data) - 0.02, max(good_data) + 0.02)
  else:
    pl.ylim(min(bad_data) - 1, max(good_data) + 1) 
  pl.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This script analyses tag usefulness')
  parser.add_argument('-f', '--folder', help='Image folder name', required=True)
  args = parser.parse_args()
  
  main('./' + args.folder + '/')