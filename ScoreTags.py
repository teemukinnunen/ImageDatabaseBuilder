#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse, json
import math
import string
from os import listdir
from os.path import isfile, join
import pylab as pl
from math import sqrt

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

def main(folder, norm):
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
  
  # HISTOGRAM PLOTTING
  all_tags.sort(key=lambda tag: good_hist[tag] + bad_hist[tag], reverse=True)
  all_tags = all_tags[:200]
  show_tags = False
  fig = pl.figure()
  good_data = [good_hist[tag] for tag in all_tags]
  bad_data = [-bad_hist[tag] for tag in all_tags]
  
  # Normalization
  if norm == 'L1':
    good_sum = 1.0 * sum(good_data)
    bad_sum = 1.0 * -sum(bad_data)
    good_data = [d / good_sum for d in good_data]
    bad_data = [d / bad_sum for d in bad_data]
  if norm == 'L2':
    good_norm2 = 0.0
    for d in good_data:
      good_norm2 += d*d
    good_norm2 = sqrt(good_norm2)
    good_data = [d / good_norm2 for d in good_data]
    bad_norm2 = 0.0
    for d in bad_data:
      bad_norm2 += d*d
    bad_norm2 = sqrt(bad_norm2)
    bad_data = [d / bad_norm2 for d in bad_data]
    
  if norm == '0':
    pl.title('Avainsanojen frekvenssi')
  else:
    pl.title('Avainsanojen {}-normalisoitu frekvenssi'.format(norm))
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
  
  ax.legend((u'Hyödylliset kuvat', u'Hyödyttömät kuvat'), 'lower right')
  
  '''if norm == None:
    pl.ylim(min(bad_data) - 0.02, max(good_data) + 0.02)
  elif norm == 'L1':
    pl.ylim(min(bad_data) - 1, max(good_data) + 1) 
  elif norm == 'L2':'''
    
  pl.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This script analyses tag usefulness')
  parser.add_argument('-f', '--folder', help='Image folder name', required=True)
  parser.add_argument('-n', '--norm', help='Normalization method (0, L1, L2)', required=True)
  args = parser.parse_args()
  
  main('./' + args.folder + '/', args.norm)