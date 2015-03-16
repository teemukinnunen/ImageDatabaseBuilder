import numpy as np
import cv2
import os, sys, argparse
from os import listdir
from os.path import isfile, join
import json

# This is the actual clustering algorithm
def cluster_images(image_paths, metadata):
  img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)

def get_image_paths(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [folder + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    images = [f + '.jpg' for f in files]
    metadata = [f + '.txt' for f in files]
    return (images, metadata)
  
def main(argv):
  # Parse script parameters
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  args = parser.parse_args()
  folder = '../' + args.folder_name + '/'
  
  # Read metadata from files
  (images, metadata_paths) = get_image_paths(folder)
  metadata = []
  for path in metadata_paths:
    with open(path, 'r') as f:
      md = json.load(f)
      metadata.append(md)
  
  # Start the main algorithm
  cluster_images(images, metadata)

if __name__ == '__main__':
  main(sys.argv[1:])