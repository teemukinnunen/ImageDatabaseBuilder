# This script contains common helper functions
from os import listdir
from os.path import isfile, join
import os, sys, argparse

def get_folder_argument():
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  args = parser.parse_args()
  return args.folder_name

def get_image_paths(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [folder + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    images = [f + '.jpg' for f in files]
    metadata = [f + '.txt' for f in files]
    return (images, metadata)

def save_features(folder_name, features):
  with open('./features/' + folder_name + '.txt', 'w') as f:
    f.write("asd\n")
  

def load_features(folder_name):
  features = []
  return features
  