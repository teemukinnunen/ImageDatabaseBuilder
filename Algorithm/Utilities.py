# This script contains common helper functions
from os import listdir
from os.path import isfile, join
import os, sys, argparse
import shutil
import os

def copy_images(input_folder, output_folder, img_paths, md_paths):
  if input_folder == output_folder:
    print "input folder can't be same as output folder!"
    return
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  print "Copying {} photos".format(len(img_paths))
  for i in range(len(img_paths)):
    shutil.copy2(input_folder + img_paths[i], output_folder + img_paths[i])
    shutil.copy2(input_folder + md_paths[i], output_folder + md_paths[i])

def get_filename(path): # probably doesn't work on other than Windows
  if '/' not in path:
    return path
  else:
    return path.split('/')[-1]

def get_folder(path):
  return path[:path.rfind('/')+1]
  
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
