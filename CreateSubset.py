from os import listdir
from os.path import isfile, join
import os, sys, argparse

def get_folder_arguments():
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  args = parser.parse_args()
  return (args.input_folder, args.output_folder

def get_image_paths(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [folder + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    images = [f + '.jpg' for f in files]
    metadata = [f + '.txt' for f in files]
    return (images, metadata)

def get_bb_images(image_paths, metadata_paths, latlong, w, h):
  for i in range(len(image_paths)):
    with open(metadata_paths[i], 'r') as f
      md = json.load(f)
      gps = [float(md['gps'][0]), float(md['gps'][1]
    
def main():
  folder_name = get_folder_argument()
  folder = './' + folder_name + '/'
  (image_paths, metadata_paths) = get_image_paths(folder)
  w = 0.005
  h = 0.005
  rautatietori = [24.57, 60.10]
  subset = get_bb_images(image_paths, metadata_paths, rautatietori, w, h)
  
if __name__ == '__main__':
  main()