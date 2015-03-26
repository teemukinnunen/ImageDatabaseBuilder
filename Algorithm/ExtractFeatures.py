import numpy as np
import cv2
import json
from Utilities import *

def extract_features(image_path):
  img = cv2.imread(image_path, cv2.IMREAD_COLOR)
  surf = cv2.SIFT(400)
  kp, des = surf.detectAndCompute(img, None)
  print "extracted", image_path

def main():
  folder = get_folder_argument()
  base_folder = '../' + folder + '/'
  
  (image_paths, metadata_paths) = get_image_paths(base_folder)
  all_features = []
  for i in range(len(image_paths)):
    fs = extract_features(image_paths[i])
    all_features.append(fs)
  
if __name__ == '__main__':
  main()