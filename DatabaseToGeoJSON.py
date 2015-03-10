import argparse, json
from os import listdir
from os.path import isfile, join

def get_metadata_files(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [folder + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    files = [f + '.txt' for f in files]
    return files

def main(folder):
  metadata_files = get_metadata_files('./' + folder + '/')
  points = []
  for filename in metadata_files:
    with open(filename, 'r') as f:
      md = json.load(f)
      point = md['gps']
      point = [float(point[0]), float(point[1])]
      points.append(point)
  
  multipoint = {}
  multipoint['type'] = 'MultiPoint'
  multipoint['coordinates'] = points
  
  mp_feature = {}
  mp_feature['type'] = 'Feature'
  mp_feature['geometry'] = multipoint
  
  feature_collection = {}
  feature_collection['type'] = 'FeatureCollection'
  feature_collection['features'] = [mp_feature]
  
  with open('./' + folder + '.json', 'w') as f:
    json.dump(feature_collection, f, indent=2)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This script turns metadata of images into GeoJSON points')
  parser.add_argument('-f', '--folder', help='Image folder name', required=True)
  args = parser.parse_args()
  
  main(args.folder)