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