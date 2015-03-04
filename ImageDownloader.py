import flickr
import urllib, urlparse
import os, sys, argparse
import Image
import shutil, json
import StringIO, collections

def get_image_name(url):
  return url.split('/')[-1]

def download_image(url):
  data = urllib.urlopen(url).read()
  s = StringIO.StringIO(data)
  image = Image.open(s)
  return image

def save_image_and_data(image_folder, url, title, id, owner, tags, description, gps):
  # Make folder if it doesn't exist
  if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    
  # Download image
  image = download_image(url)
  name = get_image_name(url)

  # Save image
  image_path = image_folder + name
  image.save(image_path)
  
  # Save metadata in a text file
  metadata = {}
  metadata['url'] = url
  metadata['title'] = title
  metadata['id'] = ''
  metadata['owner'] = ''
  metadata['tags'] = tags
  metadata['description'] = description
  metadata['gps'] = ''
  text_path = image_folder + name.split('.')[0] + '.txt'
  with open(text_path, 'w') as f:
    json.dump(metadata, f)
  
  #print "File, tags:", text_path, tags_string
  '''with open(text_path, 'w') as text_file:
    text_file.write(tags_string)'''
  

def main(argv):
  flickr.API_KEY = 'ba158eb66e7f9f3448a275079e6f38e4'
  flickr.API_SECRET = 'ac8257dabd7125da'
  
  # Script parameters
  save_images = None
  image_folder = None
  image_dl_count = None
  
  parser = argparse.ArgumentParser(description='This script downloads images from Flickr.')
  parser.add_argument('-s', '--save', help='Save bool', required=True)
  parser.add_argument('-f', '--folder', help='Save folder', required=True)
  parser.add_argument('-a', '--amount', help='Image dl amount', required=True)
  args = parser.parse_args()
  
  save_images = args.save == 'true'
  image_folder = './' + args.folder + '/'
  image_dl_count = int(args.amount)
  
  print "Parameters:", save_images, image_folder, image_dl_count
  
  helsinki_id = 565346
  per_page = min(image_dl_count, 300)
  photos = []
  for page in range(image_dl_count / per_page):
    page_photos = flickr.photos_search(woe_id = helsinki_id, has_geo = 1, per_page = per_page, page = page)
    photos.extend(page_photos)
  
  urls = []
  for photo in photos:
    url = photo.getSmall()
    
    tags = []
    if photo.__getattr__('tags') != None:
      tags = [tag.text.encode('utf-8') for tag in photo.__getattr__('tags')]
    
    title = photo.__getattr__('title').encode('utf-8')
    description = photo.__getattr__('description').encode('utf-8')
    #print "Title:", title.encode('utf-8'), "Description:", description.encode('utf-8')
    gps = photo.getLocation()
    if save_images:
      save_image_and_data(image_folder, url, title, 'id', 'owner', tags, description, gps)
    
    '''exif = photo.getExif()
    for tag in exif.tags:
      print '%s: %s' % (tag.label, tag.raw)'''
  
  print "\nPhotos:", len(photos), "Image download count:", image_dl_count
  seen = set()
  uniq = [x for x in urls if x not in seen and not seen.add(x)]
  print "Duplicates: ", len(urls) - len(uniq)
  
  
if __name__ == '__main__':
  main(sys.argv[1:])