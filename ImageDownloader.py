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

def save_image_and_data(image_folder, url, title, id, owner, tags, description, gps, datetaken):
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
  metadata['id'] = id
  metadata['owner'] = owner
  metadata['tags'] = tags
  metadata['description'] = description
  metadata['gps'] = gps
  metadata['datetaken'] = datetaken
  text_path = image_folder + name.split('.')[0] + '.txt'
  with open(text_path, 'w') as f:
    json.dump(metadata, f)  

def main(argv):
  with open('API key here.txt', 'r') as f:
    lines = f.read().splitlines()
    if len(lines) >= 2:
      flickr.API_KEY = lines[0]
      flickr.API_SECRET = lines[1]
    else:
      print "Insert API_KEY and API_SECRET to 'API key here.txt'"
  
  # Script parameters
  save_images = True
  image_folder = None
  image_dl_count = None
  
  parser = argparse.ArgumentParser(description='This script downloads images from Flickr.')
  parser.add_argument('-f', '--folder_name', help='Save folder name', required=True)
  parser.add_argument('-a', '--photo_amount', help='Amount of images to download', required=True)
  args = parser.parse_args()
  
  image_folder = './' + args.folder_name + '/'
  image_dl_count = int(args.photo_amount)
  
  print "Parameters:", image_folder, image_dl_count
  
  new_york_id = 2459115
  helsinki_id = 565346
  per_page = min(image_dl_count, 300)
  photos = []
  for page in range(image_dl_count / per_page):
    #page_photos = flickr.photos_search(woe_id = helsinki_id, has_geo = 1, per_page = per_page, page = page, min_taken_date=315532800)
    page_photos = flickr.photos_search(woe_id = helsinki_id, has_geo = 1, per_page = per_page, page = page)
    photos.extend(page_photos)
  print "Found", len(photos), "photos"
  urls = []
  failed_downloads = 0
  for i in range(len(photos)):
    try:
      print "Downloading photos... {}/{}\r".format(i+1, len(photos)), 
      photo = photos[i]
      url = photo.getMedium()
      
      tags = []
      if photo.__getattr__('tags') != None:
        tags = [tag.text.encode('utf-8') for tag in photo.__getattr__('tags')]
      
      title = photo.title.encode('utf-8')
      description = photo.description.encode('utf-8')
      gps = photo.getLocation()
      id = photo.id.encode('utf-8')
      owner = photo.owner.id.encode('utf-8')
      datetaken = '' #photo.datetaken
      #print datetaken
      if save_images:
        save_image_and_data(image_folder, url, title, id, owner, tags, description, gps, datetaken)
    except:
      print "Exception while downloading photo at index {}".format(i)
      failed_downloads += 1
    '''exif = photo.getExif()
    for tag in exif.tags:
      print '%s: %s' % (tag.label, tag.raw)'''
  
  print "Photos:", len(photos), "Successful downloads:", len(photos) - failed_downloads
  seen = set()
  uniq = [x for x in urls if x not in seen and not seen.add(x)]
  print "Duplicates: ", len(urls) - len(uniq)
  
  
if __name__ == '__main__':
  main(sys.argv[1:])