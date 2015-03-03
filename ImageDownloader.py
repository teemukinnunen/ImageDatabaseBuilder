import flickr
import urllib, urlparse
import os
import sys
import Image
import shutil
import StringIO
import collections

def get_image_name(url):
  return url.split('/')[-1]

def download_image(url):
  data = urllib.urlopen(url).read()
  s = StringIO.StringIO(data)
  image = Image.open(s)
  return image

def save_image_and_data(url, description, tags_string):
  # Download image
  image = download_image(url)
  name = get_image_name(url)

  # Save image
  image_path = './Images/' + name
  image.save(image_path)
  
  # Save tags in a text file
  text_path = './Images/' + name.split('.')[0] + '.txt'
  #print "File, tags:", text_path, tags_string
  with open(text_path, 'w') as text_file:
    text_file.write(tags_string)
  

def main():
  flickr.API_KEY = 'ba158eb66e7f9f3448a275079e6f38e4'
  flickr.API_SECRET = 'ac8257dabd7125da'
  
  save_images = True
  
  #content_type 1 = photos, 3 = 'other', 6 = photos and 'other', 7 = all
  #has_geo
  
  helsinki_id = 565346
  image_dl_count = 5000
  per_page = min(image_dl_count, 300)
  photos = []
  for page in range(image_dl_count / per_page):
    page_photos = flickr.photos_search(woe_id = helsinki_id, has_geo = 1, per_page = per_page, page = page)
    photos.extend(page_photos)
  
  urls = []
  for photo in photos:
    url = photo.getSmall()
    
    tags = photo.__getattr__('tags')
    tags_string = ''
    if tags != None:
      tags_string = ', '.join([tag.text.encode('utf-8') for tag in tags])
    
    title = photo.__getattr__('title')
    description = photo.__getattr__('description')
    #print "Title:", title.encode('utf-8'), "Description:", description.encode('utf-8')
    
    if save_images:
      save_image_and_data(url, description, tags_string)
    
    '''exif = photo.getExif()
    for tag in exif.tags:
      print '%s: %s' % (tag.label, tag.raw)'''
    
  print "\nPhotos:", len(photos), "Image download count:", image_dl_count
  seen = set()
  uniq = [x for x in urls if x not in seen and not seen.add(x)]
  print "Duplicates: ", len(urls) - len(uniq)
  
  
if __name__ == '__main__':
  main()