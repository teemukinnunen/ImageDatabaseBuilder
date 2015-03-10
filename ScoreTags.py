import argparse, json

def main(folder):
  with open(folder + 'ratings.txt', 'r') as f:
    ratings = json.load(f)
  frequencies = {}
  scores = {}
  for filename in ratings:
    if ratings[filename] == None:
      continue
    with open(filename + '.txt') as f:
      md = json.load(f)
      tags = md['tags']
      title_tags = md['title'].split(' ') # append title words to tags (testing...)
      if len(title_tags) != 0:
        tags.extend(title_tags)
      for tag in tags:
        tag = tag.encode('utf-8')
        if tag not in frequencies:
          frequencies[tag] = 0
        frequencies[tag] += 1
        if tag not in scores:
          scores[tag] = 0
        rating = int(ratings[filename])
        if rating == 0:
          rating = -1
        scores[tag] += rating
        print tag, scores[tag]
  for tag in frequencies:
    print tag, frequencies[tag], scores[tag]
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This script analyses tag usefulness')
  parser.add_argument('-f', '--folder', help='Image folder name', required=True)
  args = parser.parse_args()
  
  main('./' + args.folder + '/')