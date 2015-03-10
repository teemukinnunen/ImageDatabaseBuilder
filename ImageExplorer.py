import Tkinter
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join
import argparse, json
import random

class ImageExplorer(Tkinter.Tk):
  def __init__(self, parent, folder):
    Tkinter.Tk.__init__(self, parent)
    self.parent = parent
    self.folder = './' + folder + '/'
    self.protocol("WM_DELETE_WINDOW", self.save_and_exit)
    self.initialize()
  
  def save_and_exit(self):
    self.save_ratings()
    self.quit()
    
  def init_file_paths(self):
    base_path = self.folder
    all_file_names = [f for f in listdir(base_path) if isfile(join(base_path, f))]
    all_file_names.sort()
    self.paths = [base_path + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    random.shuffle(self.paths)

  def update_rating(self):
    self.lbl_rating['text'] = 'Rating: ' + self.rating_to_text(self.get_current_rating())
    
  def update_metadata(self):
    metadata_path = self.paths[self.image_index] + '.txt'
    with open(metadata_path, 'r') as f:
      self.current_metadata = json.load(f)
    md = self.current_metadata
    self.update_rating()
    self.lbl_title['text'] = 'Title: ' + md['title']
    self.lbl_description['text'] = 'Description: ' + md['description']
    self.lbl_tags['text'] = 'Tags: ' + ', '.join(md['tags'])
    self.lbl_owner['text'] = 'Owner: ' + md['owner']
    self.lbl_id['text'] = 'ID: ' + md['id']
    self.lbl_gps['text'] = 'GPS: ' + ', '.join(md['gps'])
    
  def update_image(self):
    image_path = self.paths[self.image_index] + '.jpg'
    img = Image.open(image_path)
    self.current_photo = ImageTk.PhotoImage(img)
    self.image_label = Tkinter.Label(image=self.current_photo)
    self.image_label.pack(side="top")
    self.image_label.grid(column=0, columnspan=3, row=0, sticky='EW')
    
  def init_image(self):
    self.update_image()
    self.update_metadata()
  
  def prev_image(self, *args):
    self.image_index -= 1
    self.image_index %= len(self.paths)
    self.init_image()
    
  def next_image(self, *args):
    self.image_index += 1
    self.image_index %= len(self.paths)
    self.init_image()
  
  def get_current_path(self):
    return self.paths[self.image_index]
  
  def rating_to_text(self, rating):
    if rating == 0:
      return 'Bad'
    elif rating == 1:
      return 'Good'
    if rating == None:
      return 'Not set'
  
  def get_current_rating(self):
    path = self.get_current_path()
    if path in self.ratings:
      return self.ratings[path]
    else:
      return None
  
  def set_current_rating(self, rating):
    self.ratings[self.get_current_path()] = rating
    self.update_rating()
    self.next_image()
  
  def init_ratings(self):
    self.ratings = {}
    filepath = self.folder + 'ratings.txt'
    if isfile(filepath):
      with open(filepath, 'r') as f:
        try:
          self.ratings = json.load(f)
        except:
          pass
  
  def save_ratings(self):
    filepath = self.folder + 'ratings.txt'
    with open(filepath, 'w') as f:
      json.dump(self.ratings, f)
  
  # Event when rating the image
  def rate_image(self, event):
    print event.type
    if event.keysym == 'plus':
      self.set_current_rating(1)
    elif event.keysym == 'minus':
      self.set_current_rating(0)
    elif event.keysym == 'Delete':
      self.set_current_rating(None)
  
  # Initialize the widget
  def initialize(self):
    self.image_index = 0
    self.paths = []
    self.metadata = {}
    self.current_photo = None
    self.ratings = {}  # image filename -> 0 or 1
    
    self.init_file_paths()
    self.init_ratings()
  
    self.grid()
    
    # Image navigation buttons
    self.prev_button = Tkinter.Button(self, text=u"Prev image", command=self.prev_image)
    self.prev_button.grid(column=0, row=8)
    self.next_button = button = Tkinter.Button(self, text=u"Next image", command=self.next_image)
    self.next_button.grid(column=1, row=8)
    
    # Metadata labels
    self.lbl_rating = Tkinter.Label(self)
    self.lbl_rating.grid(column=0, row=1)
    self.lbl_title = Tkinter.Label(self)
    self.lbl_title.grid(column=0, row=2)
    self.lbl_description = Tkinter.Label(self)
    self.lbl_description.grid(column=0, row=3)
    self.lbl_tags = Tkinter.Label(self)
    self.lbl_tags.grid(column=0, row=4)
    self.lbl_owner = Tkinter.Label(self)
    self.lbl_owner.grid(column=0, row=5)
    self.lbl_id = Tkinter.Label(self)
    self.lbl_id.grid(column=0, row=6)
    self.lbl_gps = Tkinter.Label(self)
    self.lbl_gps.grid(column=0, row=7)
    
    # Hotkey bindings
    self.bind('<KeyRelease-Left>', self.prev_image)
    self.bind('<KeyRelease-Right>', self.next_image)
    self.bind('<KeyRelease-KP_Add>', self.rate_image)
    self.bind('<KeyRelease-plus>', self.rate_image)
    self.bind('<KeyRelease-KP_Subtract>', self.rate_image)
    self.bind('<KeyRelease-minus>', self.rate_image)
    self.bind('<KeyRelease-Delete>', self.rate_image)
    
    self.init_image()
    
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='This script displays images for manual review.')
  parser.add_argument('-f', '--folder', help='Image folder name', required=True)
  args = parser.parse_args()
  
  app = ImageExplorer(None, args.folder)
  #app.resizable(width=False, height=False)
  #app.minsize(width=500, height=500)
  #app.maxsize(width=500, height=500)
  app.title('Image Explorer')
  app.mainloop()
