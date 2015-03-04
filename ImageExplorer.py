import Tkinter
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join
import argparse, json

class ImageExplorer(Tkinter.Tk):
  def __init__(self, parent, folder):
    Tkinter.Tk.__init__(self, parent)
    self.parent = parent
    self.folder = './' + folder + '/'
    self.initialize()
    
  def init_file_paths(self):
    base_path = self.folder
    all_file_names = [f for f in listdir(base_path) if isfile(join(base_path, f))]
    all_file_names.sort()
    self.paths = [base_path + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]

  def update_image(self):
    image_path = self.paths[self.image_index] + '.jpg'
    img = Image.open(image_path)
    self.current_photo = ImageTk.PhotoImage(img)
    self.image_label = Tkinter.Label(image=self.current_photo)
    self.image_label.pack()
    metadata_path = self.paths[self.image_index] + '.txt'
    with open(metadata_path, 'r') as f:
      self.current_metadata = json.load(f)
    print self.current_metadata
    
  def prev_image(self):
    self.image_index -= 1
    self.image_index %= len(self.paths)
    self.update_image()
    
  def next_image(self):
    self.image_index += 1
    self.image_index %= len(self.paths)
    self.update_image()
  
  def initialize(self):
    self.image_index = 0
    self.paths = []
    self.current_photo = None
    
    self.init_file_paths()
  
    self.grid()
    
    self.prev_button = Tkinter.Button(self, text=u"Prev image", command=self.prev_image)
    self.prev_button.grid(column=0, row=1)
    self.next_button = button = Tkinter.Button(self, text=u"Next image", command=self.next_image)
    #button.pack()
    self.next_button.grid(column=1, row=1)
    
    #self.image_label = Tkinter.Label()
    self.update_image()
    
    self.image_label.grid(column=0, row=0, sticky='EW')
        
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='This script displays images for manual review.')
  parser.add_argument('-f', '--folder', help='Image folder name', required=True)
  args = parser.parse_args()
  
  app = ImageExplorer(None, args.folder)
  app.title('Image Explorer')
  app.mainloop()
