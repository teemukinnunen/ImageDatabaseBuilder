import Tkinter
from PIL import Image, ImageTk
from os import listdir
from os.path import isfile, join

class ImageExplorer(Tkinter.Tk):

  def __init__(self, parent):
    Tkinter.Tk.__init__(self, parent)
    self.parent = parent
    self.initialize()
    
  def init_file_paths(self):
    base_path = './Images/'
    all_file_names = [f for f in listdir(base_path) if isfile(join(base_path, f))]
    all_file_names.sort()
    self.paths = [base_path + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]

  def update_image(self):
    image_path = self.paths[self.image_index] + '.jpg'
    self.current_photo = ImageTk.PhotoImage(Image.open(image_path))
    self.image_label.photo = self.current_photo
    tags_path = self.paths[self.image_index] + '.txt'
    self.current_tags = ''
    with open(tags_path, 'r') as tag_file:
      self.current_tags = tag_file.readline()
      print self.current_tags
    
  def prev_image(self):
    self.image_index -= 1
    self.update_image()
    
  def next_image(self):
    self.image_index += 1
    self.update_image()
  
  def initialize(self):
    self.image_index = 0
    self.paths = []
    self.current_photo = None
    
    self.init_file_paths()
  
    self.grid()
    
    button = Tkinter.Button(self, text=u"Prev image", command=self.prev_image)
    button.grid(column=0, row=1)
    button = Tkinter.Button(self, text=u"Next image", command=self.next_image)
    button.grid(column=1, row=1)
    
    self.image_label = Tkinter.Label()
    self.update_image()
    
    self.image_label.grid(column=0, row=0, sticky='EW')
        
    
if __name__ == "__main__":
  app = ImageExplorer(None)
  app.title('Image Explorer')
  app.mainloop()
