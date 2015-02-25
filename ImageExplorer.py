import Tkinter

class ImageExplorer(Tkinter.Tk):
  def __init__(self, parent):
    Tkinter.Tk.__init__(self, parent)
    self.parent = parent
    self.initialize()
    
  def initialize(self):
    self.grid()
    
    #self.entry = Tkinter.Entry(self)
    #self.entry.grid(column=0, row=0, sticky='EW')
    
    button = Tkinter.Button(self, text=u"Prev image")
    button.grid(column=0, row=1)
    button = Tkinter.Button(self, text=u"Next image")
    button.grid(column=1, row=1)
    
    image_label = Tkinter.Label(self, anchor="w")
    image_label.grid(column=0, row=0, sticky='EW')
    
    path = './Images/' + '15993263284_74e8ace135_m.jpg'
    image = Tkinter.PhotoImage(file=path)
    
if __name__ == "__main__":
  app = ImageExplorer(None)
  app.title('Image Explorer')
  app.mainloop()
