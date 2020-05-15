import os

import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase


def showbase():
    from direct.showbase.ShowBase import ShowBase
    p3d.load_prc_file_data('', 'window-type none')
    base = ShowBase()
    return base


def modelpath():
    # i = p3d.Filename.from_os_specific('./blender_samples/Maze.obj')
    # print('OK')
    return p3d.Filename.from_os_specific('./blender_samples/Maze.obj')


# showbase().loader.load_model(modelpath())

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        im = self.loader.load_model(modelpath())
        # Load the environment model.
        self.scene = self.loader.loadModel(im)#("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)


app = MyApp()
app.run()