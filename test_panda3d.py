from math import pi, sin, cos
from direct.showbase.ShowBase import ShowBase
from direct.task import Task


# class MyApp(ShowBase):
#     def __init__(self):
#         ShowBase.__init__(self)
#
#         # Load the environment model.
#         self.scene = self.loader.loadModel("blender_samples/Maze.bam")
#         # Reparent the model to render.
#         self.scene.reparentTo(self.render)
#         # Apply scale and position transforms on the model.
#         self.scene.setScale(0.25, 0.25, 0.25)
#         self.scene.setPos(-8, 42, 0)
#
#         # Add the spinCameraTask procedure to the task manager.
#         self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
#
#     # Define a procedure to move the camera.
#     def spinCameraTask(self, task):
#         angleDegrees = task.time * 6.0
#         angleRadians = angleDegrees * (pi / 180.0)
#         self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
#         self.camera.setHpr(angleDegrees, 0, 0)
#         return Task.cont


class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("blender_samples/Maze.bam")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(1.0, 1.0, 1.0)
        self.scene.setPos(0, 22, -5) # just places your object. As a viewer you are at (0,0,0). Good choice is 0, 42, -10
        # Look at https://docs.panda3d.org/1.10/python/programming/camera-control/default-camera-driver
        self.disableMouse() # If you want to use the camera pos you need to uncomment here
        # self.oobe() # Use this to have a "God-mode" camera viewing everything (including the placed camera below). Controls are back even if disableMouse is on
        self.camera.setPos(0,0,15) # you will render a scene with this so now it is not activated
        self.camera.setHpr(0, -40, 0) # where you are looking too. Rotates the camera around x,y,z axis. y: like looking up/down with your head. z: tilting your head


app = MyApp()
app.run()

