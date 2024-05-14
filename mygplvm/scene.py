import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw
import glfw.GLFW as GLFW_CONSTANT

def initialize_glfw():
  glfw.init()
  glfw.window_hint(GLFW_CONSTANT.GLFW_CONTEXT_VERSION_MAJOR, 3)
  glfw.window_hint(GLFW_CONSTANT.GLFW_CONTEXT_VERSION_MINOR, 3)
  glfw.window_hint(
    GLFW_CONSTANT.GLFW_OPENGL_PROFILE,
    GLFW_CONSTANT.GLFW_OPENGL_CORE_PROFILE
  )
  glfw.window_hint(
    GLFW_CONSTANT.GLFW_OPENGL_FORWARD_COMPAT,
    GLFW_CONSTANT.GLFW_TRUE
  )
  glfw.window_hint(GLFW_CONSTANT.GLFW_DOUBLEBUFFER, GL_FALSE)

  window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "My Game", None, None)
  glfw.make_context_current(window)
  glfw.set_input_mode(
    window,
    GLFW_CONSTANT.GLFW_CURSOR,
    GLFW_CONSTANT.GLFW_CURSOR_HIDDEN
  )

  return window

from meshes.car_regular_mesh import CarRegularMesh
from meshes.car_voxel_mesh import CarVoxelMesh

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_END = 1

class Scene:
    def __init__(self) -> None:
        self.mode = 'mesh'  # 'mesh' or 'voxel'

        self.cars = []
        self.textures = []
        self.shaders = {
            'car_regular_mesh': self.createShader('meshes/car_regular.vert', 'meshes/car_regular.frag')
        }
        self.lights = [] 

        self.cars.append(Car('objs/2005 toyota - Salin.obj', self.shaders['car_regular_mesh'], [0,0,0], [0,0]))

    def createShader(self, vertexFilePath, fragmentFilePath):
        with open(vertexFilePath, 'r') as f:
            vertex_src = f.readlines()
        
        with open(fragmentFilePath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )

        return shader

class Car:
  def __init__(self, filename, shader, position, eulers) -> None:
      self.position = np.array(position, dtype=np.float32)
      self.eulers = np.array(eulers, dtype=np.float32)

      self.regular_mesh = CarRegularMesh(filename, shader)
      #self.voxel_mesh = CarVoxelMesh(filename)

if __name__ == '__main__':
    initialize_glfw()
    glClearColor(0.1, 0.2, 0.2, 1)
    glEnable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    scene = Scene()
