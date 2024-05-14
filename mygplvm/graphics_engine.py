from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr

class GraphicsEngine:
  def __init__(self) -> None:
    #self.wood_texture = Material('gfx/wood.jpeg')
    #self.cube_mesh = Mesh('models/2005 toyota - Salin.obj')

    # initialize opengl
    glClearColor(0.1, 0.2, 0.2, 1)
    glEnable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    self.shader = self.createShader('shaders/vertex.txt', 'shaders/fragment.txt')
    glUseProgram(self.shader)
    glUniform1i(glGetUniformLocation(self.shader, 'imageTexture'), 0)

    projection_transform = pyrr.matrix44.create_perspective_projection(
        fovy = 45, aspect = 640/480,
        near = 0.1, far = 100, dtype=np.float32
    )
    glUniformMatrix4fv(
        glGetUniformLocation(self.shader, 'projection'),
        1, GL_FALSE, projection_transform
    )

    self.modelMatrixLocation = glGetUniformLocation(self.shader, 'model')
    self.viewMatrixLocation = glGetUniformLocation(self.shader, 'view')
    self.lightLocation = {
      'position': [
        glGetUniformLocation(self.shader, f'Lights[{i}].position')
        for i in range(8)
      ],
      'color': [
        glGetUniformLocation(self.shader, f'Lights[{i}].color')
        for i in range(8)
      ],
      'strength': [
        glGetUniformLocation(self.shader, f'Lights[{i}].strength')
        for i in range(8)
      ]
    }
    self.cameraPosLoc = glGetUniformLocation(self.shader, 'cameraPosition')

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

