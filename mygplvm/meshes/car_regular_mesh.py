from meshes.mesh import Mesh

class CarRegularMesh(Mesh):
  def __init__(self, filename: str, shader):
    super().__init__(filename, shader)
