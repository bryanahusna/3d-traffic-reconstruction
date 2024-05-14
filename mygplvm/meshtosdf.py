print('checkpoint1')
from mesh_to_sdf import mesh_to_voxels
import trimesh
import numpy as np
print('checkpoint2')

mesh = trimesh.load('Car.obj')

print('checkpoint3')
voxels = mesh_to_voxels(mesh, 120, pad=True)
print('checkpoint4')
voxels = np.array(voxels)
print('checkpoint5')
print(voxels.shape)
print('checkpoint6')
