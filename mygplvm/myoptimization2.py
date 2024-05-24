import open3d as o3d
import trimesh
import skimage
from scipy.fftpack import idct
import numpy as np
import pytorch_volumetric as pv
import pytorch_kinematics as pk

def idctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = idct(x, axis=i, norm=norm)
    return x

class MyOptimization2:
    def __init__(self) -> None:
        self.gplvm = None

    def reconstruct(self, gplvm, latent_init, dct_resolution, voxel_resolution):
        dct_25_flat = gplvm.recall(latent_init)
        dct_25 = np.array(dct_25_flat).reshape((dct_resolution, dct_resolution, dct_resolution))
        dct_full = np.pad(dct_25, ((0, voxel_resolution - dct_resolution), (0, voxel_resolution - dct_resolution), (0, voxel_resolution - dct_resolution)))
        voxels_idct = idctn(dct_full)
        voxels_idct = np.pad(voxels_idct, ((1, 1), (1, 1), (1, 1)), constant_values=(1, 1))
        
        mc_vertices, mc_faces, mc_normals, _ = skimage.measure.marching_cubes(voxels_idct, level=0)
        mc_mesh = trimesh.Trimesh(vertices=mc_vertices, faces=mc_faces, normals=mc_normals)
        mc_mesh.apply_translation([-((voxel_resolution - 4)//2), -((voxel_resolution - 4)//2), -((voxel_resolution - 4)//2)])

        # convert to open3d
        o3d_mesh = mc_mesh.as_open3d

        # convert to sdf
        obj = pv.MeshObjectFactory(mesh=o3d_mesh)
        sdf = pv.MeshSDF(obj)
        return sdf
