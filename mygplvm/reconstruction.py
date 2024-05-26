import numpy as np
from scipy.fftpack import dct, idct
import skimage
import trimesh
from math import sin, cos, tan, radians


from mygplvm.mygplvm import MyGPLVM

class Reconstruction():
    def __init__(self, voxel_resolution = 64, dct_resolution = 25) -> None:
        self.dct_resolution = dct_resolution
        self.voxel_resolution = voxel_resolution

    def fit_from_obj(self):
        pass

    def fit_from_sdf(self, sdfs, voxel_resolution=64):
        self.voxel_resolution = voxel_resolution
        dct_trimmeds = []
        for sdf in sdfs:
            dct_full = dctn(sdf)
            dct_trimmeds.append(dct_full[:self.dct_resolution, :self.dct_resolution, :self.dct_resolution])
        
        self.gplvm = MyGPLVM()
        dct_flatteneds = []
        for dct_trimmed in dct_trimmeds:
            dct_flatteneds.append(dct_trimmed.flatten())
        return self.gplvm.simple_gplvm(dct_flatteneds, 'vehicle_models', latent_dimension = 2, epsilon = 0.001, maxiter = 50)

    def reconstruct_from_x(self, x):
        dct_n_flat = self.gplvm.recall(x)
        dct_n = np.array(dct_n_flat).reshape((self.dct_resolution, self.dct_resolution, self.dct_resolution))
        dct_full = np.pad(dct_n, ((0, self.voxel_resolution - self.dct_resolution), (0, self.voxel_resolution - self.dct_resolution), (0, self.voxel_resolution - self.dct_resolution)))
        
        voxels_idct = idctn(dct_full)
        voxels_idct = np.pad(voxels_idct, ((1, 1), (1, 1), (1, 1)), constant_values=(1, 1))
        mc_vertices, mc_faces, mc_normals, _ = skimage.measure.marching_cubes(voxels_idct, level=0)
        mc_mesh = trimesh.Trimesh(vertices=mc_vertices, faces=mc_faces, normals=mc_normals)
        translate_offset = -(self.voxel_resolution - 4) // 2
        mc_mesh.apply_translation([translate_offset, translate_offset, translate_offset])
        mc_mesh.apply_transform([
            [cos(radians(90))  , 0, sin(radians(90))  , 0],
            [0                      , 1, 0                      , 0],
            [-sin(radians(90)) , 0, cos(radians(90))  , 0],
            [0                      , 0, 0                      , 1]
        ])
        return mc_mesh

def dctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = dct(x, axis=i, norm=norm)
    return x

def idctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = idct(x, axis=i, norm=norm)
    return x