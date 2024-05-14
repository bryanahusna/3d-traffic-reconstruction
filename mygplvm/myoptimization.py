from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin, pi, log, exp
from scipy.fftpack import idct
import skimage
import trimesh
import pyrender
from PIL import Image

from scipy.optimize import fmin_cg, fmin_tnc, fmin_slsqp  # Non-linear SCG
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA  # For X initialization
from sklearn.preprocessing import StandardScaler  # To standardize data
from sklearn.gaussian_process import kernels
from sklearn.datasets import load_wine
import time
from tqdm import tqdm
# from fake_dataset import generate_observations, plot
from datetime import datetime

def idctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = idct(x, axis=i, norm=norm)
    return x

def likelihood(var, *args):
    # YYT, N, D, latent_dimension, = args
    input_image, gplvm, dct_resolution, voxel_resolution, = args

    # var: [latent_1, latent_2, theta_y, T_x, T_y, T_z]
    latent_1 = var[0]
    latent_2 = var[1]
    theta_y = var[2] / 10
    T_x = var[3]
    T_y = var[4]
    T_z = var[5]

    mesh_pose = np.array([
        [cos(theta_y), 0, sin(theta_y), T_x],
        [0, 1, 0, T_y],
        [-sin(theta_y), 0, cos(theta_y), T_z],
        [0, 0, 0, 1]
    ])
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 62],
        [0, 0, 0, 1]
    ])

    # GPLVM Recall
    dct_25_flat = gplvm.recall([latent_1, latent_2])
    dct_25 = np.array(dct_25_flat).reshape((dct_resolution, dct_resolution, dct_resolution))
    dct_full = np.pad(dct_25, ((0, voxel_resolution - dct_resolution), (0, voxel_resolution - dct_resolution), (0, voxel_resolution - dct_resolution)))
    voxels_idct = idctn(dct_full)
    # print(np.sum(np.abs(voxels_idct)))
    # if np.sum(np.abs(voxels_idct)) < 1e-12:
    #     return 0
    try:
        mc_vertices, mc_faces, mc_normals, _ = skimage.measure.marching_cubes(voxels_idct, level=0)
        mc_mesh = trimesh.Trimesh(vertices=mc_vertices, faces=mc_faces, normals=mc_normals)
        mc_mesh.apply_translation([-28, -28, -28])

        # Render silhouette
        mesh_pyrender = pyrender.Mesh.from_trimesh(mc_mesh, smooth=False)
        scene = pyrender.Scene()
        # Lighting
        pl = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        # Camera
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1)
        
        # Scene
        scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
        scene.add(mesh_pyrender, pose=mesh_pose)
        scene.add(pl, pose=np.eye(4))
        scene.add(pc, pose=camera_pose)
        
        # pyrender.Viewer(scene, use_raymond_lighting=True)
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)
        gray = np.asarray(Image.fromarray(color).convert('L'))

        # Calculate the energy
        energy = 0
        for i in range(512):
            for j in range(512):
                energy += (gray[i][j]/255 * input_image[i][j]/255) + ( (1 - gray[i][j]/255) * (1 - input_image[i][j]/255))
        loss = -energy / (512 * 512)
        print(var, '=>', loss)
        return loss
    except:
        return 0
    

class MyOptimization:
    def __init__(self):
        self.iteration = 0
        self.name = None

        # parameters
        self.gplvm = None
        self.latent_1 = None
        self.latent_2 = None
        self.theta_y = None
        self.T_x = None
        self.T_y = None
        self.T_z = None

    def save_vars(self, var):
        self.iteration += 1
        print(f'Iteration {self.iteration}')
        # if self.iteration%10 == 0:
        #     timestamp = str(datetime.now()).replace(" ", "_")
        #     np.save("results-reconstruct/" + str(self.name) + "_" + timestamp + ".npy", var)

    def reconstruct(self, input_image, gplvm, dct_resolution, voxel_resolution, experiment_name="experiment", epsilon = 0.001, maxiter = 10):
        ''' Implementation of GPLVM algorithm, returns data in latent space
        '''
        global name
        name = experiment_name
        
        # Initialize X through PCA
        # First var approximation
        var = [-14, -2, pi * 10, 0, 0, 0]


        t1 = time.time()

        # Optimization
        # var = fmin_cg(likelihood, var, args=tuple((input_image, gplvm, dct_resolution, voxel_resolution,)), epsilon = epsilon, disp=True, callback=self.save_vars, maxiter=maxiter)
        # var = fmin_tnc(likelihood, var, bounds=((-40, 40), (-20, 30), (0, 2*10*pi), (50, 50), (50, 50), (50, 50)), args=tuple((input_image, gplvm, dct_resolution, voxel_resolution,)), epsilon = epsilon, disp=True, callback=self.save_vars, maxfun=maxiter, approx_grad=True)
        var = fmin_slsqp(likelihood, var, bounds=((-40, 40), (-20, 30), (0, 2*10*pi), (-50, 50), (-50, 50), (-50, 50)), args=tuple((input_image, gplvm, dct_resolution, voxel_resolution,)), epsilon = epsilon, disp=True, callback=self.save_vars, iter=maxiter)
        
        print("time:", time.time() - t1)

        var = list(var)
        # np.save("results-reconstruct/" + str(name) + "_final.npy", var)

        self.latent_1 = var[0]
        self.latent_2 = var[1]
        self.theta_y = var[2] / 10
        self.T_x = var[3]
        self.T_y = var[4]
        self.T_z = var[5]
        print(var)

