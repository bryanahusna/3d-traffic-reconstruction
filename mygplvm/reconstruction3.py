from ultralytics import YOLO
import torch
import cv2
import numpy as np
import pyrender
import trimesh
import math
from math import sin, cos

import sys
sys.path.append('./mygplvm')
from mygplvm.myoptimization3 import MyOptimization3

class Reconstruction3:
    def __init__(self, mygplvm, voxel_resolution = 64, dct_resolution = 25) -> None:
        self.mygplvm = mygplvm
        self.mo3 = MyOptimization3()
        self.voxel_resolution = voxel_resolution
        self.dct_resolution = dct_resolution

        self.device_name = 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device_name = '0'
            self.yolov9 = YOLO("yolov9c-seg.pt").to('cuda')
        else:
            self.device_name = 'cpu'
            self.yolov9 = YOLO("yolov9c-seg.pt").to('cpu')
    
    # yaw in radians
    def predict(self, cv_image, yaw = 0, class_name = 'car'):
        if math.floor(cv_image.shape[0]) == 0 or math.floor(cv_image.shape[1]) == 0: return None
        segmented = self.yolov9.predict(cv_image)[0]

        maximum_idx = None
        maximum_area = None
        for i, box in enumerate(segmented.boxes):
            if maximum_idx == None or box.xywh[0][2] * box.xywh[0][3] > maximum_area:
                maximum_idx = i
                maximum_area = box.xywh[0][2] * box.xywh[0][3]

        if maximum_idx is None: return None
        maximum_mask = segmented.masks[maximum_idx].data[0].cpu().numpy().astype(np.uint8) * 255
        maximum_polygon = segmented.masks[maximum_idx]

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        resized_img = make_square(maximum_mask)
        cv2.imwrite('reconstruction3_1.png', resized_img)

        r = pyrender.OffscreenRenderer(256, 256)
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 64],
            [0, 0, 0, 1]
        ])
        light_pose = np.array([
            [1, 0, 0, 62],
            [0, 1, 0, 62],
            [0, 0, 1, 62],
            [0, 0, 0, 1]
        ])

        
        latent_variables = []
        energies = []
        for x0 in self.mygplvm.X:
            for dx1 in range(-1, 2):
                for dx2 in range(-1, 2):
                    x = [x0[0] + dx1, x0[1] + dx2]
                    mc_mesh = self.mo3.reconstruct(self.mygplvm, x, self.dct_resolution, self.voxel_resolution)
                    mc_mesh.apply_transform([
                        [cos(yaw)  , 0, sin(yaw)  , 0],
                        [0                      , 1, 0                      , 0],
                        [-sin(yaw) , 0, cos(yaw)  , 0],
                        [0                      , 0, 0                      , 1]
                    ])
                    bbox = np.max(np.abs(np.array(trimesh.bounds.corners(mc_mesh.bounding_box.bounds))), axis=0)
                    bbox_bound = max(bbox[0], bbox[1])

                    pl = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=50000)
                    pc = pyrender.OrthographicCamera(bbox_bound, bbox_bound)

                    scene = pyrender.Scene(ambient_light=[0.6, 0.6, 0.6], bg_color=[1.0, 1.0, 1.0])
                    mesh_pyrender = pyrender.Mesh.from_trimesh(mc_mesh, smooth=False)
                    #pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1280/720)
                    scene.add(mesh_pyrender, pose=np.eye(4))
                    scene.add(pl, pose=light_pose)
                    scene.add(pc, pose=camera_pose)
                    color, _ = r.render(scene)
                    cv2.imwrite('reconstruction3.png', color)
                    
                    rendered_occupancy = np.any(color != 255, axis=2) * 255
                    
                    energy = np.sum(resized_img & rendered_occupancy) / np.sum(resized_img | rendered_occupancy)
                    energies.append(energy)
                    latent_variables.append(x)
                    # energies.append({ 'x': x, 'energy': energy})

        r.delete()
        latent_variable = latent_variables[energies.index(max(energies))]
        return self.mo3.reconstruct(self.mygplvm, latent_variable, dct_resolution=self.dct_resolution, voxel_resolution=self.voxel_resolution)



def make_square(im, desired_size = 256):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    resized = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return resized
