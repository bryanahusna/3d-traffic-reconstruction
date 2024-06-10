import cv2
import numpy as np
import os
import sys

sys.path.append('./deepbox')
from util.post_processing import gen_3D_box,draw_3D_box,draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

calib_file = 'D:/Bryan/TA/10 Vehicle Rotation/cersar_3d_detection/3D_detection-master/testing/calib.txt'
box2d_dir = 'D:/Bryan/TA/10 Vehicle Rotation/cersar_3d_detection/3D_detection-master/testing/label_2/'

classes = ['car','van','truck','pedestrian','person_sitting','cyclist','tram']
cls_to_ind = {cls:i for i,cls in enumerate(classes)}

dims_avg = np.loadtxt(r'./deepbox/dataset/voc_dims.txt',delimiter=',')


cam_to_img = get_cam_data(calib_file)
fx = cam_to_img[0][0]
u0 = cam_to_img[0][2]
v0 = cam_to_img[1][2]


class Deepbox:
    def __init__(self) -> None:
        # Construct the network
        self.model = bbox_3D_net((224,224,3))
        self.model.load_weights("D:/Bryan/TA/10 Vehicle Rotation/cersar_3d_detection/weights/weights.h5")

    ## detc_2ds: [class_name, [x1, y1, x2, y2]]
    def predict(self, cv_image, detc_2ds):
        img = cv_image

        yaws = []
        for data in detc_2ds:
            cls = data[0]
            box_2D = np.asarray(data[1],dtype=float)
            xmin = box_2D[0]
            ymin = box_2D[1]
            xmax = box_2D[2]
            ymax = box_2D[3]

            patch = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            patch = cv2.resize(patch, (224, 224))
            patch = patch - np.array([[[103.939, 116.779, 123.68]]])
            patch = np.expand_dims(patch, 0)

            prediction = self.model.predict(patch)

            # compute dims
            dims = dims_avg[cls_to_ind[cls]] + prediction[0][0]

            # Transform regressed angle
            box2d_center_x = (xmin + xmax) / 2.0
            # Transfer arctan() from (-pi/2,pi/2) to (0,pi)
            theta_ray = np.arctan(fx /(box2d_center_x - u0))
            if theta_ray<0:
                theta_ray = theta_ray+np.pi

            max_anc = np.argmax(prediction[2][0])
            anchors = prediction[1][0][max_anc]

            if anchors[1] > 0:
                angle_offset = np.arccos(anchors[0])
            else:
                angle_offset = -np.arccos(anchors[0])

            bin_num = prediction[2][0].shape[0]
            wedge = 2. * np.pi / bin_num
            theta_loc = angle_offset + max_anc * wedge

            theta = theta_loc + theta_ray
            # object's yaw angle
            yaw = np.pi/2 - theta
            yaws.append(yaw)

            # points2D = gen_3D_box(yaw, dims, cam_to_img, box_2D)
            # draw_3D_box(img, points2D)

        return yaws
