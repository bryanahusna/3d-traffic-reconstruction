from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys

# stop python from writing so much bytecode
# sys.path.append(os.getcwd())
sys.path.append('./m3d_rpn')
# sys.dont_write_bytecode = True
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

conf_path = "/mnt/c/Bryan/TA/weights/m3d_rpn/m3d_rpn_depth_aware_test_config.pkl"
weights_path = "/mnt/c/Bryan/TA/weights/m3d_rpn/m3d_rpn_depth_aware_test"


class M3DRPN:
    def __init__(self) -> None:
        # load config
        self.conf = edict(pickle_read(conf_path))
        self.conf.pretrained = None

        # -----------------------------------------
        # torch defaults
        # -----------------------------------------
        # defaults
        init_torch(self.conf.rng_seed, self.conf.cuda_seed)

        # -----------------------------------------
        # setup network
        # -----------------------------------------
        # net, load weights, and switch modes for evaluation
        self.net = import_module('models.' + self.conf.model).build(self.conf)
        load_weights(self.net, weights_path, remove_module=True)
        self.net.eval()

        # print(pretty_print('conf', self.conf))

    def predict(self, cv_image):
        # -----------------------------------------
        # test kitti
        # -----------------------------------------
        
        # import read_kitti_cal
        from lib.imdb_util import read_kitti_cal

        # imlist = list_files(os.path.join(test_path, dataset_test, 'validation', 'image_2', ''), '*.png')
        preprocess = Preprocess([self.conf.test_scale], self.conf.image_means, self.conf.image_stds)

        # init
        test_start = time()
        im = cv_image

        # read in calib
        p2 = read_kitti_cal("./m3d_rpn/calib.txt")
        p2_inv = np.linalg.inv(p2)

        # forward test batch
        aboxes = im_detect_3d(im, self.net, self.conf, preprocess, p2)
        results = []
        for boxind in range(0, min(self.conf.nms_topN_post, aboxes.shape[0])):
            box = aboxes[boxind, :]
            score = box[4]
            cls = self.conf.lbls[int(box[5] - 1)]

            if score >= 0.75:
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                width = (x2 - x1 + 1)
                height = (y2 - y1 + 1)

                # plot 3D box
                x3d = box[6]
                y3d = box[7]
                z3d = box[8]
                w3d = box[9]
                h3d = box[10]
                l3d = box[11]
                ry3d = box[12]

                # convert alpha into ry3d
                coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

                step_r = 0.3*math.pi
                r_lim = 0.01
                box_2d = np.array([x1, y1, width, height])

                z3d, ry3d, verts_best = hill_climb(p2, p2_inv, box_2d, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, step_r_init=step_r, r_lim=r_lim)

                # predict a more accurate projection
                coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                alpha = convertRot2Alpha(ry3d, coord3d[2], coord3d[0])

                x3d = coord3d[0]
                y3d = coord3d[1]
                z3d = coord3d[2]

                y3d += h3d/2

                results.append([cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score])
        return results
