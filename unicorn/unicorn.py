import sys
sys.path.append('./unicorn')

import cv2
import numpy as np
import torchvision.transforms as transforms
from src.model import load_model_from_path
from src.utils.pytorch import get_torch_device
from src.utils.mesh import save_mesh_as_obj, normalize


class Unicorn:
    def __init__(self, model_name) -> None:
        self.device = get_torch_device()
        
        self.model_name = model_name
        self.model = load_model_from_path(self.model_name).to(self.device)
        self.model.eval()

    def predict_cv_image(self, cv_image):
        # out = path_mkdir(args.input + '_rec')
        # n_zeros = int(np.log10(len(data) - 1)) + 1
        transform = transforms.ToTensor()
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        imgs = transform(rgb_image).unsqueeze(0).to(self.device)

        # imgs = inp['imgs'].to(self.device)
        meshes, RT, bkgs = self.model.predict_mesh_pose_bkg(imgs)
        return meshes, RT, bkgs
