from torchvision import transforms
from PIL import Image

import numpy as np

class BaselineTransforms():

    def __init__(self):
        self.T_x = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96), interpolation=Image.CUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(x.shape).expand(3, -1, -1))
        ])

        self.T_y = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96), interpolation=Image.NEAREST),# "non-nearest" interpolation breaks mask --> one-hot-encode
            transforms.ToTensor(),
            transforms.Lambda(lambda x: mask2label(x))
        ])        

    @staticmethod
    def mask2label(mask, data_set_name="ADPKD", show_vals = False):
        """converts mask png to one-hot-encoded label"""    

        if show_vals:
            print("unique values ", np.unique(mask)) 
    
        #unique_vals corespond to mask class values after transforms        
        if(data_set_name == "ADPKD"):
            L_KIDNEY = 0.5019608
            R_KIDNEY = 0.7490196
            unique_vals = [R_KIDNEY, L_KIDNEY]

        mask = mask.squeeze()
        
        s = mask.shape

        ones, zeros = np.ones(s), np.zeros(s)

        one_hot_map = [np.where(mask == unique_vals[targ], ones, zeros)
                    for targ in range(len(unique_vals))]

        one_hot_map = np.stack(one_hot_map, axis=0).astype(np.uint8)

        return one_hot_map