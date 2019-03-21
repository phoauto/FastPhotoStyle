"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import process_stylization
from photo_wct import PhotoWCT
from photo_gif import GIFSmoothing

class PhotoStyler():
    def __init__(self):
        self.p_wct = PhotoWCT()
        self.p_wct.load_state_dict(torch.load('./PhotoWCTModels/photo_wct.pth'))
        # self.p_wct.cuda(0)
        self.p_pro = GIFSmoothing(r=35, eps=0.001)
    
    def stylize(self, cont_img, styl_img, cont_seg, styl_seg):
        return process_stylization.stylization(
            stylization_module=self.p_wct,
            smoothing_module=self.p_pro,
            content_image_path='./images/content1.png',
            style_image_path='./images/style1.png',
            content_seg_path=[],
            style_seg_path=[],
            output_image_path='/tmp/example.png',
            cuda=0,
            save_intermediate=False,
            no_post=False
        )