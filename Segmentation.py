from ultralytics import YOLO
import cv2
import numpy as np
import torch


class SegmentationModel:
    def __init__(self,weights_path='SegWeights.pt',device='cpu'):
        self.model=YOLO(weights_path)
        self.classes=self.model.names
        self.device=device
    def predict(self,img_path,plot=False):
        im = cv2.imread(img_path)
        im = cv2.resize(im, (640, 640))
        res = self.model(im, retina_masks=True,conf=0.5,device=self.device,verbose=False)
        orig_img=[]
        processed_masks=[]
        masks=[]
        classes=[]
        masked_img=[]
        try:
            masks = res[0].masks.numpy()
            masks = masks.data.astype(bool)
            classes = res[0].boxes.cls
            orig_img = res[0].orig_img
            if plot:
                masked_img=res[0].plot()
        except:
            pass
        for m in masks:
            croped=np.zeros_like(orig_img, dtype=np.uint8)
            croped[m] = orig_img[m]
            im = torch.tensor(croped)
            im = im / 255
            im = im.permute(2, 0, 1)
            processed_masks.append(im)
        classes_names=[self.classes[int(i)] for i in classes]
        return processed_masks,classes_names,masked_img