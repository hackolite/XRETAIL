from paddleocr import PaddleOCR,draw_ocr
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import torch
import os
import numpy as np
import torchvision
import torch
import cv2


class PaddleService:

        def __init__(self, image_path=None):
            self.conf =  None
            if image_path != None:
                self.image_pil = Image.open(image_path).convert("RGB")
                self.image_cv  = np.array(self.image_pil)

        def preprocessing(self, image=None):
            nimg = np.array(image)
            ocvim = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(ocvim, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.image = image

        def get_result(self):
            pass

        def load_models(self, models={}):
            pass

        def get_ocr_engine_options(self):
            """map scan result with document text"""
            pass

        def names_mapping(self):
            pass

        def set_ocr_engine_options(self, options=None):
            pass

        def get_ocr_engine_center(self):
            pass


        def load_image(self, image=None):
            self.image_pil = image
            self.image_cv  = np.array(self.image_pil)


        def scan(self):
            #if image == None:
            #    image= self.image_cv
            ocr = PaddleOCR(lang='en', use_angle_cls=True) # need to run only once to download and load model into memory
            result = ocr.ocr( self.image_cv, cls=True)
            self.result = result
            return self.result
