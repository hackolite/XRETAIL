from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
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

model_tag_path   = "./tags.pt"
model_price_path = "./prices_4.pt"



class TrOcrService:

        def __init__(self, image_path=None):
            self.conf =  None
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

            if image_path != None:
                self.image_pil = Image.open(image_path).convert("RGB")
                self.image_cv  = np.array(self.image_pil)


        def preprocessing(self):
            nimg = np.array(self.image_pil)
            ocvim = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(ocvim, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.image_cv = image

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


        def load_image(self,image=None):
            self.image_pil = image
            self.image_cv = np.array(self.image_pil)

        def scan(self):
            pixel_values = self.processor(images=self.image_cv, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
