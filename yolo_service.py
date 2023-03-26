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

tag   = "tags.pt"
price = "prices_4.pt"



class YoloService:

        def __init__(self, image_path=None):
            self.conf =  None
            if image_path != None:
                self.image_pil = Image.open(image_path).convert("RGB")
                self.image_cv  = np.array(self.image_pil)
            #self.tags_model  = None
            #self.prices_model = None

        def preprocessing(self):
            nimg = np.array(self.image_pil)
            ocvim = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(ocvim, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.image_cv = image

        def get_result(self):
            pass

        def load_models(self, models={"tag":"tags.pt", "price":"prices_4.pt"}):
            for name, model_path in models.items():
                    setattr(self, name, YOLO(model_path))


        def get_ocr_engine_options(self):
            """map scan result with document text"""
            pass

        def names_mapping(self):
            pass

        def set_ocr_engine_options(self, options=None):
            pass

        def get_ocr_engine_center(self):
            pass


        def parse_tags_info(self, results=None, image=None):
                names = {0:"pad", 2:"ean", 1:"price"}
                im_dict = {"pad":None, "ean":None, "price":None}
                for res in results:
                        boxes = res.boxes
                        for box in boxes:
                            label = names[int(box.boxes.tolist()[0][-1])]
                            im_dict[label] = tuple(box.xyxy.tolist()[0])
                return im_dict


        def parse_prices_info(self, results=None, image=None):
              names = {0:"box1", 2:"euros", 1:"box2"}
              im_dict = {"box1":None, "box2":None, "euros":None}
              for res in results:
                boxes = res.boxes
                for box in boxes:
                      label = names[int(box.boxes.tolist()[0][-1])]
                      im_dict[label] = tuple(box.xyxy.tolist()[0])
              return im_dict


        def load_image(self, image_pil=None):
                self.image_pil = image_pil
                self.image_cv = np.array(self.image_pil)


        def scan(self, modele=None):
            if modele == "tag":
                print("TAG")
                result = self.tag.predict(source=self.image_pil)
                return result
            else:
                print("PRICE")
                result = self.price.predict(source=self.image_pil)
                return result
