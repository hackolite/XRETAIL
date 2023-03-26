from paddle_service import PaddleService
from trocr_service import TrOcrService
from yolo_service import YoloService
from PIL import Image
from gtin import has_valid_check_digit

class TagOcr:

    def __init__(self):
        pass
        self.paddle_process = None

    def load_ocr(self, ocr_list=["paddle", "trocr", "yolo"]):
        """tools for detection and recognition of text blocks and text blocks"""
        self.paddle_process     = PaddleService()
        self.trocr_process_box1 = TrOcrService()
        self.trocr_process_box2 = TrOcrService()
        self.yolo_process       = YoloService()
        self.yolo_process.load_models()

    def execute(self, image):
        self.image_pil = image
        response = {"gtin":None, "prix":None}
        #paddle
        self.paddle_process.load_image(image=self.image_pil)
        result = self.paddle_process.scan()

        for item in result:
            for i in item:
                    if has_valid_check_digit(i[-1][0]):
                        response["gtin"] = i[-1][0]

        print(response)

        self.yolo_process.load_image(image_pil=self.paddle_process.image_pil)

        self.yolo_process.preprocessing()
        result_tag = self.yolo_process.scan(modele="tag")
        print("tag", result_tag)
        block_info = self.yolo_process.parse_tags_info(results=result_tag)
        price_im = self.yolo_process.image_pil.crop(block_info["price"])
        self.yolo_process.load_image(image_pil=price_im)
        result = self.yolo_process.scan(modele="price")
        print("price", result)
        price_info = self.yolo_process.parse_prices_info(results=result)
        box1 = price_im.crop(price_info["box1"])
        box2 = price_im.crop(price_info["box2"])


        self.trocr_process_box1.load_image(box1)
        self.trocr_process_box2.load_image(box2)


        text_1 = self.trocr_process_box1.scan()
        text_2 = self.trocr_process_box2.scan()
        print("trocr", text_1, text_2)


if __name__ == '__main__':
    ocr = TagOcr()
    ocr.load_ocr()
    img1 = Image.open("./img.jpg").convert("RGB")
    ocr.execute(image=img1)
