from paddle_service import PaddleService
from trocr_service import TrOcrService
from yolo_service import YoloService
from PIL import Image
from gtin import has_valid_check_digit

def main():
    response = {"gtin":None, "prix":None}

    paddle_process = PaddleService(image_path="img.jpg")
    result = paddle_process.scan()

    for item in result:
        for i in item:
                if has_valid_check_digit(i[-1][0]):
                    response["gtin"] = i[-1][0]

    print(response)

    yolo_process = YoloService()
    yolo_process.load_image(image_pil=paddle_process.image_pil)
    yolo_process.load_models()
    yolo_process.preprocessing()
    result_tag = yolo_process.scan(modele="tag")
    print("tag", result_tag)

    block_info = yolo_process.parse_tags_info(results=result_tag)
    price_im = yolo_process.image_pil.crop(block_info["price"])
    yolo_process.load_image(image_pil=price_im)
    result = yolo_process.scan(modele="price")
    print("price", result)
    price_info = yolo_process.parse_prices_info(results=result)
    box1 = price_im.crop(price_info["box1"])
    box2 = price_im.crop(price_info["box2"])

    trocr_process_box1 = TrOcrService()
    trocr_process_box2 = TrOcrService()

    trocr_process_box1.load_image(box1)
    trocr_process_box2.load_image(box2)

    text_1 = trocr_process_box1.scan()
    text_2 = trocr_process_box2.scan()
    print("trocr", text_1, text_2)





if __name__ == '__main__':
    main()
