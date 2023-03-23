

class Paddle:

        def __init__(self):
            pass
            self.image = None

        def preprocessing(self, image=None):
            nimg = np.array(image)
            ocvim = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(ocvim, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.image = image

        def get_result(self):
            pass

        def load_models(self, modeles={}):
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

        def scan_image(self, image=None):
            pass
