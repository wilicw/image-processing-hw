import cv2

class cifar_vgg:

    def __init__(self):
        pass

    def add_image(self, image):
        self.image = cv2.imread(image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.show_image = cv2.resize(self.image, (128, 128))

    

