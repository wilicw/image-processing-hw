import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageProcess():

    def __init__(self):
        self.image = None

    def assert_image(self):
        assert self.image is not None, "Please load image first!"

    def add_image(self, image):
        print(image)
        self.image = cv2.imread(image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def color_separation(self):
        self.assert_image()

        r, g, b = cv2.split(self.image)
        rimg = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)])
        gimg = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
        bimg = cv2.merge([np.zeros_like(b), np.zeros_like(b), b])

        plt.subplot(1, 3, 3)
        plt.imshow(bimg)
        plt.title("Blue")

        plt.subplot(1, 3, 2)
        plt.imshow(gimg)
        plt.title("Green")
        
        plt.subplot(1, 3, 1)
        plt.imshow(rimg)
        plt.title("Red")
        
        plt.show()




