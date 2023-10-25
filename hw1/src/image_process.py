import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageProcess:
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

    def color_transformation(self):
        self.assert_image()
        cv_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        r, g, b = cv2.split(self.image)
        avg_gray = np.float32(r / 3 + g / 3 + b / 3)
        plt.subplot(1, 2, 1)
        plt.imshow(cv_gray, cmap="gray")
        plt.title("Weighted Gray")
        plt.subplot(1, 2, 2)
        plt.imshow(avg_gray, cmap="gray")
        plt.title("Average Gray")
        plt.show()

    def color_extration(self):
        self.assert_image()
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        lower = np.uint8([20, 0, 0])
        upper = np.uint8([85, 255, 255])
        mask = cv2.inRange(hsv_image, lower, upper)
        removed_image = cv2.bitwise_not(
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), self.image.copy(), mask
        )
        plt.subplot(1, 2, 1)
        plt.imshow(self.image)
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(removed_image)
        plt.title("Removed")
        plt.show()

    def gaussian_callback(self, x):
        self.assert_image()
        blur = cv2.GaussianBlur(self.image, (2 * x + 1, 2 * x + 1), 0)
        cv2.imshow("Gaussian Blur", blur)

    def gaussian_filter(self):
        self.assert_image()
        cv2.namedWindow("Gaussian Blur")
        cv2.imshow("Gaussian Blur", self.image)
        cv2.createTrackbar("Blur", "Gaussian Blur", 1, 5, self.gaussian_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

