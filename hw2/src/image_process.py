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

    def draw_conutor(self):
        self.assert_image()
        image = self.image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=50,
        )
        circles = np.uint16(np.around(circles))
        blank = np.zeros((image.shape[0], image.shape[1], image.shape[2]), np.uint8)
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(blank, (i[0], i[1]), 2, (255, 255, 255), 3)
        cv2.imshow("detected circles", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("detected circles", blank)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def count_coin(self):
        self.assert_image()
        image = self.image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=50,
        )
        return len(circles[0, :])

    def histogram_equalization(self):
        self.assert_image()
        image = self.image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(image)

        # manual histogram equalization
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")
        equalized_image_manual = cdf[image]

        plt.subplot(2, 3, 1)
        plt.title("Original")
        plt.imshow(image, cmap="gray")
        plt.subplot(2, 3, 4)
        plt.hist(image.ravel(), 256, [0, 256])
        plt.subplot(2, 3, 2)
        plt.title("Equalized")
        plt.imshow(equalized_image, cmap="gray")
        plt.subplot(2, 3, 5)
        plt.hist(equalized_image.ravel(), 256, [0, 256])
        plt.subplot(2, 3, 3)
        plt.title("Equalized (manual)")
        plt.imshow(equalized_image_manual, cmap="gray")
        plt.subplot(2, 3, 6)
        plt.hist(equalized_image_manual.ravel(), 256, [0, 256])
        plt.show()

    def opening(self):
        self.assert_image()
        image = self.image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.erode(image, kernel, iterations=1)
        result = cv2.dilate(result, kernel, iterations=1)
        plt.imshow(result, cmap="gray")
        plt.show()

    def closing(self):
        self.assert_image()
        image = self.image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.dilate(image, kernel, iterations=1)
        result = cv2.erode(result, kernel, iterations=1)
        plt.imshow(result, cmap="gray")
        plt.show()
