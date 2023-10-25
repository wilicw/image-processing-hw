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

    def bilateral_callback(self, x):
        self.assert_image()
        blur = cv2.bilateralFilter(self.image, 2 * x + 1, 75, 75)
        cv2.imshow("Bilateral Blur", blur)

    def bilateral_filter(self):
        self.assert_image()
        cv2.namedWindow("Bilateral Blur")
        cv2.imshow("Bilateral Blur", self.image)
        cv2.createTrackbar("Blur", "Bilateral Blur", 1, 5, self.bilateral_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def median_callback(self, x):
        self.assert_image()
        blur = cv2.medianBlur(self.image, 2 * x + 1)
        cv2.imshow("Median Blur", blur)

    def median_filter(self):
        self.assert_image()
        cv2.namedWindow("Median Blur")
        cv2.imshow("Median Blur", self.image)
        cv2.createTrackbar("Blur", "Median Blur", 1, 5, self.median_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def conv2D(self, image, kernel):
        kernel = np.flipud(np.fliplr(kernel))
        kernel_size = kernel.shape[0]
        image_height = image.shape[0]
        image_width = image.shape[1]
        padding_size = kernel_size // 2
        padding_image = np.zeros(
            (image_height + padding_size * 2, image_width + padding_size * 2)
        )
        padding_image[padding_size:-padding_size, padding_size:-padding_size] = image
        output_image = np.zeros(image.shape)
        for i in range(image_height):
            for j in range(image_width):
                output_image[i, j] = (
                    kernel * padding_image[i : i + kernel_size, j : j + kernel_size]
                ).sum()
        return output_image

    def sobelx(self):
        self.assert_image()
        filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sobelX = self.conv2D(gray, filter)
        sobelX = np.uint8(np.absolute(sobelX))
        cv2.imshow("Sobel X", sobelX)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sobely(self):
        self.assert_image()
        filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sobelY = self.conv2D(gray, filter)
        sobelY = np.uint8(np.absolute(sobelY))
        cv2.imshow("Sobel Y", sobelY)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sobel(self):
        self.assert_image()
        filterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        filterY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sobelX = self.conv2D(gray, filterX)
        sobelY = self.conv2D(gray, filterY)
        sobel = np.sqrt(sobelX ** 2 + sobelY ** 2)
        sobel = np.uint8(np.absolute(sobel))
        cv2.imshow("Sobel", sobel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

