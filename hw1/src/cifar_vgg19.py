import cv2, os
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from PIL import Image 

class cifar_vgg:
    def __init__(self):
        self.transforms = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.3),
                v2.RandomRotation(degrees=15),
                v2.RandomVerticalFlip(p=0.3),
            ]
        )

    def add_image(self, image):
        self.image = cv2.imread(image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.show_image = cv2.resize(self.image, (128, 128))

    def augmentation(self, image):
        return self.transforms(image)

    def show_augmented_images(self):
        images = os.listdir("Q5_image/Q5_1")
        i = 0
        for image in images:
            if ".png" not in image and ".jpg" not in image:
                continue
            img = Image.open("Q5_image/Q5_1/" + image)
            img = self.augmentation(img)
            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.title(image.split(".")[0])
            i += 1
        plt.show()
