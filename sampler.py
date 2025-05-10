import os
import random


def sample(source, count, target):
    all_images = []
    for cat in os.listdir(source):
        for img in os.listdir(os.path.join(source, cat)):
            image_path = os.path.join(source, cat, img)
            all_images.append((image_path, cat))
    
    randomImages = random.sample(all_images, count)

    with open(target, 'w') as tar:
        for rimg, label in randomImages:
            tar.write(rimg + " " + label+ "\n")

