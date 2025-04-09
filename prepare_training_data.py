import os
import cv2
import numpy as np

input_folder = "BigDataSet"
output_images_file = "images.npy"
output_labels_file = "labels.npy"
image_size = 32  

images = []
labels = []


letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letter_to_label = {letter: idx for idx, letter in enumerate(letters)}


for letter in letters:
    letter_folder = os.path.join(input_folder, letter)
    if os.path.isdir(letter_folder):
        for filename in os.listdir(letter_folder):
            if filename.endswith(".png"):
                
                image = cv2.imread(os.path.join(letter_folder, filename), 0)

               
                resized_img = cv2.resize(image, (image_size, image_size))

                normalized_img = resized_img / 255.0

                
                images.append(normalized_img)
                labels.append(letter_to_label[letter])


images_array = np.array(images)
labels_array = np.array(labels).reshape(-1, 1)


np.save(output_images_file, images_array)
np.save(output_labels_file, labels_array)
