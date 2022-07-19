# Read images and json annotations from data directory
# and save them to a new directory
import os
import json
import cv2
import numpy as np
import pandas as pd

def read_images(data_dir, name):
    """
    Read images from data directory
    :param data_dir:
    :return:
    """
    images = []
    for file in os.listdir(data_dir):
        if file.endswith(".png"):
            images.append((file,cv2.imread(os.path.join(data_dir, file))))
    return images[images[0] == name]


# Read annotations
def read_annotations(data_dir):
    return pd.read_csv(os.path.join(data_dir, "instances_classes.csv"))


# Get images
images = read_images("data/task_road_sign_classes_backup_2022_06_23_09_49_27/data/", '4.png')


image_idx = 0 # Image index
name = 0     # Specify image name
image = 1   # Specify image

# Get annotations for each image
annotations = read_annotations("data/annotations/")

# Get annotation for specific image on index specified
image_data = annotations[annotations["file_name"] == images[image_idx][name]]

# Get bounding box
x1,y1,x2,y2 = image_data['x1'],image_data['y1'],image_data['x2'],image_data['y2']
pointA = (int(x1),int(y1))
pointB = (int(x2),int(y2))

# Get image by index
img = images[image_idx][image]
# Draw bounding box
cv2.rectangle(img, pointA, pointB, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

# Label bounding box
#cv2.putText(image, labels[image_idx][name], (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

# Normalize images
rows=300
cols=300
resized_img = cv2.resize(img, (rows, cols))
cv2.normalize(img, resized_img, 0, 255, cv2.NORM_MINMAX)

# Show normalized image
cv2.imshow('dst_rt', normalizedImg)

# Show image with bounds
cv2.imshow('Image Line', img)
cv2.waitKey(0)



