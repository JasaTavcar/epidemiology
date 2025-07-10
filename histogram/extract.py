import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

NUM_COLUMNS = 161
UNIT_SIZE = 100

# Load and preprocess
image_path = './hist.png'
img = cv2.imread(image_path)

#cut bottom two rows
#img = img[:-2, :, :]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scale_img = cv2.imread("scale.png")
unit = scale_img.shape[0]

column_width = gray.shape[1] / NUM_COLUMNS

M = []
F = []
for i in range(NUM_COLUMNS):
    # Get center position of the column
    center_x = int(i * column_width + column_width // 2)

    # Extract 1-pixel-wide vertical slice
    vertical_slice = gray[:, center_x]

    # Compute vertical gradient (difference between adjacent pixels)
    gradient = np.zeros_like(vertical_slice)
    for j in range(1, len(vertical_slice)):
        gradient[j] = abs(vertical_slice[j] - vertical_slice[j - 1])
    m = len(gradient)
    f = len(gradient)
    for i in range(len(gradient)):
        if gradient[i] > 50: 
            m = i
            while i < len(gradient) and gradient[i] > 0:
                i += 1
            while i < len(gradient) and gradient[i] < 100:
                i += 1
            f = i
            break

    # convert pixel value to real number on scale
    m = len(gradient) - m
    f = len(gradient) - f
    m = int(m * UNIT_SIZE / unit)
    f = int(f * UNIT_SIZE / unit)
    M.append(m)
    F.append(f)

# save to csv
df = pd.DataFrame({'M': M, 'F': F})
df.to_csv('hist.csv')