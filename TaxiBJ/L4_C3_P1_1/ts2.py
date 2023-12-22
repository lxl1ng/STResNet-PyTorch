import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from PIL import Image

response_2d1 = np.loadtxt('./save/0data.txt')
response_2d2 = np.loadtxt('./save/1data.txt')
img = (response_2d1-response_2d2).astype(np.uint8)
im = Image.fromarray(img)
# if im.mode == "F":
#     image = im.convert('RGB')
im.save("your_file.jpeg")
