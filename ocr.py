#!/bin/bash/python3

import easyocr
import pyautogui
from PIL import Image

screenWidth, screenHeight = pyautogui.size()

im = pyautogui.screenshot('image.jpg', region=(1350, 375, 220, 700))
img = Image.open('image.jpg')

# Load the image
reader = easyocr.Reader(['en'])
text = reader.readtext(img)
for line in text:
    print(line[1])
