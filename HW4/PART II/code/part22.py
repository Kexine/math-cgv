import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageTk
import scipy
from tkinter import *

class ImageMarker:
    def __init__(self, image):
        self.image = image
        # self.canvas = canvas
        # self.draw = draw
        self.prevx = None
        self.prevy = None

    def activate_paint(self, canvas, event):
        global lastx, lasty
        canvas.bind('<B1-Motion>', self.paint)
        lastx, lasty = event.x, event.y

    def paint(self, draw, event, prevx, prevy):
        x, y = event.x, event.y
        self.canvas.create_line((prevx, prevy, x, y), width=3)
        draw.line((prevx, prevy, x, y), fill='black', width=3)
        prevx, prevy = x, y



if __name__ == "__main__":

    image = Image.open()
    marker = ImageMarker(image)