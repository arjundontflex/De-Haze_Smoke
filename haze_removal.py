import PIL.Image as Image
import skimage.io as io
import numpy as np
import time
from gf import guided_filter
from numba import jit
import tkinter as tk
from tkinter import filedialog
import cv2

class HazeRemoval:
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        pass

    def open_image(self, img_path):
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double) / 255.
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)

    @jit
    def get_dark_channel(self, radius=7):
        print("Computing dark channel prior...")
        start = time.time()
        tmp = self.src.min(axis=2)
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - radius)
                rmax = min(i + radius, self.rows - 1)
                cmin = max(0, j - radius)
                cmax = min(j + radius, self.cols - 1)
                self.dark[i, j] = tmp[rmin:rmax + 1, cmin:cmax + 1].min()
        print("Time:", time.time() - start)

    def get_air_light(self):
        print("Computing air light prior...")
        start = time.time()
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows * self.cols * 0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark >= threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:, :].mean(axis=0)
        print("Time:", time.time() - start)

    @jit
    def get_transmission(self, radius=7, omega=0.95):
        print("Computing transmission...")
        start = time.time()
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - radius)
                rmax = min(i + radius, self.rows - 1)
                cmin = max(0, j - radius)
                cmax = min(j + radius, self.cols - 1)
                pixel = (self.src[rmin:rmax + 1, cmin:cmax + 1] / self.Alight).min()
                self.tran[i, j] = 1. - omega * pixel
        print("Time:", time.time() - start)

    def guided_filter(self, r=60, eps=0.001):
        print("Computing guided filter transmission...")
        start = time.time()
        self.gtran = guided_filter(self.src, self.tran, r, eps)
        print("Time:", time.time() - start)

    def recover(self, t0=0.1):
        print("Recovering...")
        start = time.time()
        self.gtran[self.gtran < t0] = t0
        t = self.gtran.reshape(*self.gtran.shape, 1).repeat(3, axis=2)
        self.dst = (self.src.astype(np.double) - self.Alight) / t + self.Alight
        self.dst = np.clip(self.dst * 255, 0, 255).astype(np.uint8)
        print("Time:", time.time() - start)

    def save_results(self):
        cv2.imwrite("img/src.jpg", (self.src * 255).astype(np.uint8)[:, :, (2, 1, 0)])
        cv2.imwrite("img/dark.jpg", (self.dark * 255).astype(np.uint8))
        cv2.imwrite("img/tran.jpg", (self.tran * 255).astype(np.uint8))
        cv2.imwrite("img/gtran.jpg", (self.gtran * 255).astype(np.uint8))
        cv2.imwrite("img/dst.jpg", self.dst[:, :, (2, 1, 0)])
        io.imsave("test3.jpg", self.dst)

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        hr = HazeRemoval()
        hr.open_image(file_path)
        hr.get_dark_channel()
        hr.get_air_light()
        hr.get_transmission()
        hr.guided_filter()
        hr.recover()
        hr.save_results()
    else:
        print("No image selected.")
