import os
import glob
import numpy as np
import cv2
import moviepy.editor as mpy
from natsort import natsorted
from tqdm import tqdm

def vread(path, T=100):
    cap = cv2.VideoCapture(path)
    gif = [cap.read()[1][:,:,::-1] for i in range(T)]
    gif = np.array(gif)
    cap.release()
    return gif

def resize(path_from, path_to):

    D0 = natsorted(glob.glob(os.path.join(path_from, "*")))

    for d0 in tqdm(D0): # Normal, Freeze, Random...
        D1 = natsorted(glob.glob(d0+"/*"))
        d0 = d0.split("/")[-1]
        os.mkdir(os.path.join(path_to, d0))

        for d1 in D1: # 0~90
            gif = vread(d1)
            gif = [cv2.resize(frame, (64,64)) for frame in gif]
            clip = mpy.ImageSequenceClip(gif, fps=20)
            d1 = d1.split("/")[-1]
            clip.write_gif(os.path.join(path_to, d0, d1), fps=20)

if __name__ == '__main__':
    PATH = "./demos"
    os.mkdir("./demos_mini")

    path_from = "./demos"
    path_to = "./demos_mini"
    resize(path_from, path_to)

