import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from skimage.feature import hog


class Imgs:
    def __init__(self, db="Jaffe"):
        self.db = db

        self.folders = np.array(["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"])
        self.f_count = {
            'Angry': 0,
            'Disgust': 0,
            'Fear': 0,
            'Happy': 0,
            'Neutral': 0,
            'Sad': 0,
            'Surprise': 0,
            'Total': 0
        }

        self.test_fe = []
        self.test_lb = []
        self.train_fe = []
        self.train_lb = []

        self.populate()

    def populate(self):
        dir_path = os.path.join(os.curdir, os.path.join("Assets", self.db.lower()))

        fe = []
        lb = []

        for i in range(self.folders.shape[0]):
            img_path = os.path.join(dir_path, self.folders[i])
            files = os.listdir(img_path)
            for f in files:
                self.f_count[self.folders[i]] += 1
                self.f_count["Total"] += 1

                img = cv2.imread(os.path.join(img_path, f), 0)
                img = cv2.resize(img, (120, 120))   # Will be replaced with a cropping function

                fe.append(img)
                lb.append([i])

        fe = np.array(fe)
        lb = np.array(lb)

        self.split(fe, lb)
        self.print_info()

    def print_info(self):
        print("Read in", self.f_count["Total"], "images from the", self.db, "database.")
        for f in self.folders:
            print("\t", self.f_count[f], "\t", f)
        print("\t---------------------")
        print("\t", self.train_lb.shape[0], "\t Training Examples")
        print("\t", self.test_lb.shape[0], "\t Testing Examples")

    def split(self, fe, lb):
        self.train_fe, self.test_fe, self.train_lb, self.test_lb = train_test_split(fe, lb, test_size=.25,
                                                                                    random_state=13)


def run_hog(imgs):
    n_imgs = []
    for img in imgs:
        des, vis = hog(img, orientations=6, pixels_per_cell=(6, 6),
                       cells_per_block=(5, 5), visualise=True, transform_sqrt=False)
        n_imgs.append(des)
    n_imgs = np.array(n_imgs)
    return n_imgs