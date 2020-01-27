import numpy as np
import autolib
import keras
import cv2
from glob import glob

class image_generator(keras.utils.Sequence):
    def __init__(self, img_path, datalen, batch_size=32, augm=True, proportion=0.15, shape=(128,128,3)):
        self.shape = shape
        self.augm = augm
        self.img_cols = shape[0]
        self.img_rows = shape[1]
        self.batch_size = batch_size
        self.img_path = img_path
        self.proportion = proportion
        self.datalen = datalen

    def __data_generation(self, img_path):
        batchfiles = np.random.choice(img_path, size=self.batch_size)
        xbatch = []
        ybatch = []

        for i in batchfiles:
            try:
                img = cv2.imread(i)
                img = cv2.resize(img, (self.img_cols, self.img_rows))

                xbatch.append(img)
            except:
                print(i)

        xflip = np.array([cv2.flip(i, 1) for i in xbatch])
        xbatch = np.concatenate((xbatch, xflip))
        ybatch = xbatch


        if self.augm == True:
            X_bright, _ = autolib.generate_brightness(xbatch, ybatch, proportion=self.proportion)
            X_gamma, _ = autolib.generate_low_gamma(xbatch, ybatch, proportion=self.proportion)
            X_night, _ = autolib.generate_night_effect(xbatch, ybatch, proportion=self.proportion)
            X_shadow, _ = autolib.generate_random_shadows(xbatch, ybatch, proportion=self.proportion)
            X_chain, _ = autolib.generate_chained_transformations(xbatch, ybatch, proportion=self.proportion)
            X_noise, _ = autolib.generate_random_noise(xbatch, ybatch, proportion=self.proportion)
            X_rev, _ = autolib.generate_inversed_color(xbatch, ybatch, proportion=self.proportion)
            X_glow, _ = autolib.generate_random_glow(xbatch, ybatch, proportion=self.proportion)
            X_cut, _ = autolib.generate_random_cut(xbatch, ybatch, proportion=self.proportion)

            xbatch = np.concatenate((xbatch, X_gamma, X_bright, X_night, X_shadow, X_chain, X_noise, X_rev, X_glow, X_cut))/255
            ybatch = xbatch
        else:
            xbatch = xbatch/255
            ybatch = xbatch

        return xbatch, ybatch

    def __len__(self):
        return int(self.datalen/self.batch_size)

    def __getitem__(self, index):
        X, Y = self.__data_generation(self.img_path)
        return X, Y
