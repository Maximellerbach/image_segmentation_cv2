from glob import glob

import cv2
import numpy as np
from keras.models import load_model, save_model

from datagenerator import image_generator
import functions
import model


def test_model(model, paths, n=16, size=(256, 256)):
    for i in range(n):
        path = np.random.choice(paths)
        img = cv2.imread(path)/255
        pred = model.predict(np.expand_dims(img, axis=0))[0]

        cv2.imshow('pred', pred)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# fe = load_model("fe.h5")
# fe.summary()
dos = glob("C:\\Users\\maxim\\Desktop\\car_img\\*")
datalen = len(dos)

fe = model.create_segm_model()
fe.compile("adam", loss="mse", metrics=["accuracy"])

for i in range(10):
    
    fe.fit_generator(image_generator(dos, datalen, 32, augm=True), steps_per_epoch=datalen//32, epochs=1,
                    validation_data=image_generator(dos, datalen, 32, augm=True), validation_steps=datalen//20//32,
                    max_queue_size=4, workers=8)
        
