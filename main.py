from functions import *
from glob import glob

cap = create_cap(0) # "F:\\video-fh4\\MFAwpTjlSY_Trim.mp4" FtcBrYpjnA_Trim
resize_fact= (1, 1)

while(1):
    img, gray, blurred, v = capture_img(cap, resize_fact)

    im2, cnts, hierarchy = detect_edges(blurred, v)
    h, w, ch = img.shape

    masked, img = get_mask(cnts, hierarchy[0], img)

    cv2.imshow('img', img)
    cv2.imshow('gray', im2)
    cv2.imshow('blurred', blurred)
    cv2.imshow('masked', masked)

    cv2.waitKey(1)