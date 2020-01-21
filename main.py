from functions import *

cap = create_cap(0)

while(1):
    img, gray, blurred = capture_img(cap)

    _, cnts, hierarchy = detect_edges(blurred)
    h, w, ch = img.shape

    masked, img = get_mask(cnts, hierarchy[0], img)

    cv2.imshow('img', img)
    cv2.imshow('gray', gray)
    cv2.imshow('blurred', blurred)
    cv2.imshow('masked', masked)

    cv2.waitKey(1)