from functions import *

cap = create_cap(0)

while(1):
    img, gray, blurred = capture_img(cap)

    cnts = detect_edges(blurred)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.imshow('gray', gray)
    cv2.imshow('blurred', blurred)


    cv2.waitKey(1)