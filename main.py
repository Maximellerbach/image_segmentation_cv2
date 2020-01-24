from functions import *
from glob import glob

cap = create_cap(0) # "C:\\Users\\maxim\\video-fh4\\MFAwpTjlSY_Trim.mp4" # "C:\\Users\\maxim\\video-fh4\\FtcBrYpjnA_Trim.mp4"
# paths = glob("C:\\Users\\maxim\\image_course\\*")
resize_fact= (1,1)

cv2.namedWindow('img2')
cv2.setMouseCallback('img2', onclick_cnt)

while(1):
    img, gray, blurred, v = capture_img(cap, resize_fact)
# for path in paths:
#     img, gray, blurred, v = read_img(path, resize_fact)

    cnts, hierarchy = detect_edges(blurred, v)
    mask, img2 = get_masks(cnts, hierarchy[0], img)
    get_masked(cnts, mask, img)


    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    cv2.imshow('blurred', blurred)
    cv2.imshow('mask', mask/255)


    cv2.waitKey(1)