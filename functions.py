import numpy as np
import cv2
import time

edges = []
queue = []

def create_cap(c = 0):
    cap = cv2.VideoCapture(c)
    return cap

def capture_img(cap, size):
    _, img = cap.read()
    h, w, ch = img.shape
    img = cv2.resize(img, (int(w*size[0]), int(h*size[1])))
    v = np.median(img)
    img = close_img(img, 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(img, 6, 35, 35)

    return img, gray, blurred, v

def read_img(path, size):
    img = cv2.imread(path)
    h, w, ch = img.shape
    img = cv2.resize(img, (int(w*size[0]), int(h*size[1])))
    v = np.median(img)
    img = close_img(img, 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(img, 6, 20, 20)
    
    return img, gray, blurred, v

def auto_canny(image, v, sigma=0.33):
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged
    
def detect_edges(img, v):
    edge = auto_canny(img, v, sigma=0.6)
    # edge = average_edges(edge, n=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.dilate(edge, kernel, iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

    return cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def get_masks(cnts, hierarchy, img):
    h, w, ch = img.shape
    img2 = np.zeros_like(img)
    mask = np.zeros((h, w))
    max_area = w*h
    # its = len(cnts)
    for it, cnt in enumerate(cnts):
        hcnt = hierarchy[it][-1]
        if -1<hcnt<=0:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            b = (area/max_area)*255
            g = (cx/w)*255
            r = (cy/h)*255

            # b, g, r = np.random.randint(0, 127, 3, dtype=np.uint8)

            cv2.drawContours(mask, [cnt], 0, it, -1)
            cv2.drawContours(img2, [cnt], 0, (int(b), int(g), int(r)), -1)

    return mask, img2

def get_masked(cnts, mask, img):
    h, w, ch = img.shape
    img2 = np.zeros_like(img)
    while(len(queue)>0):
        poly = np.zeros((h, w), np.uint8)
        region = np.zeros((h, w), np.uint8)
        x, y = queue[0]
        cnt_index = mask[y][x]
        cnt = cnts[int(cnt_index)]

        cv2.polylines(poly, [cnt], True, 255, thickness=3)
        outs, _ = cv2.findContours(poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = outs[0]

        # hull = cv2.convexHull(cnt)
        # cv2.fillPoly(region, [cnt], 255)

        cv2.drawContours(region, [out], 0, 255, -1)
        img2[region==255] = img[region==255]
        cv2.imwrite("regions\\"+str(time.time())+".png", img2)
        del queue[0]

def onclick_cnt(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        queue.append([x,y])

def get_cnt_child(cnts, cnt_index):
    return

def close_img(img, c=0):
    img[:, :3] = c
    img[:, -4:] = c
    img[:3, :] = c
    img[-4:, :] = c

    return img

def average_edges(edge, n=5):
    if len(edges)<n:
        edges.append(edge)
    else:
        edges.append(edge)
        del edges[0]

    k = len(edges)
    weights = [(i/k)**2 for i in range(1, k+1)]

    av = np.array(np.average(edges, axis=0, weights=weights), dtype=np.uint8)
    return av

if __name__ == "__main__":
    import main