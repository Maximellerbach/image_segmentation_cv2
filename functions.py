import numpy as np
import cv2
edges = []

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
    blurred = cv2.bilateralFilter(img, 5, 70, 70)

    return img, gray, blurred, v

def auto_canny(image, v, sigma=0.33):
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def detect_edges(img, v):
    edge = auto_canny(img, v, sigma=0.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3,3))
    dilated = cv2.dilate(edge, kernel)
    return cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def get_mask(cnts, hierarchy, img):
    mask = np.zeros(img.shape, np.uint8)
    for it, cnt in enumerate(cnts):
        hcnt = hierarchy[it][-1]
        if -1<hcnt: #  and cv2.contourArea(cnt)>150
            b, g, r = np.random.randint(0, 127, 3, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, (int(b), int(g), int(r)), -1)
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 1)

    return mask, img

def close_img(img, c=0):
    img[:, :3] = c
    img[:, -4:] = c
    img[:3, :] = c
    img[-4:, :] = c

    return img

def average_edges(edge):
    if len(edges)<5:
        edges.append(edge)
    else:
        edges.append(edge)
        del edges[0]

    av = np.array(np.average(edges, axis=0), dtype=np.uint8)
    return av

if __name__ == "__main__":
    import main