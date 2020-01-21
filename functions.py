import numpy as np
import cv2
edges = []

def create_cap(c = 0):
    cap = cv2.VideoCapture(c)
    return cap

def capture_img(cap):
    _, img = cap.read()
    img = close_img(img, 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(img, 9, 35, 35)

    return img, gray, blurred

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def detect_edges(img):
    edge = auto_canny(img)
    # edge = average_edges(edge)
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (5,5))
    dilated = cv2.dilate(edge, kernel)
    return cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def get_mask(cnts, hierarchy, img):
    mask = np.zeros(img.shape, np.uint8)
    for it, cnt in enumerate(cnts):
        hcnt = hierarchy[it][-1]
        if hcnt != -1 and hcnt<=1:
            b, g, r = np.random.randint(0, 255, 3, dtype=np.uint8)
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