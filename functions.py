import numpy as np
import cv2
edges = []

def create_cap(c = 0):
    cap = cv2.VideoCapture(c)
    return cap

def capture_img(cap):
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(img, 9, 75,75)

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
    edge = average_edges(edge)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dilated = cv2.dilate(edge, kernel)
    _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts
    

def average_edges(edge):
    if len(edges)<5:
        edges.append(edge)
    else:
        edges.append(edge)
        del edges[0]

    av = np.array(np.average(edges, axis=0), dtype="uint8")
    return av