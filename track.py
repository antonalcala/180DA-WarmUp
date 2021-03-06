import numpy as np
import cv2

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# combination of code from:
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html (masking)
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097 (K-Means)
# https://www.analyticsvidhya.com/blog/2021/08/getting-started-with-object-tracking-using-opencv/ (box around mask)

def find_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins=numLabels)

	hist = hist.astype("float")
	hist /= hist.sum()

	return hist

def plot_colors2(hist, centroids):
	bar = np.zeros((50, 300, 3), dtype="uint8")
	startX = 0

	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar

cap = cv2.VideoCapture(0)

while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #picks out the colour "red"
    lower = np.array([160,100,20])
    upper = np.array([179,255,255])
    mask = cv2.inRange(hsv, lower, upper)

    contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # crashes if there is nothing to detect; stop with if statement
    if contours:
        max_contour = contours[0]
        for contour in contours:
            if cv2.contourArea(contour)>cv2.contourArea(max_contour):
                max_contour=contour  
        contour=max_contour
        approx=cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True),True)
        x,y,w,h=cv2.boundingRect(approx)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)

    #Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    if contours:
        cv2.imshow('section', frame[y:y+h, x:x+w])

    # currently unable to stop crashing if there is no red to evaluate the K-Means of
    if contours:
        img = frame[y+4:y+h-4,x+4:x+w-4]
    else:
        img = frame
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent 	as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    cv2.imshow('dominant', bar)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#when everything done, release the capture
cap.release()
cv2.destroyAllWindows()
