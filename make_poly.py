import cv2
import os

output_dir = './'
img_name = '00000001.jpg'
image = cv2.imread('new_frames/bag/'+img_name)
# x, y, width, height = (153, 65, 65, 64)#cv2.selectROI(img)

patch_size = 32
d = int(patch_size/2)

refPt = []
 
def click(event, x, y, flags, param):
	# grab references to the global variables
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image, (x,y), 1, (0,255,0))

cv2.namedWindow("image")
cv2.setMouseCallback("image", click)
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break