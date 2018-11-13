import cv2
import os

output_dir = './'
img_name = '00000001.jpg'
img = cv2.imread('new_frames/bag/'+img_name)
x, y, width, height = (153, 65, 65, 64)#cv2.selectROI(img)
x, y = int(x + width/2), int(y + height/2)
# print(x, y, width, height)
patch_size = 32
d = int(patch_size/2)
cv2.rectangle(img, (x-d,y-32-d), (x+d,y-32+d), (0,0,255), 1)
# cv2.rectangle(img, (x,y), (x+width,y+height), (0,0,255), 3)
img_name = os.path.join(output_dir,img_name)
# print(img_name)
cv2.imwrite(img_name, img)