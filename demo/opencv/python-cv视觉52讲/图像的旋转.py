import cv2 as cv
import  matplotlib.pyplot as plt
import numpy as np
im=cv.imread('im/coin.jpg')
gray_img=cv.cvtColor(im,cv.COLOR_BGR2GRAY)
print(im.shape)
(h,w)=im.shape[:2]
print(h)
print(w)
center=(w/2,h/2)
M=cv.getRotationMatrix2D(center,30,1)
rotated=cv.warpAffine(im,M,(w,h))
# plt.imshow(rotated)
# plt.show()





(thresh,output2)=cv.threshold(gray_img,120,255,cv.THRESH_BINARY)
print(thresh)
#print(output2)
output=cv.GaussianBlur(output2,(5,5),1)
out=cv.Canny(output,180,255)
plt.imshow(out,cmap=plt.get_cmap('gray'))
circles=cv.HoughCircles(out)
circles=np.uint16(np.around(circles))
# plt.show()
# def mask_of_image(image):
#     height=im.shape[0]
#     polygons=np.array([[(0,height),(1200,height),(250,100)]])
#     mask=np.zeros_like(im)
#     cv.fillPoly(mask,polygons,255)
#     masked_image=cv.bitwise_and(im,mask)
#     return masked_image
# lines=cv.HoughLinesP(out,1,np.pi/180,30)
# for line in lines:
#     x1,y1,x2,y2=line[0]
#     cv.line(im,(x1,y1),(x2,y2),(0,255,0),4)
#     #mask_of_image(out)
for i in circles[0,:]:
    cv.circle(im,(i[0],i[1],i[2]),(0,255,0),2)
    cv.circle(im,(i[0],i[1],i[2]),(0,255,0),2)
plt.imshow(im)
plt.show()