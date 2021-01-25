# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:07:20 2021

@author: 电脑
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from pylab import *

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 640
    height_new = 320
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new


#读取数据
matchpoints = np.loadtxt(r"C:\\Users\\电脑\\Desktop\\matchpoints.txt")
height,width = matchpoints.shape

inlinerpoints = np.loadtxt(r"C:\\Users\\电脑\\Desktop\\inlinerpoints.txt")
height1,width1 = inlinerpoints.shape

#读取原始图像
path1 = r"C:\\Users\\电脑\\Desktop\\dst1.txt"
files1 = open(path1)
lines1 = files1.readlines()
print(path1)
for line in lines1:
    print(line)
    img_src1 = cv2.imread(line[:])
    #img_src = cv2.cvtColor(img_src,cv2.COLOR_BAYER_BG2BGR)
    print(img_src1)
#    cv2.imshow("img_dst1",img_src1)

path = r"C:\\Users\\电脑\\Desktop\\src.txt"
files = open(path)
lines = files.readlines()
print(path)
for line in lines:
    print(line)
    img_src = cv2.imread(line[:])
    #img_src = cv.cvtColor(img_src,cv.COLOR_BAYER_BG2BGR)
    print(img_src)
#    cv2.imshow("img_dst",img_src)
    

#首先实现16进制转成8进制
num=int(width/4)
for i in range(num):
    matchpoints2 = matchpoints[i*height:(i+1)*height,:] / 256
    inlinerpoints2 = inlinerpoints[i*height1:(i+1)*height1,:] / 256

dstx = matchpoints2[:,0]
dstx = list(map(int,dstx))
dstxx = [i + 64 for i in dstx]#对应每个位置加64，图像拼接像素下X位置转变
# print(dstx)
# print(dstxx)
dsty = matchpoints2[:,1]
dsty = list(map(int,dsty))
print(dsty)

dstx1 = matchpoints2[:,2]
dstx11 = list(map(int,dstx1))
#dstx1 = [i + 64 for i in dstx11]#对应每个位置加64，图像拼接像素下X位置转变

dsty1 = matchpoints2[:,3]
dsty1 = list(map(int,dsty1))


srcx = inlinerpoints2[:,0]
srcy = inlinerpoints2[:,1]
srcx1 = inlinerpoints2[:,2]
srcy1 = inlinerpoints2[:,3]


#data1 = (matchpoints2[:,0],matchpoints2[:,1])
#B = np.trunc(data1)
C = [list(i) for i in zip(dstx,dsty)]

CC = [list(i) for i in zip(dstx11,dsty1)]
#C = np.trunc(C)
#b = [tuple(x) for x in C]
b = [tuple(x) for x in C]
print(b)

bb = [tuple(x) for x in CC]
print(bb)

print("******************************************************")


# data2 = (inlinerpoints2[:,0],inlinerpoints2[:,1])
# B1 = np.trunc(data2)
# C1 = [list(i) for i in zip(srcx,srcy)]
# C1 = np.trunc(C1)
# #b = [tuple(x) for x in C]
# b1 = tuple(tuple(x) for x in C1)
# print(b1)


for coor in b:
    for co in bb:
        cv2.circle(img_src1,coor,0,(100,20,255),-1)
        cv2.circle(img_src,co,0,(100,20,255),-1)
# for co in bb:
#     cv2.circle(img_src1,co, 0,(100,20,25),-1)
#cv2.imshow('img',img_src)
hmerge = np.hstack((img_src,img_src1))
#hmerge = img_resize(hmerge)
cv2.imwrite(r"D:\fy.png", hmerge)

 
#读取图片信息到数组中
im = np.array(Image.open(r"D:\fy.png"))
#绘制图像
plt.imshow(im)
#带连线特征点
# x = np.array(list(zip(dstx,dstx1))).flatten()
# x = list(map(int,x))
# y = np.array(list(zip(dsty,dsty1))).flatten()
# y = list(map(int,y))
# print(x)
# print(y)

x = np.array(list(zip(dstx1,dstxx))).flatten()
x = list(map(int,x))
y = np.array(list(zip(dsty1,dsty))).flatten()
y = list(map(int,y))
print(x)
print(y)
#co = [range(255),range(255),range(255)]
#plot(x, y, 'r')
#将数组中的前两个点进行连线
for i in range(100):
    plt.plot(x[2*(i-1):2*i], y[2*(i-1):2*i],color = 'r')

#添加标题信息
plt.title('Plotting: "point_result"') 
#隐藏坐标轴
#plt.axis('off')
#显示到屏幕窗口
plt.show()



# #cv2.resizeWindow("result", 128,64)
# # Black color in BGR 
# # color = (0, 0, 255)
# # lineType = 1
# # # Line thickness of 5 px 
# # for co in b:
# #     #hmerge = cv2.line(hmerge,co,(87,28),color,lineType)
# #     for co1 in bb:
# #         #hmerge = cv2.line(hmerge,co,(87,28),color,lineType)
# #         hmerge = cv2.line(hmerge,co,co1,color,lineType)
        
        
        
        
# # for i in range(len(C)):

# #     plt.plot(C[i],CC[i], color='r')
# #     plt.scatter(C[i],CC[i], color='b')

# # cv2.namedWindow('result',0) #设置为WINDOW_NORMAL可以任意缩放
# # cv2.resizeWindow("result", 128,64)
# # cv2.imshow("result",hmerge)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # cv2.imwrite(r"D:\fy.png", hmerge)




