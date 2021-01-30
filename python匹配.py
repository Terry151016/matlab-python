# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 09:28:42 2021

@author: 电脑
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:07:20 2021

@author: 电脑
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'


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

path2 = r"C:\Users\电脑\Desktop\matrix.txt"
matr = np.loadtxt(path2)
matr=matr/256
matr[8]=1
#根据txt保存路径，读取原始图像
path1 = r"C:\Users\电脑\Desktop\dst.txt"
files1 = open(path1)
lines1 = files1.readlines()
img_dst=[]
img_dst1=[]
print(path1)


img_dst = cv2.imread(lines1[0].strip("\n"),0)
img_dst1 = cv2.imread(lines1[0].strip("\n"),0)
h1,w1=img_dst.shape
    #img_src = cv2.cvtColor(img_src,cv2.COLOR_BAYER_BG2BGR)
    #print(img_dst)
#    cv2.imshow("img_dst1",img_dst)

path = r"C:\\Users\\电脑\\Desktop\\src.txt"
files = open(path)
lines = files.readlines()
print(path)
img_src=[]
img_src1=[]
for line in lines:
    print(line)
    img_src = cv2.imread(line[:],0)
    img_src1 = cv2.imread(line[:],0)
    #img_src = cv.cvtColor(img_src,cv.COLOR_BAYER_BG2BGR)
    #print(img_src)
#    cv2.imshow("img_dst",img_src)

#首先将图像坐标16位转成8位
num = int(width/4)
for i in range(num):
    matchpoints2 = matchpoints[i*height:(i+1)*height,:] / 256
    inlinerpoints2 = inlinerpoints[i*height1:(i+1)*height1,:] / 256

#粗匹配图像的特征点坐标预处理
dstx = matchpoints2[:,0]
dstx = list(map(int,dstx))
dstxx = [i + w1 for i in dstx]  #对应每个位置加64，图像拼接像素下X位置转变
dsty = matchpoints2[:,1]
dsty = list(map(int,dsty))
dstx1 = matchpoints2[:,2]
dstx1 = list(map(int,dstx1))
dsty1 = matchpoints2[:,3]
dsty1 = list(map(int,dsty1))

#点内匹配点的坐标预处理
srcx = inlinerpoints2[:,0]
srcx = list(map(int,srcx))
srcxx = [i + w1 for i in srcx]
srcy = inlinerpoints2[:,1]
srcy = list(map(int,srcy))
srcx1 = inlinerpoints2[:,2]
srcx1 = list(map(int,srcx1))
srcy1 = inlinerpoints2[:,3]
srcy1 = list(map(int,srcy1))

#粗匹配点的坐标获取
dst_xy = [list(i) for i in zip(dstx,dsty)]  #dst特征点坐标，粗匹配
dst_xyy = [list(i) for i in zip(dstx1,dsty1)]
dstxy = [tuple(x) for x in dst_xy]  
#print(dstxy)
dstxyy = [tuple(x) for x in dst_xyy]
#print(dstxyy)
for i in dstxy:
    for j in dstxyy:
        cv2.circle(img_dst,i,0,(100,20,25),-1)
        cv2.circle(img_src,j,0,(100,20,25),-1)

hmerge = np.hstack((img_src,img_dst))  #图像拼接
#hmerge = img_resize(hmerge)
cv2.imwrite(r"D:\fy.png", hmerge)

print("******************************************************")

#内点匹配点的坐标获取
src_xy = [list(i) for i in zip(srcx,srcy)]  #dst特征点坐标，内点匹配
src_xyy = [list(i) for i in zip(srcx1,srcy1)]
srcxy = [tuple(x) for x in src_xy]  
#print(srcxy)
srcxyy = [tuple(x) for x in src_xyy]
#print(srcxyy)
for i in srcxy:
    for j in srcxyy:
        cv2.circle(img_dst1,i,0,(100,20,25),-1)
        cv2.circle(img_src1,j,0,(100,20,25),-1)
hmerge1 = np.hstack((img_src1,img_dst1))
#hmerge = img_resize(hmerge)
cv2.imwrite(r"D:\fy1.png", hmerge1)

plt.subplot(311)
#读取图片信息到数组中
im = np.array(Image.open(r"D:\fy.png"))
#绘制图像
plt.imshow(im)

#带连线特征点
x = np.array(list(zip(dstx1,dstxx))).flatten()
x = list(map(int,x))
y = np.array(list(zip(dsty1,dsty))).flatten()
y = list(map(int,y))
print(x)
print(y)
#将数组中的前两个点进行连线
for i in range(100):
    plt.plot(x[2*(i-1):2*i], y[2*(i-1):2*i])
#添加标题信息
plt.title('Plotting: "粗匹配"') 
#隐藏坐标轴
plt.axis('off')
#显示到屏幕窗口
# plt.show()

plt.subplot(312)
#读取图片信息到数组中
im1 = np.array(Image.open(r"D:\fy1.png"))
#绘制图像
plt.imshow(im1)
#带连线特征点
x1 = np.array(list(zip(srcx1,srcxx))).flatten()
x1 = list(map(int,x1))
y1 = np.array(list(zip(srcy1,srcy))).flatten()
y1 = list(map(int,y1))
print(x1)
print(y1)
#将数组中的前两个点进行连线
for i in range(100):
    plt.plot(x1[2*(i-1):2*i], y1[2*(i-1):2*i])
#添加标题信息
plt.title('Plotting: "内点匹配"') 
#隐藏坐标轴
plt.axis('off')
#显示到屏幕窗口

matr1=np.zeros((3, 3), np.float)
#imgi=np.zeros((320, 320), np.float)

for i in range(3):
    for j in range(3):
        matr1[i,j]=matr[i*3+j]
#matr1=matr.reshape((3,3),order='F')
matr1=(np.linalg.inv(matr1))
matr2=np.zeros((2, 3), np.float)
for i in range(2):
    for j in range(3):
        matr2[i,j]=matr1[i,j]
imgh=cv2.warpAffine(img_dst,matr2,(w1,h1),borderValue=(255,255,255)) 
#cv2.imshow('1',imgh)
#cv2.waitKey()
imgi=(imgh/2+img_src/2)
imgk=imgi.astype(np.uint8) 

plt.subplot(313)
plt.axis('off')
plt.imshow(imgk,cmap=plt.cm.gray)
plt.show()
#cv2.imshow('1',imgk)
#cv2.waitKey()






