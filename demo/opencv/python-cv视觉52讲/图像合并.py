'''
图像拼接技术
1：图像融合步骤
a)导入两张图片


'''
import cv2 as cv                            #引入库
'''
cv.imread(path,readstyle)
第一个参数为路径，第二 个参数为读取方式：
读取方式如下：
cv.IMREAD_COLOR:读入一副彩色图片
cv.IMREEAD_GRAYSCALE:以灰度模式读入图片
cv.IMREAD_UNCHANGED:读入一副图片，包括alpha通道
该函数返回是一个ndarray格式的三维数组，(417, 633, 3) 前两个为图片的宽X长，第三位为图像的深度
3维彩图
'''
bg=cv.imread('im/bg.jpg',cv.IMREAD_COLOR)   #导入图片
fg=cv.imread('im/fg.jpg',cv.IMREAD_COLOR)   #导入图片
print(bg.shape)#显示图片bg.jpg的大小和维度
print(fg.shape)#显示图片fg.jpg的大小和维度
#图像合并，需要调整图片大小，使其保持一致
dim=(800,400)
'''
void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR)  

参数说明：
src - 原图
dst - 目标图像。当参数dsize不为0时，dst的大小为size；否则，它的大小需要根据src的大小，参数fx和fy决定。dst的类型（type）和src图像相同
dsize - 目标图像大小
所以，参数dsize和参数(fx, fy)不能够同时为0
fx - 水平轴上的比例因子。
fy - 垂直轴上的比例因子。
最后一个参数插值方法，是默认值，放大时最好选 INTER_LINEAR ，缩小时最好选 INTER_AREA。
'''
bg_size=cv.resize(bg,dim,interpolation=cv.INTER_AREA)
fg_size=cv.resize(fg,dim,interpolation=cv.INTER_AREA)
print(bg_size.shape)
print(fg_size.shape)
'''
可实现两个大小、类型均相同的数组（一般为 Mat 类型）按照设定权重叠加在一起。

void addWeighted(InputArray src1,double alpha,InputArray src2,double beta,double gamma,OutputArray dst,int dtype =-1);

src1,需要加权的第一个数组，通常是一个 Mat。
alpha,第一个数组的权重。
src2,需要加权的第二个数组，需要和第一个数组拥有相同的尺寸和通道数。
beta,第二个数组的权重。
gamma,dst[i] = src1[i] * alpha + src2[i] * beta + gamma ; 通常设为 0。
dst,输出的数组，需要和输入的两个数组拥有相同的尺寸和通道数。
dtype ，输出阵列的可选深度，默认值为 -1。深度为数据存储类型，有 8 位，16 位，32 位等等。
'''
blend=cv.addWeighted(bg_size,0.3,fg_size,0.7,0)#图像融合
cv.imshow('bg',blend)
cv.waitKey()