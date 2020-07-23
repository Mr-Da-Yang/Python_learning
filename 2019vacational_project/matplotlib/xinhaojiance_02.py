import numpy as np
import math
import scipy.signal as juan
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['KaiTi']#黑体：SimHei  宋体：SimSun   楷体KaiTi    微软雅黑体：Microsoft YaHei
plt.rcParams['axes.unicode_minus'] = False#这两用于写汉字

n1 = np.arange(0,32,1)
dom = [True if (i>=8 and i<=23) else False for i in n1] #使用列表解析的方法
s=0*(n1<=7)+(0.7**n1)*dom+0*(n1>=24)#信号的表示

noise=np.random.normal(0, 0.004, len(n1))
x=s+noise#将均值为0，方差为0.004的噪声与信号相加

h1=(0.5**(15-n1))*(n1<=15)#为了便于对照，我们将几个滤波器长度都设成一样
h2=(0.9**(15-n1))*(n1<=15)
h3=0*(n1<=7)+(0.7**(31-n1))*dom+0*(n1>=24)



def convolve(h):#两函数进行卷积
    y=juan.convolve(x,h/(math.sqrt(sum(h**2))),mode='full')
    return y

y1=convolve(h1)
y2=convolve(h2)
y3=convolve(h3)

fig1,(ax1,ax2)=plt.subplots(2,1)
ax1.stem(s,use_line_collection='True',label='原始信号')
ax2.stem(x,use_line_collection='True',label='加噪信号')
fig2,(ax3,ax4,ax5)=plt.subplots(3,1)
ax3.stem(y1,use_line_collection='True',label='h1滤波')
ax4.stem(y2,use_line_collection='True',label='h2滤波')
ax5.stem(y3,use_line_collection='True',label='匹配滤波')
ax1.legend()#写图例
ax2.legend(loc="upper right")
ax3.legend()
ax4.legend()
ax5.legend()
plt.show()
