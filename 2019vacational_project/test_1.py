import numpy as np
import math
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
pH0=0.3 #H0的先验概率
pH1=0.7 #H1的先验概率
c00,c11,c01,c10=0,0,2,1 #代价因子
Bayes_Threshold=(pH0*(c10-c00))/(pH1*(c01-c11))  #根据报告分析可以求出个个门限
MAP_Threshold=pH0/pH1
MAX_MIN_Threshold=math.exp(0.2-0.5)
N_P_Threshold=math.exp(1.3-0.5)


def siranbi(x):#不同的x值产生不同似然比函数，用于与判决门限的比较
    siranbi=norm.pdf(x,1,1)/norm.pdf(x,0,1)
    return siranbi

def panjue(menxian):
    d1 = 0     #这里令d1=0，每当有数据满足siranbi(x) > menxian时，累加一次
    s = np.random.normal(0, 1, 1000)
    for i in s:
        x = i
        if siranbi(x) > menxian:
            d1 = d1 + 1
    return (d1 / 1000)#返回虚警概率

menxians =[Bayes_Threshold,MAP_Threshold,MAX_MIN_Threshold,N_P_Threshold]
jifenxians=[0.5+math.log(Bayes_Threshold),0.5+math.log(MAP_Threshold),0.2,1.3]
real_cdf=[]#实验中的虚警概率，存放4个数据，用于后面的直方图
for menxian in menxians:#遍历不同的门限
    real_cdf.append(panjue(menxian))#调用判决函数，根据4种不同的门限计算出模拟产生的虚警概率
ideal_cdf=[]#理论中的虚警概率，存放4个数据，用于后面的直方图
for d in jifenxians:
    p_xujing, err1 = quad(lambda x: norm.pdf(x, 0, 1), d, float("inf"))#从X0——inf积分，求出理论上的虚警概率
    ideal_cdf.append(round(p_xujing,2))

name_list=['Bayes','MAP','nimimax','N-P']
x = list(range(len(ideal_cdf)))
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x, ideal_cdf, width=width, label='ideal',tick_label=name_list, fc='y')#将理论上的虚警概率在直方图上显示
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, real_cdf, width=width, label='real', fc='r')#将实验中的虚警概率在直方图上显示
plt.title("Comparison between theoretical value and real value ")
plt.legend()
plt.show()

