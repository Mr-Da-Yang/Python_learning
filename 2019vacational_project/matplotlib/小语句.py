import numpy as np
import scipy.special
from scipy.integrate import quad


a=np.random.rand(3,3)-0.5          #3*3矩阵    0-1
b=np.random.normal(0,pow(8, -0.5),(3,4)) #3*4矩阵   均值=0，方差根号8的倒数
jifen, err1 = quad(lambda x: x**2, 0, np.inf)#lambda相当于def一个函数，这里简写，否则x未识别