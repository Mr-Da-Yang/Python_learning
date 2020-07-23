import scipy.signal as juan
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,10*np.pi,0.1)
s_1=np.sin(x+np.pi*5)
s_2=np.cos(x)

fig, ax= plt.subplots()
ax.plot(x,s_1)
ax.plot(x,s_2)

y=juan.convolve(s_1,s_2) #卷积运算
c=np.arange(len(x) + len(x) - 1)
ax.plot(c,y,color='y')
plt.show()