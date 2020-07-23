import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as juan
n1=np.arange(8,24,1)
s=0.7**n1
noise=np.random.normal(0, 1, len(n1))
x=s+noise
n2=np.arange(0,16,1)
h1=0.7**n1
y=juan.convolve(h1,x,mode='full')




fig,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(s,color='r')

ax2.plot(x,color='g')
ax3.plot(np.arange(-len(n1)+1,len(n1)),y)


plt.show()