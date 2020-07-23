##################### #两sin卷积
import scipy.signal as juan
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,5*np.pi,0.1)
s_1=np.sin(x)
s_2=np.cos(x)

fig, ax= plt.subplots()
ax.plot(x,s_1)
ax.plot(x,s_2)

y=juan.convolve(s_1,s_2) #卷积运算
c=np.arange(len(x) + len(x) - 1)
ax.plot(c,y,color='y')
plt.show()

#########################np.arange()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t1=np.arange(0,8,0.1)
t2 = np.arange(8, 23, 0.1)
t3 = np.arange(23,30,0.1)
print(type(t3))
s1 = 0*t1
s2 = 0.7**t2
s3 = 0*t3
print(type(s3))
fig, ax = plt.subplots()
ax.plot(t1,s1,color='b')
ax.plot(t2, s2,color='b')
ax.plot(t3,s3,color='b')
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()
plt.show()
