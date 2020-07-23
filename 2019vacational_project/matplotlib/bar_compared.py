import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['KaiTi']#黑体：SimHei  宋体：SimSun   楷体KaiTi    微软雅黑体：Microsoft YaHei
plt.rcParams['axes.unicode_minus'] = False#这两用于写汉字

men=[100,200,300]
women=[10,30,50]
x=np.arange(1,len(men)+1)   #x=np.array([1,2,3])
width=0.2

fig, ax = plt.subplots()
ax.bar(x,men,width,color='SkyBlue',label='男')
rects2 = ax.bar(x+width,women,width,color='r',label='女')

plt.xticks(x+width/2,['G1','G2','G3'],fontsize=14)
ax.set_ylabel("Scores",fontsize=14)
ax.set_xlabel("pepole")
ax.set_title("分数")
# ax.axis([1,4,0,400])
# ax.set_xlim(t.min(), t.max())
# ax.set_ylim(s.min(), s.max())
# ax.grid()
for a,b in zip(x,men):
    plt.text(a,b+1,'%.0f'%b,ha = 'center',va = 'bottom',fontsize=14)
for c,d in zip(x+width,women):
    plt.text(c,d+1,'%.0f'%d,ha = 'center',va = 'bottom',fontsize=14)

ax.grid()
ax.legend(loc='upper right')#写图例
plt.show()

