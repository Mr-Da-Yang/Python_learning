import numpy as np
a=np.arange(0,14)

s=0*(a<=3)+0*(a>=8)+(0.7**a)*(a>=4 and a<=7)
print(s)