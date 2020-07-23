import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

s=np.random.normal(0, 1, 500)
s_fit = np.linspace(s.min(), s.max())
plt.plot(s_fit, st.norm(2, 3).pdf(s_fit), linewidth=2, c='g')
plt.show()
