import matplotlib.pyplot as plt
from die import Die
values = []
die_1 = Die()
die_2 = Die(8)
nums=[]
for a in range(1000):
    value = die_1.roll() + die_2.roll()
    values.append(value)
for frequncy in range(2, die_1.num_sides+die_2.num_sides+1):
    num = values.count(frequncy)
    nums.append(num)
print(nums)

x = list(range(2, die_1.num_sides+die_2.num_sides+1))
plt.scatter(x, nums, c=nums, cmap=plt.cm.Blues, edgecolors='none',s=50)
plt.title('frequncey of two dice', fontsize=15)
plt.xlabel('value of two dice', fontsize=15)
plt.ylabel('frequncy')
plt.tick_params(axis='both',labelsize=20)
plt.show()

