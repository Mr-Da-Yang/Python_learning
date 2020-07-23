import pygal
from die import Die
die_1 = Die()
die_2 = Die(6)

results = []
for roll_num in range(5000):
    result = die_1.roll() * die_2.roll()
    results.append(result)

frequencies =[]
max_result = die_1.num_sides * die_2.num_sides
for value in range(1, max_result+1):
    frequency = results.count(value)
    frequencies.append(frequency)
print(frequencies)
hist = pygal.Bar()
hist.title = 'Results of rolling D6*D6 5000 times.'
hist.x_labels = list(range(1, max_result+1))
hist.x_title = 'Result'
hist.y_title = 'Frequency of Result'

hist.add('D6*D6', frequencies)
hist.render_to_file('dice_multiple_visual.svg')