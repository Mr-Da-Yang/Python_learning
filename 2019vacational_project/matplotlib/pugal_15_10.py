from random import choice
import pygal

class Random():

    def __init__(self, num=5):
        self.num = num
        self.x = [0]
        self.y = [0]

    def fill_walk(self):
        while len(self.x) < self.num:
            x = choice([-1, 1])
            x_dir = choice([0, 1, 2, 3, 4])
            x_step = x * x_dir

            y = choice([-1, 1])
            y_dir = choice([0, 1, 2, 3, 4])
            y_step = y * y_dir

            if x_step == 0 and y_step == 0:
                continue
            next_x = self.x[-1] + x_step
            next_y = self.y[-1] + y_step

            self.x.append(next_x)
            self.y.append(next_y)

random = Random()
random.fill_walk()
hist = pygal.Bar()
hist.title = 'random'

hist.x_labels = [random.x]
hist.add('randoming',random.y)
hist.render_to_file('aaa.svg')