import pygal
from pygal.style import LightColorizedStyle as LCS, LightenStyle as LS

my_style = LS('#333666', base_style=LCS)
chart = pygal.Bar(style=my_style, x_label_rotation=45, show_legend=False)

chart.title = 'Python Projects'
chart.x_labels = ['httpie', 'django', 'flask']
plot_dict = [{'value':16101, 'label':'Description of httpie.','xlink':'https://github.com/vinta'},
             {'value':15028, 'label':'Description of django.','xlink':'https://www.baidu.com'},
             {'value':14798, 'label':'Description of flask.','xlink':'https://www.vmall.com/?cid=91895'},]


chart.add('', plot_dict)
chart.render_to_file('bar_description.svg')