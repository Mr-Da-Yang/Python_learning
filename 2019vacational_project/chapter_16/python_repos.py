import requests
import pygal
from pygal.style import LightColorizedStyle as LCS, LightenStyle as LS
from pygal.style import RotateStyle
url = 'https://api.github.com/search/repositories?q=language:python&sort=stars'
r = requests.get(url)
print(r)
print("Status code:", r.status_code)
response_dict = r.json()
print(r.json())
print("Total repositories:", response_dict['total_count'])
repo_dicts = response_dict['items']
print("Repositories returned:", len(repo_dicts))


print("\nSelected information about each respository:")
names, plot_dicts = [], []
for repo_dict in repo_dicts:
    names.append(repo_dict['name'])
    plot_dict = {'value': repo_dict['stargazers_count'],
                 'label': str(repo_dict['description']),
                 'xlink':repo_dict['html_url'],}
    print(repo_dict['description'])
    plot_dicts.append(plot_dict)


my_style = RotateStyle('#333666', base_style=LCS)
my_config = pygal.Config()
my_config.title_font_size = 100
my_config.x_label_rotation = 45
my_config.show_legend = False

my_config.label_font_size = 50
my_config.major_label_font_size = 18
my_config.truncate_label = 10
my_config.show_y_guides = False
my_config.width = 1000

chart = pygal.Bar(my_config, style = my_style)
chart.title = 'Most-Starred Python Project on Github'

chart.x_labels = names
chart.add('', plot_dicts)
chart.render_to_file('python_repos.svg')