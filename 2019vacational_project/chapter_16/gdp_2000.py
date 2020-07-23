import csv
from country_codes import get_country_code
from pygal_maps_world.maps import World
from pygal.style import RotateStyle, LightColorizedStyle

gdps = {}
filename = 'gdp_csv.csv'
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        if row[2] == '2000':
            value = int(float(row[3]))
            country_name = row[0]
            code = get_country_code(country_name)
            if code:
                gdps[code] = value

gdp_1, gdp_2, gdp_3 = {}, {},{}
for name, gdp in gdps.items():
    if gdp < 50000000000:
        gdp_1[name] = gdp
    elif gdp < 100000000000:
        gdp_2[name] = gdp
    else:
        gdp_3[name] = gdp

wm_style = RotateStyle('#338899', base_style=LightColorizedStyle)
wm = World(style=wm_style)
wm.title = 'World Gdp in 2000, by Country'
wm.add('<50000000000', gdp_1)
wm.add('<100000000000', gdp_2)
wm.add('>100bn', gdp_3)
wm.render_to_file('gdp_2000.svg')




