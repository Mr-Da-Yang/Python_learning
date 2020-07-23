import requests
from retrying_parse_url import parse_url
import re
import json


#1.url
url='https://36kr.com/'
#2.response
html_str=parse_url(url)
#3.提取
ret = re.findall('window.initialState=(.*?)</script>', html_str)[0]
print(ret
      )
#4.save
dic = json.loads(ret)
with open('36_kr.json','w',encoding='utf-8')as f:
    f.write(json.dumps(dic,ensure_ascii=False,indent=2))