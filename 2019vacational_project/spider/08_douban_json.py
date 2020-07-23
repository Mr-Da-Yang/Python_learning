import requests
import json
from pprint import pprint
url ='https://movie.douban.com/j/search_subjects?type=tv&tag=%E7%83%AD%E9%97%A8&page_limit=50&p'
headers ={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400',
          'Cookie':'ll="118165"; bid=V_TsjlNsmd4; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1563621874%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DLhJi5fLDcm1893qZHXPucB0vZ5Vg1023PaJe43IgPzYYo1NWi6vh8qCaYUbGY3gz%26wd%3D%26eqid%3Dcf6ddc3000359349000000035d32f9d8%22%5D; _pk_ses.100001.4cf6=*; ap_v=0,6.0; __utma=30149280.1927116107.1563621874.1563621874.1563621874.1; __utmb=30149280.0.10.1563621874; __utmc=30149280; __utmz=30149280.1563621874.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=223695111.18233745.1563621874.1563621874.1563621874.1; __utmb=223695111.0.10.1563621874; __utmc=223695111; __utmz=223695111.1563621874.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __yadk_uid=dHx2QCiYsDp28DiPQtZKk5VViDY9xswd; _vwo_uuid_v2=D37D9375C8C2C51D4FDA3B83D1C4654A0|7a2c7339d84781623766ca52d8d4623f; _pk_id.100001.4cf6=0185c03e7d6d5e4d.1563621874.1.1563621893.1563621874.'}

response = requests.get(url, headers=headers)
with open('dou1.txt','w',encoding='utf-8')as f:
    f.write(response.content.decode())
dic = json.loads(response.content.decode())

with open('douban.txt','w',encoding='utf-8')as f:
    f.write(json.dumps(dic,ensure_ascii=False,indent=2))
with open('douban.txt','r',encoding='utf-8')as f:
    ret = json.loads(f.read())

    print(type(f.read()))