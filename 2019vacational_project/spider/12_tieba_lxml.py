from lxml import etree
import requests
url = 'https://tieba.baidu.com/mo/q/m?word=%E6%9D%8E%E6%AF%85&page_from_search=index&tn6=bdISP&tn4=bdKSW&tn7=bdPSB&lm=16842752&lp=6093&sub4=%E8%BF%9B%E5%90%A7'
headers ={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}
response = requests.get(url,headers=headers)

html =etree.HTML(response.text)
ret_list = html.xpath("//li[@*]")
print(ret_list)
for li in ret_list:
    item={}
    item['href'] = li.xpath('./a/@href')[0] if len(li.xpath('./a/@href'))>0 else None
    item['url'] = li.xpath('./a/div[1]/span/text()')[0] if len(li.xpath('./a/div[1]/span/text()')) >0 else None
    print(item)