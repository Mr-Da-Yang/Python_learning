import requests
from retrying import retry
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5558.400 QQBrowser/10.1.1695.400"}


@retry(stop_max_attempt_number = 3)
#    _表示只在这py文件中使用
def _parse_url(url):
    print("*" *100)
    response = requests.get(url, headers=headers, timeout=3)
    assert response.status_code == 200
    return response.content.decode()

def parse_url(url):
    try:
        html_str =_parse_url(url)
        return html_str

    except Exception as e:
        html_str =None
        print(e)


if __name__ == '__main__':
    url ="www.baidu.com"
    parse_url(url)

