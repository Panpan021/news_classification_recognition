import csv
import random
import time
from bs4 import BeautifulSoup
import requests


def get_url(file):
    with open(file, encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        # 处理行数据
        lst_u = []
        for row in csv_reader:
            lst_u.append([row[0], row[1]])
        return lst_u


def get_news_text(url):
    time.sleep(random.uniform(1, 3))
    res = requests.get(url)  # 新闻的网址
    res.encoding = res.apparent_encoding
    # 根据网站的编码方式修改解码方式，因为网页的文字编码方式多种多样有UTF-8 GBK这些解码方式如果不一致容易发生乱码，所以解码方式最好不要统一，而是交给网页自己来决定
    soup = BeautifulSoup(res.text, 'html5lib')  # 使用html5lib样式来解析网页
    # print(soup)#查看页面源代码
    title = soup.select('h1')[0].text  # 输出标题

    data = soup.select('p')  # 元素选择器
    news_text = ''
    for p in data:
        news_text += p.text.strip()
        # news_text += '\n'
    # print(news_text)
    return title, news_text


def write_csv(lst):
    # 写入进csv文件
    with open("data/all_news_text.csv", "w", encoding="utf-8", newline="") as f:
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        name = ['mark', 'news_text']
        csv_writer.writerow(name)
        # 4. 写入csv文件内容
        for l in lst:
            mark = l[0]
            news_text = l[1]
            z = [mark, news_text]
            csv_writer.writerow(z)
        print("写入数据成功")
        # 5. 关闭文件
        f.close()


if __name__ == '__main__':
    url = get_url("data/all_news_urls.csv")
    lst = []
    try:
        for u in url:
            if u != ['mark', 'url']:
                if u[1] != '':
                    if u[1].split('/')[2] == 'www.news.cn' and u[1].split('/')[4][:4].isdigit():
                        try:
                            title, news_txt = get_news_text(u[1])
                            lst.append([u[0], news_txt])
                            print('正在爬取第%s条新闻:%s' % (url.index(u) + 1, title))
                        except:
                            print('\033[91m跳过%s\033[0m' % u[1])
                            continue
                    else:
                        print('\033[91m跳过%s\033[0m' % u[1])
                else:
                    print('\033[91m跳过空值\033[0m')
            else:
                continue
    except:
        print('\033[91mFAILED!!! 爬取失败\033[0m')
    write_csv(lst)
