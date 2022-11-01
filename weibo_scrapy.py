# !/usr/bin/env python3.8
# -*- coding:utf-8 -*-
# author:XieCheng
# datetime:2022/11/01
# software: PyCharm

import pandas as pd
import requests
from bs4 import BeautifulSoup
import lxml
import re


def url_generate(page):
    """
    - # 翻页
    url_: #当鲁迅付完尾款#
    """
    url_ = "https://s.weibo.com/weibo?q=%23%E5%BD%93%E9%B2%81%E8%BF%85%E4%BB%98%E5%AE%8C%E5%B0%BE%E6%AC%BE%23&page="
    url_g = url_ + str(page)
    return url_g


def get_response_wb(url):
    """
    - # 当页数据获取
    note: 修改 cookie,登录-检查-网络 查看
    """
    # 修改
    cookie = " 手动修改cookie "

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Cookie': cookie,
        'Host': 's.weibo.com',
        'Pragma': 'no-cache',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"',
        'sec-ch-ua-mobile': '?0',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print("无响应信息...")


def html_deal(response_text):
    """
    - # 当页数据处理抓取
    检查界面，确定根据什么标签获取那些数据等
    """
    beautiful = BeautifulSoup(response_text, "html")

    # 根据网页检查查询微博用户信息
    # cl = beautiful.find_all("p", {"class":"txt"})
    cl = beautiful.find_all("p", class_="txt")
    df = pd.DataFrame(columns=["a", "b"])
    for i, item in enumerate(cl):
        mc = item["nick-name"]
        nr = item.text.strip().strip("\u200b").strip()
        # 展开数据也在页面中, 可以通过展开\收起判断是否完整内容
        if item.find_all("a")[-1].find(text=True) != "展开":
            # df.loc[i, "a"] = mc
            # df.loc[i, "b"] = nr
            df = df.append({"a": mc, "b": nr}, ignore_index=True)

    return df


if __name__ == "__main__":
    a = []
    for page_i in range(20):
        url_complete = url_generate(page_i)
        html = get_response_wb(url_complete)
        df_i = html_deal(html)
        a.append(df_i)

    df2 = pd.concat(a)
    print("爬取数据量: ", len(df2))
    df2.head()
