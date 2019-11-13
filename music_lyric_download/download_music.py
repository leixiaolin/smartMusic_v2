# -*- coding:utf-8 -*-
# https://blog.csdn.net/qq_39317214/article/details/80573046
# 第一部分：爬取数据

import requests
import re
import os
import json
from bs4 import BeautifulSoup


# 发起响应
def get_html(url):
    headers = {
        'Host': 'music.163.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.8'
    }
    proxies = {
        'https': 'https://110.52.234.72',
        'https': 'https://119.101.112.66'
    }
    formdata = {
        "params": 'Wk3drXP2/Nj8YbOQoL3ORmBM784lqxwm0VELQyBipJWx/rd8fUmklRZ6vL+G1f2dbZ/8WE7f25gWe+2BdXp3+d2AwkiTy5DxeVd4SiHX5qat+jU642hSysQVtHDfJHmCi6rjndr/YEBSccqnzIbueeA9H08OlzAZoYa5T6xlbQpxgtTdX5E1MF6R71ykxkS8',
        "encSecKey": '1d5e93ee97662d6f9dfaf07dbe4b9d4f9ffe6b90b484d8acc14696214a556000198d51ce3d87d9123db07f96307c919c02d84fa4a204e9d0a387404141fd43400fb2ec9aaa07ae99d99df133cc6d4c31ee8ab7859d83351b154c1ab2bed81a84159a25956ed1485551639e37fc3502ab049a03051ca40f85ef4dd648aabe9286'
    }
    try:
        response = requests.get(url, headers=headers)
        html = response.content
        return html
    except:
        print('request error')
        pass


# 函数：按照歌曲id，提取歌词内容
def download_by_music_id(music_id):
    headers = {
        'Host': 'music.163.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.8'
    }
    lrc_url = 'http://music.163.com/api/song/lyric?' + 'id=' + str(music_id) + '&lv=1&kv=1&tv=-1'
    r = requests.get(lrc_url, headers=headers)
    json_obj = r.text
    j = json.loads(json_obj)
    try:
        lrc = j['lrc']['lyric']
        pat = re.compile(r'\[.*\]')
        lrc = re.sub(pat, "", lrc)
        lrc = lrc.strip()
        return lrc
    except:
        pass


# 函数：按照歌手id，发起请求，解析网页，提取歌曲id
def get_music_ids_by_musician_id(singer_id):
    singer_url = 'http://music.163.com/artist?id={}'.format(singer_id)
    r = get_html(singer_url)
    soupObj = BeautifulSoup(r, 'lxml')
    song_ids = soupObj.find('textarea').text
    jobj = json.loads(song_ids)
    ids = {}
    for item in jobj:
        print(str(item['id']) + ":" + item['name'])
        ids[item['name']] = item['id']
    return ids


# 创建文件夹，在文件夹下存储每首歌的歌词
# os.mkdir创建目录，os.chdir改变当前工作目录到指定的路径
def download_lyric(uid):
    try:
        os.mkdir("./data/" + str(uid))
    except:
        pass

    os.chdir("./data/" + str(uid))
    music_ids = get_music_ids_by_musician_id(uid)
    for key in music_ids:
        try:
            text = download_by_music_id(music_ids[key])
            file = open(key + '.txt', 'a')
            file.write(key + '\n')
            file.write(str(text))
            file.close()
        except:
            file.close()
            continue


if __name__ == '__main__':
    download_lyric(12138269)