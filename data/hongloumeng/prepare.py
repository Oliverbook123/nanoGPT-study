"""
准备《红楼梦》字符级数据集（120 回全文）。

思路沿用 shakespeare_char/prepare.py：先把全文拼成 input.txt，再按字符编码成
uint16 的 token id，存 train.bin / val.bin / meta.pkl，train.py 可直接使用。
下载部分基于「requests + BeautifulSoup 抓 purepen.com 目录页」的原始脚本，
在真实站点上验证后修了几处会翻车的地方（见下方注释）。

数据源：http://www.purepen.com/hlm/  （GB18030 编码，正文在唯一的 <center> 标签里）

产出：
    raw/NNN.txt        每回一个纯文本（缓存；重复运行会跳过已下载的）
    input.txt          拼接后的全文
    train.bin/val.bin  uint16 token id
    meta.pkl           vocab_size / stoi / itos

用法：
    python data/hongloumeng/prepare.py             # 下载（缺什么下什么）+ 编码
    python data/hongloumeng/prepare.py --refetch   # 忽略缓存，全部重新下载
"""
import os
import re
import sys
import time
import pickle
import requests
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = 'http://www.purepen.com/hlm/'
USER_AGENT = {'User-Agent': 'Mozilla/5.0 (compatible; nanoGPT-dataset-prep)'}
# 【修】只用正则挑章节链接。目录页里除了 001.htm–120.htm，还有两个 ../index.html 导航链接，
#      原来那种 base_url + link 的拼接会得到 http://www.purepen.com/hlm/../index.html，
#      于是把站点首页当成一章抓下来。
CHAPTER_RE = re.compile(r'^(\d{3})\.htm$')
# 【修】服务端声明的编码（ISO-8859-1）是错的，实际是 GB18030（GB2312 的超集）
ENCODING = 'gb18030'

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, 'raw')
INPUT_FILE = os.path.join(HERE, 'input.txt')

REFETCH = '--refetch' in sys.argv


def fetch(url, retries=3):
    """带超时/UA/重试的 GET。【修】原代码没有 timeout，站点一慢就会永久挂住。"""
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=20, headers=USER_AGENT)
            r.raise_for_status()
            r.encoding = ENCODING  # 【修】每个函数里都要设，否则 .text 会按 ISO-8859-1 解出乱码
            return r
        except requests.RequestException as e:
            last = e
            print(f"  第 {attempt} 次失败：{e}")
            time.sleep(attempt)  # 退避
    print(f"  放弃：{url}（{last}）")
    return None


def get_all_links(url):
    """获取给定URL页面中的所有子链接（保留原函数名/签名）"""
    r = fetch(url)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = a.get('href')
        if isinstance(href, str) and href.strip():
            links.append(href.strip())
    return links


def get_text_from_url(url):
    """从给定URL获取<title>和<center>标签内的文本内容（保留原函数名/签名）"""
    r = fetch(url)
    if r is None:
        return None, ""
    soup = BeautifulSoup(r.text, 'html.parser')

    title = soup.title.string.strip() if soup.title and soup.title.string else "无标题"
    # 正文在唯一的 <center> 里；用 get_text() 保留原文换行与段首缩进，
    # 只做规范化：统一换行、压掉多余空行。注意不要用 strip=True，
    # 那会把每个文本节点首尾空白吃掉，段落缩进就没了。
    centers = [c.get_text() for c in soup.find_all('center')]
    body = "\n".join(centers).replace('\r\n', '\n').replace('\r', '\n')
    body = re.sub(r'\n{3,}', '\n\n', body).strip()
    if not body:  # 兜底：万一某页结构不同
        body = "\n".join(p.get_text() for p in soup.find_all('p'))
    return title, f"{title}\n\n{body}"


def safe_filename(name):
    """把标题里文件系统不接受的字符换掉（macOS/Windows 都不能有 / \\ : * ? \" < > |）"""
    return re.sub(r'[/\\:*?"<>|\n]', '_', name).strip()


def save_text_to_file(text, filename):
    """将文本内容保存到文件，使用utf-8编码（保留原函数名/签名）"""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


def download_all(base_url=BASE_URL):
    """下载全部章节到 raw/，返回按回数排序的 (编号, 标题, 文件路径) 列表"""
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)

    all_links = get_all_links(base_url)
    chapters = []
    for link in all_links:
        if not link:
            continue
        # 去掉 #anchor / query，再匹配 NNN.htm
        m = CHAPTER_RE.match(link.split('#')[0].split('?')[0].strip())
        if m:
            chapters.append((int(m.group(1)), link))
    chapters = sorted(set(chapters))
    print(f"目录页共 {len(all_links)} 个链接，其中章节 {len(chapters)} 回"
          f"（{chapters[0][0]:03d}–{chapters[-1][0]:03d}）" if chapters else "没找到章节链接")
    if not chapters:
        raise SystemExit("目录页解析不出章节链接，站点结构可能变了")

    saved = []
    for i, (num, href) in enumerate(chapters, 1):
        # 【修】文件名用回数编号，不用标题：标题里有全角空格等字符，容易踩文件系统的雷
        out_path = os.path.join(RAW_DIR, f"{num:03d}.txt")
        if os.path.exists(out_path) and not REFETCH:
            with open(out_path, encoding='utf-8') as f:
                title = f.readline().strip()
            print(f"[{i}/{len(chapters)}] 已有缓存，跳过：{os.path.relpath(out_path, HERE)}")
            saved.append((num, title, out_path))
            continue

        url = urljoin(base_url, href)
        title, text = get_text_from_url(url)
        if not text:
            print(f"[{i}/{len(chapters)}] 跳过（抓取失败）：{url}")
            continue
        save_text_to_file(text, out_path)
        saved.append((num, title, out_path))
        print(f"[{i}/{len(chapters)}] {title}  ->  {os.path.relpath(out_path, HERE)}"
              f"（{len(text):,} 字）")
        time.sleep(0.3)  # 做个有礼貌的爬虫，别给人家站点压力
    return saved


def build_input(chapters):
    """把每回拼成一份 input.txt，回与回之间留空行"""
    parts = []
    for _num, _title, path in chapters:
        with open(path, encoding='utf-8') as f:
            parts.append(f.read().strip())
    data = '\n\n\n'.join(parts)
    with open(INPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(data)
    return data


def main():
    chapters = download_all()
    print(f"\n已就绪 {len(chapters)} 回")

    data = build_input(chapters)
    print(f"length of dataset in characters: {len(data):,}")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"vocab size: {vocab_size:,}")
    if vocab_size >= 65536:  # train.bin 用 uint16 存，超了就装不下
        raise ValueError(
            f"vocab_size={vocab_size} 超过 uint16 上限，需要改成 uint32 或先过滤生僻字符")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    n = len(data)
    train_data, val_data = data[:int(n * 0.9)], data[int(n * 0.9):]
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    train_ids.tofile(os.path.join(HERE, 'train.bin'))
    val_ids.tofile(os.path.join(HERE, 'val.bin'))

    with open(os.path.join(HERE, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}, f)

    print("\n产出：")
    for name in ['input.txt', 'train.bin', 'val.bin', 'meta.pkl']:
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            print(f"  {name:<10} {os.path.getsize(p):>12,} 字节")
    print(f"\n下一步训练：python train.py config/train_hongloumeng_char.py")


if __name__ == "__main__":
    main()
