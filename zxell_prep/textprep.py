"""前処理で共有するテキスト整形(extract_sample.py / tokenize_full.py から import)。

2 スクリプトで同一のロジックを使うことが重要(トークナイザ学習のサンプルと
本番トークン化で文の作り方が食い違うと、圧縮効率の見積りが狂う)。
"""

import html
import re

WS = re.compile(r"\s+")
_TAIL = re.compile(r"[\s…\.。・]+$")   # 末尾の省略記号・句点(要約の「…」切り詰め対策)
# HTML 残骸(review20 7 章: 本文の 0.005% にタグ、en の 0.28% にエンティティが生で残る)
_TAG = re.compile(r"<(?:/?[A-Za-z][^<>]{0,200}|!--.*?--)>", re.S)


def clean(s):
    """空白・改行・タブを 1 個のスペースに正規化(1 記事 1 行で扱うため)。
    2026-09-27: HTML タグの除去と文字実体参照(&amp; 等)の復号を追加。タグ除去 → 復号の順
    (逆だと &lt;p&gt; のような「タグに見える文字列」を誤って消す)。"""
    if not s:
        return ""
    if "<" in s:
        s = _TAG.sub(" ", s)
    if "&" in s:
        s = html.unescape(s)
    return WS.sub(" ", s).strip()


def _appears_near_start(needle, hay, slack=80):
    """needle が hay の冒頭付近(needle 長 + slack 文字以内)に現れるか。
    needle 末尾の「…」等は落として比較する(description は本文冒頭の切り詰めであることが多い)。"""
    core = _TAIL.sub("", needle)
    if len(core) < 10:          # 短すぎる断片は偶然一致しやすいので判定しない
        return False
    return core in hay[: len(core) + slack]


def compose_ja(title, description, content):
    """ja 記事のテキスト組み立て。

    確定事項 11(ja のみ title + description + content_text を連結)を前提に、
    2026-09-27 の改良: 後続部分の冒頭に既に含まれている部分は連結しない。
    ja のフィードは description が本文冒頭の写し、本文が title から始まる、という
    ケースが多く、素直に連結すると同じ文が 2〜3 回繰り返される(review19/20)。

    判定順: まず description が content 冒頭にあれば省く → 次に title が
    (残った)テキスト冒頭にあれば省く。他言語は content のみなので対象外。
    戻り値: (text, dropped) — dropped は省いた部位の集合(統計用)。
    """
    t, d, c = clean(title), clean(description), clean(content)
    dropped = set()
    body = c
    if d:
        if c and _appears_near_start(d, c):
            dropped.add("description")
        else:
            body = (d + " " + c) if c else d
    if t:
        if body and _appears_near_start(t, body):
            dropped.add("title")
        else:
            body = (t + " " + body) if body else t
    return body, dropped
