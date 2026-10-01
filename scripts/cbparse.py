"""Helpers to pull embedded JSON out of Cricbuzz Next.js pages."""
import json, re, requests, time, os

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch(url, cache_name=None, refresh=False):
    path = os.path.join(RAW, cache_name) if cache_name else None
    if path and os.path.exists(path) and not refresh:
        return open(path, encoding="utf-8").read()
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    if path:
        open(path, "w", encoding="utf-8").write(r.text)
    time.sleep(0.6)
    return r.text


def flight(html):
    """Concatenate the React flight payload chunks into one decoded string."""
    out = []
    for m in re.finditer(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', html, re.S):
        try:
            out.append(json.loads('"' + m.group(1) + '"'))
        except Exception:
            pass
    return "".join(out)


def objects_for_key(text, key):
    """Yield every JSON value that follows "key": in text."""
    dec = json.JSONDecoder()
    for m in re.finditer(r'"%s":' % re.escape(key), text):
        try:
            val, _ = dec.raw_decode(text, m.end())
            yield val
        except Exception:
            continue
