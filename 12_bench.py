import time, random, pickle, urllib.request
from concurrent.futures import ThreadPoolExecutor

users = list(pickle.load(open(r"D:\MiND\recs.pkl", "rb")).keys())
URL = "http://127.0.0.1:8000/recommend?user_id="
N = 2000

def hit(_):
    u = random.choice(users)
    t0 = time.time()
    urllib.request.urlopen(URL + u).read()
    return time.time() - t0

t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    lat = list(ex.map(hit, range(N)))
dur = time.time() - t0
lat.sort()
print("请求数:", N)
print("QPS   :", round(N / dur, 1))
print("P50   :", round(lat[int(N * 0.5)] * 1000, 1), "ms")
print("P99   :", round(lat[int(N * 0.99)] * 1000, 1), "ms")