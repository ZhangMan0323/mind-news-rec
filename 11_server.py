# 服务接口：
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)
recs = pickle.load(open(r"D:\MiND\recs.pkl", "rb"))

@app.get("/health")
def health():
    return jsonify({"status": "ok", "users": len(recs)})

@app.get("/recommend")
def recommend():
    u = request.args.get("user_id", "")
    n = int(request.args.get("n", 10))
    if u not in recs:
        return jsonify({"error": "unknown user"}), 404
    return jsonify({"user_id": u, "items": recs[u][:n]})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)