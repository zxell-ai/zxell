"""zxell 分散学習クライアント v1。

サーバに登録 → 管理者の承認を待つ → タスクを取得 → 処理 → 結果を提出、を繰り返す。
v1 の処理はダミー（学習は行わない）で、分散基盤の疎通・リース・リトライの検証が目的。

対応 OS: Windows / Linux（Python 3.10+、依存は requests のみ）。

使い方:
    python client.py                     # 常駐ループ（Ctrl+C で終了）
    python client.py --once              # キューが空になったら終了（動作確認用）
    python client.py --server http://127.0.0.1:8000 --types preprocess,eval

設定は引数または環境変数（ZXELL_SERVER_URL / ZXELL_CLIENT_NAME / ZXELL_CLIENT_TYPES /
ZXELL_STATE_FILE / ZXELL_POLL_SECONDS）。API キーは初回登録時に発行され、
状態ファイル（既定 ~/.zxell/client.json）に保存される。再登録したい場合はこのファイルを消す。
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

POLL_SECONDS_DEFAULT = 30
RETRY_MAX = 5  # 通信エラー時のリトライ回数（1 リクエストあたり）
LOG_PREFIX = "[zxell-client]"


def log(msg):
    print("%s %s" % (LOG_PREFIX, msg), flush=True)


# ---------- 設定・状態ファイル ----------


def default_state_file():
    return Path.home() / ".zxell" / "client.json"


def load_state(path):
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    if os.name == "posix":  # API キーを含むため所有者のみ読み書き可に（Windows は既定 ACL に任せる）
        os.chmod(path, 0o600)


def detect_capabilities():
    """スペックの自己申告値。サーバ側でタスク種別・H の調整に使う。"""
    caps = {
        "os": platform.system(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "client_version": "v1-dummy",
    }
    try:  # GPU は nvidia-smi があれば拾う（Windows / Linux 共通で動く）
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, mem = out.stdout.strip().splitlines()[0].split(",", 1)
            caps["gpu"] = name.strip()
            caps["vram"] = mem.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return caps


# ---------- HTTP（リトライ付き） ----------


class Api:
    def __init__(self, server, api_key=None):
        self.server = server.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def request(self, method, path, **kw):
        """接続エラー・5xx は指数バックオフで再試行する。4xx はそのまま返す
        （呼び出し側が意味を判断する: 403=未承認、404=タスク/重みなし、409=リース喪失）。"""
        url = self.server + path
        kw.setdefault("timeout", 60)
        delay = 2
        for attempt in range(1, RETRY_MAX + 1):
            try:
                resp = self.session.request(method, url, **kw)
                if resp.status_code < 500:
                    return resp
                err = "HTTP %d" % resp.status_code
            except requests.RequestException as e:
                err = type(e).__name__
            if attempt == RETRY_MAX:
                raise RuntimeError("%s %s failed after %d attempts (%s)" % (method, path, RETRY_MAX, err))
            log("%s %s -> %s、%d 秒後にリトライ (%d/%d)" % (method, path, err, delay, attempt, RETRY_MAX))
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise AssertionError("unreachable")

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, **kw):
        return self.request("POST", path, **kw)


# ---------- 登録・承認待ち ----------


def ensure_registered(server, name, state_file):
    state = load_state(state_file)
    if state and state.get("server") == server and state.get("api_key"):
        log("登録済み: client_id=%s (%s)" % (state["client_id"], state_file))
        return state
    api = Api(server)
    resp = api.post(
        "/api/clients/register",
        json={"name": name, "capabilities": detect_capabilities()},
    )
    resp.raise_for_status()
    body = resp.json()
    state = {"server": server, "client_id": body["client_id"], "api_key": body["api_key"]}
    save_state(state_file, state)
    log("登録申請しました: client_id=%s（管理者の承認待ち）" % body["client_id"])
    return state


# ---------- タスク処理（v1 はダミー） ----------


def handle_dummy(api, task):
    """preprocess / eval / verify のダミー処理。payload は読むだけで何もしない。"""
    started = time.time()
    time.sleep(1)
    return {"dummy": True, "elapsed_sec": round(time.time() - started, 3)}, None


def handle_train(api, task):
    """train のダミー処理: 最新重みを取得して checksum を検証し、ダミー Δ を返す。
    重み配布 → ローカル計算 → Δ 提出、という本番と同じ通信パターンを踏む。"""
    started = time.time()
    resp = api.get("/api/weights/latest")
    if resp.status_code == 404:
        return None, None  # 重み未登録なら処理できない（リース失効で再キューされる）
    resp.raise_for_status()
    meta = resp.json()
    version = meta["version"]

    dl = api.get("/api/weights/%d/download" % version)
    dl.raise_for_status()
    digest = hashlib.sha256(dl.content).hexdigest()
    if digest != meta["checksum_sha256"]:
        raise RuntimeError("重み v%d の checksum 不一致" % version)

    delta = b"zxell dummy delta v1\n"  # 本実装ではここが「H ステップ学習後の重み差分」になる
    metrics = {
        "dummy": True,
        "base_weight_version": version,
        "weight_bytes": len(dl.content),
        "elapsed_sec": round(time.time() - started, 3),
    }
    return metrics, {"base_weight_version": version, "artifact": delta}


def submit_result(api, task, metrics, train_extra):
    data = {"task_id": str(task["id"])}
    files = None
    if metrics is not None:
        data["metrics"] = json.dumps(metrics)
    if train_extra:
        data["base_weight_version"] = str(train_extra["base_weight_version"])
        files = {"artifact": ("delta_task_%d.bin" % task["id"], train_extra["artifact"])}
    resp = api.post("/api/results", data=data, files=files)
    if resp.status_code == 409:  # リース失効等で他クライアントに再割当された
        log("task %d: リースを失っていたため結果は破棄されました" % task["id"])
        return False
    resp.raise_for_status()
    return True


def process_one(api, task):
    log("task %d (%s) を処理します" % (task["id"], task["type"]))
    if task["type"] == "train":
        metrics, extra = handle_train(api, task)
        if metrics is None:
            log("task %d: サーバに重みが未登録のため train を処理できません（後で再試行）" % task["id"])
            return False
    else:
        metrics, extra = handle_dummy(api, task)
    if submit_result(api, task, metrics, extra):
        log("task %d: 結果を提出しました %s" % (task["id"], json.dumps(metrics, ensure_ascii=False)))
    return True


# ---------- メインループ ----------


def main():
    parser = argparse.ArgumentParser(description="zxell client v1")
    parser.add_argument("--server", default=os.environ.get("ZXELL_SERVER_URL", "https://api.zxell.ai"))
    parser.add_argument("--name", default=os.environ.get("ZXELL_CLIENT_NAME", socket.gethostname()))
    parser.add_argument("--types", default=os.environ.get("ZXELL_CLIENT_TYPES", ""),
                        help="希望タスク種別（カンマ区切り。例: train / preprocess,eval,verify。空なら全種別）")
    parser.add_argument("--state-file", default=os.environ.get("ZXELL_STATE_FILE", str(default_state_file())))
    parser.add_argument("--poll", type=int, default=int(os.environ.get("ZXELL_POLL_SECONDS", POLL_SECONDS_DEFAULT)),
                        help="タスクが無い/承認待ちのときのポーリング間隔（秒）")
    parser.add_argument("--once", action="store_true", help="キューが空になったら終了する（動作確認用）")
    args = parser.parse_args()

    state = ensure_registered(args.server, args.name, Path(args.state_file))
    api = Api(args.server, api_key=state["api_key"])
    path = "/api/tasks/next" + ("?types=%s" % args.types if args.types else "")

    approved_seen = False
    while True:
        resp = api.get(path)
        if resp.status_code == 200:
            if not approved_seen:
                log("承認を確認しました。稼働開始します")
                approved_seen = True
            process_one(api, resp.json())
            continue  # 連続で次のタスクを取りに行く
        if resp.status_code == 404:  # 承認済みだがタスクなし
            if not approved_seen:
                log("承認を確認しました。稼働開始します")
                approved_seen = True
            if args.once:
                log("キューが空になったため終了します（--once）")
                return 0
            time.sleep(args.poll)
        elif resp.status_code == 403:  # 未承認（pending / disabled）
            log("承認待ちです（%s）。%d 秒後に再確認します" % (resp.json().get("detail", ""), args.poll))
            time.sleep(args.poll)
        elif resp.status_code == 401:
            log("API キーが無効です。%s を削除して再登録してください" % args.state_file)
            return 1
        else:
            log("想定外の応答: HTTP %d %s" % (resp.status_code, resp.text[:200]))
            time.sleep(args.poll)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("終了します")
        sys.exit(0)
