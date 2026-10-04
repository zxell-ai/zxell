# zxell_client

English | [日本語](README.ja.md)

Client v1 for zxell distributed training (Windows / Linux).
It loops over: register with the server → wait for administrator approval → fetch a task (with a lease) → process it → submit the result.

Processing in v1 is a **dummy** (`train` downloads the weights and verifies the checksum, then submits a dummy Δ;
every other type waits one second and reports completion). The purpose is to verify the distributed
infrastructure end to end — connectivity, the approval flow, leasing, and retries. Real training is implemented in phase 2 and later.

## Setup

### Windows

```bat
cd zxell_client
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python client.py
```

### Linux

```bash
cd zxell_client
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python client.py
```

The first run sends a registration request and saves the API key to `~/.zxell/client.json`
(`C:\Users\<name>\.zxell\client.json` on Windows).
Until an administrator approves it, the client shows "waiting for approval" and keeps polling; once approved, it starts working automatically.
To register again, delete this file.

## Options and environment variables

| Argument | Environment variable | Default | Description |
|---|---|---|---|
| `--server` | `ZXELL_SERVER_URL` | `https://api.zxell.ai` | Server URL |
| `--name` | `ZXELL_CLIENT_NAME` | hostname | Registered name |
| `--types` | `ZXELL_CLIENT_TYPES` | (empty = all types) | Task types to request, e.g. `train` for a GPU machine, `preprocess,eval,verify` for a CPU machine |
| `--state-file` | `ZXELL_STATE_FILE` | `~/.zxell/client.json` | Where the API key is stored |
| `--poll` | `ZXELL_POLL_SECONDS` | 30 | Polling interval in seconds when there is no task or approval is pending |
| `--once` | — | — | Exit when the queue is empty (for testing) |

Network errors and server 5xx responses are retried automatically with exponential backoff (up to 60 seconds × 5 attempts).
Disconnections caused by a change of the line's IP address (up to about 70 minutes) are also absorbed by these retries.

## Notes for use on the same LAN as the server

`https://api.zxell.ai` goes through an external public route, so machines on the LAN can also
connect to it as is. However, that route limits request bodies to 100MB, and uploads larger than that
do not go through. In that case, connect to the server directly, bypassing the public route.

Add one line with the server's LAN address to the hosts file,

- Windows: `C:\Windows\System32\drivers\etc\hosts` (edit with Notepad run as administrator)
- Linux: `/etc/hosts`

```
192.168.1.2 api.zxell.ai
```

and start the client with `--server http://api.zxell.ai` (nginx on the server speaks HTTP only. TLS is
terminated on the public route, so use `http://` rather than `https://` when connecting directly. LAN only).

When running on the server machine itself, use `--server http://127.0.0.1:8000`.
