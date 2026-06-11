# Deploying `lamu serve`

Templates for running the LAMU HTTP serving plane (`lamu serve`, default
port 8020) under systemd. User-level units only — LAMU is a single-user,
loopback-by-default stack (ADR 0005, ADR 0014); a system-level variant is
deliberately out of scope until a real multi-user deployment needs it
(ADR 0018 covers the auth side when that day comes).

## Files

| File | Installs to | Purpose |
| ---- | ----------- | ------- |
| `lamu-serve.service` | `~/.config/systemd/user/` | The unit: foreground `lamu serve`, journald logging, restart-on-failure |
| `lamu-serve.env.example` | `~/.config/lamu/lamu-serve.env` | All `LAMU_*` knobs, documented, everything optional |

## Install

```bash
mkdir -p ~/.config/systemd/user ~/.config/lamu
cp deploy/lamu-serve.service ~/.config/systemd/user/
cp deploy/lamu-serve.env.example ~/.config/lamu/lamu-serve.env   # then edit
systemctl --user daemon-reload
systemctl --user enable --now lamu-serve
```

The unit expects the binary at `~/.cargo/bin/lamu` (`cargo install --path
lamu-cli`). If you run from a target dir instead, edit `ExecStart=`.

To survive logout (headless box), enable lingering once:

```bash
loginctl enable-linger $USER
```

## Operate

```bash
systemctl --user status lamu-serve        # state + recent log lines
journalctl --user -u lamu-serve -f        # follow logs
systemctl --user restart lamu-serve       # bounce
curl -s http://127.0.0.1:8020/health      # liveness probe
```

## Behavior notes

- **Double-start guard.** `lamu serve` acquires an RAII pidfile at
  `$XDG_RUNTIME_DIR/lamu-serve-{port}.pid` and refuses to start if a live
  instance holds it. If you start the unit while a manual `lamu serve` is
  running, the unit fails, retries 6× over 2 minutes, then gives up
  (`systemctl --user reset-failed lamu-serve` to clear).
- **GPU training lock.** The advisory lock at
  `~/.local/state/lamu/scheduler.lock` is checked *per request* inside
  serve (model loads refuse while training holds the GPU). It does not
  block startup, so the unit will not flap during training runs.
- **Cloud API keys** are loaded by the binary from
  `~/.config/lamu/api-keys.env` — they do not belong in `lamu-serve.env`
  and are never read by systemd.
- **SearXNG** (`SEARXNG_URL`, web grounding) is optional and best-effort.
  The unit orders weakly after `network-online.target` only; serve runs
  fine with SearXNG down.
- **Off-loopback binding** (`LAMU_BIND_HOST=0.0.0.0`) requires an auth
  token (`LAMU_API_TOKEN` or `~/.config/lamu/api-token`) or startup
  refuses — see `docs/API.md` and ADR 0012/0018.
- **Shutdown.** SIGTERM → graceful: pidfile removed, loaded llama-server
  children reaped. `TimeoutStopSec=30` gives slow child teardown room
  before SIGKILL.
