#!/usr/bin/env python3
"""Prompt-eval harness — does a system prompt actually improve agent behavior?

Runs a deterministic task suite (tasks designed to BAIT specific LLM failure
modes) across {prompt variant} x {candidate model}, then a separate judge model
scores each response against the task's rubric. Output: a variant x failure-mode
pass-rate table + per-task detail. Answers "does prompt X help?" with a number,
not a vibe (Bible R5: data governs optimization).

Models route through the existing ~/.config/lamu/cloud-models.yaml (cloud only —
no `lamu serve`, GPU stays free). Deterministic: temperature 0, fixed corpus.

Judge integrity: pass a cross-vendor PANEL via --judges (comma list). Every
response is scored by ALL judges and a criterion PASSES only if a MAJORITY of
judges mark it pass — this cuts a single judge's self-bias. --judge stays as an
alias for a one-judge panel. --repeats N reruns each (variant x model x task) N
times so judge/model non-determinism shows up as variance, not a single coin flip.

Usage:
  python eval.py --models mimo-v2.5,deepseek-v4-flash --judge mimo-v2.5-pro \
                 --variants baseline,behavioral_core,operational_full,checklist
  python eval.py --task hallucination_trap --models mimo-v2.5 --judge deepseek-v4-pro  # one task
  python eval.py --models mimo-v2.5 --judges mimo-v2.5-pro,deepseek-v4-pro --repeats 3 # panel + repeats
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time
from dataclasses import dataclass, field

import httpx
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
CLOUD_MODELS = pathlib.Path.home() / ".config" / "lamu" / "cloud-models.yaml"
API_KEYS_ENV = pathlib.Path.home() / ".config" / "lamu" / "api-keys.env"
RESULTS = ROOT / "results"


def _load_env_file(path: pathlib.Path) -> None:
    """Source LAMU's api-keys.env into os.environ (lines `KEY=val` or
    `export KEY=val`, optional quotes). Does not overwrite an already-set var.
    Keys are NEVER printed (Bible R17)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


# ── model layer ──────────────────────────────────────────────────────────────
@dataclass
class ModelCfg:
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key: str
    chat_path: str | None = None


def load_cloud_models(path: pathlib.Path = CLOUD_MODELS) -> dict[str, ModelCfg]:
    assert path.exists(), f"cloud-models.yaml not found at {path}"
    _load_env_file(API_KEYS_ENV)
    doc = yaml.safe_load(path.read_text())
    out: dict[str, ModelCfg] = {}
    for m in doc.get("models", []):
        name = m.get("name")
        if not name or name in out:          # first entry per name wins (direct vendor endpoint)
            continue
        key = m.get("api_key") or (os.environ.get(m["api_key_env"]) if m.get("api_key_env") else None)
        out[name] = ModelCfg(
            name=name,
            provider=(m.get("provider") or "openai"),
            model_id=(m.get("model_id") or name),
            base_url=(m.get("base_url") or ""),
            api_key=(key or ""),
            chat_path=m.get("chat_path"),
        )
    return out


def call(model: ModelCfg, system: str, user: str, *, temperature: float = 0.0,
         max_tokens: int = 2048, timeout: float = 180.0) -> str:
    """One completion. OpenAI-compat by default; Anthropic if provider says so.
    Fail loud (Bible R7): a transport/HTTP error raises with context — the
    harness must never silently score a failed call as a model 'response'."""
    assert model.base_url, f"model '{model.name}' has no base_url (provider-resolved? set it in cloud-models.yaml)"
    assert model.api_key, f"model '{model.name}' has no api key (set api_key or api_key_env)"
    if model.provider == "anthropic":
        url = model.base_url.rstrip("/") + (model.chat_path or "/v1/messages")
        headers = {"x-api-key": model.api_key, "anthropic-version": "2023-06-01"}
        payload = {"model": model.model_id, "max_tokens": max_tokens, "temperature": temperature,
                   "system": system, "messages": [{"role": "user", "content": user}]}
        r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        blocks = r.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    # OpenAI-compatible (MiMo, DeepSeek, GLM, ...). Append just /chat/completions
    # to base_url — matching LAMU's chat_url (base already carries /v1 when needed,
    # e.g. xiaomimimo's base ends in /v1; deepseek accepts the un-versioned path).
    url = model.base_url.rstrip("/") + (model.chat_path or "/chat/completions")
    headers = {"Authorization": f"Bearer {model.api_key}"}
    msgs = ([{"role": "system", "content": system}] if system.strip() else []) + \
           [{"role": "user", "content": user}]
    payload = {"model": model.model_id, "messages": msgs,
               "temperature": temperature, "max_tokens": max_tokens}
    r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


# ── tasks + variants ─────────────────────────────────────────────────────────
@dataclass
class Crit:
    id: str
    ask: str           # yes/no question for the judge; PASS = good behavior
@dataclass
class Task:
    id: str
    failure_mode: str
    user: str
    rubric: list[Crit]
    system_extra: str = ""

def load_tasks(path: pathlib.Path) -> list[Task]:
    doc = yaml.safe_load(path.read_text())
    tasks = []
    for t in doc["tasks"]:
        tasks.append(Task(
            id=t["id"], failure_mode=t["failure_mode"], user=t["user"],
            system_extra=t.get("system_extra", ""),
            rubric=[Crit(c["id"], c["ask"]) for c in t["rubric"]],
        ))
    # Bible R6: validate the corpus shape up front, fail loud on a malformed task.
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    return tasks

def load_variants(path: pathlib.Path) -> dict[str, str]:
    doc = yaml.safe_load(path.read_text())
    out = {}
    for name, rel in doc["variants"].items():
        p = (ROOT / rel).resolve()
        assert p.exists(), f"variant '{name}' -> missing file {p}"
        out[name] = p.read_text()
    return out


# ── judge ────────────────────────────────────────────────────────────────────
JUDGE_SYS = (
    "You are a strict, literal code-review judge. You are given a TASK that was "
    "designed to bait a specific LLM failure mode, the assistant's RESPONSE, and "
    "a RUBRIC of yes/no criteria where YES means the assistant exhibited the GOOD "
    "behavior (avoided the trap). Judge ONLY what the response actually says — do "
    "not give benefit of the doubt. Reply with STRICT JSON: "
    '{"<crit_id>": {"pass": true|false, "why": "<=20 words"}, ...}. No prose.'
)

def judge_one(judge_model: ModelCfg, task: Task, response: str) -> dict:
    """One judge's verdict over the rubric: {crit_id: {pass: bool|None, why: str}}."""
    rubric = "\n".join(f"- {c.id}: {c.ask}" for c in task.rubric)
    user = (f"TASK ({task.failure_mode}):\n{task.user}\n\n"
            f"RESPONSE:\n{response}\n\n"
            f"RUBRIC (YES = good/avoided-the-trap):\n{rubric}\n\n"
            f"Return strict JSON keyed by criterion id.")
    try:
        raw = call(judge_model, JUDGE_SYS, user, temperature=0.0, max_tokens=1200)
    except Exception as e:                       # transport/HTTP failure for THIS judge
        return {c.id: {"pass": None, "why": f"judge-call-failed: {e}"} for c in task.rubric}
    # Validate at the boundary (Bible R23): tolerate ```json fences, reject junk loud.
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1].removeprefix("json").strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        return {c.id: {"pass": None, "why": f"judge-unparseable: {e}"} for c in task.rubric}
    return {c.id: {"pass": bool(obj.get(c.id, {}).get("pass")) if c.id in obj else None,
                   "why": (obj.get(c.id, {}) or {}).get("why", "")} for c in task.rubric}


def judge_panel(judges: list[ModelCfg], task: Task, response: str) -> dict:
    """Score `response` with a PANEL of judges and resolve each criterion by
    MAJORITY of the votes that actually parsed. Cross-vendor judges cut self-bias.

    Per criterion the consensus `pass` is:
      • True  if a strict majority of valid (non-None) votes are True
      • False if a strict majority are False
      • None  if there's no majority either way, or no valid votes at all
        (a tie, or every judge failed/was unparseable) — never a silent pass.
    Per-judge votes are kept under `votes` so results.json is auditable."""
    per_judge = {j.name: judge_one(j, task, response) for j in judges}
    out: dict[str, dict] = {}
    for c in task.rubric:
        votes = {jn: v[c.id] for jn, v in per_judge.items()}
        valid = [vv["pass"] for vv in votes.values() if vv["pass"] is not None]
        if not valid:
            consensus = None
        else:
            yes = sum(1 for p in valid if p)
            no = len(valid) - yes
            consensus = True if yes > no else False if no > yes else None  # tie -> None
        out[c.id] = {"pass": consensus,
                     "yes": sum(1 for p in valid if p), "no": sum(1 for p in valid if not p),
                     "n_valid": len(valid), "n_judges": len(judges),
                     "votes": {jn: {"pass": vv["pass"], "why": vv["why"]} for jn, vv in votes.items()}}
    return out


# ── runner ───────────────────────────────────────────────────────────────────
def run(models: list[str], judge_names: list[str], variant_names: list[str],
        only_task: str | None, repeats: int = 1) -> None:
    assert repeats >= 1, f"--repeats must be >= 1, got {repeats}"
    cfgs = load_cloud_models()
    # Dedup judges while preserving order; a 1-judge panel is the old behavior.
    judge_names = list(dict.fromkeys(judge_names))
    assert judge_names, "no judges selected"
    for n in models + judge_names:
        assert n in cfgs, f"model '{n}' not in cloud-models.yaml (have: {sorted(cfgs)[:8]}...)"
    judges = [cfgs[n] for n in judge_names]
    if len({j.provider for j in judges}) < len(judges):
        sys.stderr.write("WARN: panel has judges from the same vendor — self-bias not fully cut.\n")
    variants = load_variants(ROOT / "variants.yaml")
    for v in variant_names:
        assert v in variants, f"variant '{v}' not in variants.yaml (have: {sorted(variants)})"
    tasks = [t for t in load_tasks(ROOT / "tasks.yaml") if (only_task is None or t.id == only_task)]
    assert tasks, "no tasks selected (bad --task id?)"

    RESULTS.mkdir(exist_ok=True)
    rows = []
    total = len(variant_names) * len(models) * len(tasks) * repeats
    i = 0
    for vname in variant_names:
        for mname in models:
            for task in tasks:
                system = (variants[vname] + ("\n\n" + task.system_extra if task.system_extra else "")).strip()
                for rep in range(repeats):
                    i += 1
                    suffix = f" rep {rep+1}/{repeats}" if repeats > 1 else ""
                    sys.stderr.write(f"[{i}/{total}] {vname} x {mname} x {task.id}{suffix}\n")
                    try:
                        resp = call(cfgs[mname], system, task.user, temperature=0.0)
                    except Exception as e:                  # candidate transport/HTTP failure
                        resp = f"<call failed: {e}>"
                        verdict = {c.id: {"pass": None, "why": str(e), "yes": 0, "no": 0,
                                          "n_valid": 0, "n_judges": len(judges), "votes": {}}
                                   for c in task.rubric}
                    else:
                        verdict = judge_panel(judges, task, resp)
                    rows.append({"variant": vname, "model": mname, "task": task.id,
                                 "repeat": rep, "failure_mode": task.failure_mode,
                                 "verdict": verdict, "response": resp})
                    time.sleep(0.2)

    (RESULTS / "results.json").write_text(json.dumps(rows, indent=2))
    _report(rows, variant_names, models, judge_names, repeats)
    print(f"\nwrote {RESULTS/'report.md'} and {RESULTS/'results.json'}")


def _rate(rows) -> str:
    vals = [v["pass"] for r in rows for v in r["verdict"].values()]
    scored = [v for v in vals if v is not None]
    if not scored:
        return "n/a"
    return f"{100*sum(scored)/len(scored):.0f}% ({sum(scored)}/{len(scored)})"


def _report(rows, variant_names, models, judge_names, repeats=1) -> None:
    panel = ", ".join(f"`{j}`" for j in judge_names)
    rule = "majority of judges" if len(judge_names) > 1 else "the judge"
    lines = ["# Prompt-eval report", "",
             f"judge panel: {panel} · models: {', '.join(models)} · temp 0 · "
             f"repeats: {repeats} · {len(rows)} runs", "",
             f"PASS = the response exhibited the GOOD behavior (avoided the baited "
             f"failure mode), as scored by {rule}.",
             "", "## Overall pass-rate by prompt variant", "",
             "| variant | pass-rate |", "|---|---|"]
    for v in variant_names:
        lines.append(f"| {v} | {_rate([r for r in rows if r['variant']==v])} |")
    # by failure mode x variant
    modes = sorted({r["failure_mode"] for r in rows})
    lines += ["", "## Pass-rate by failure mode × variant", "",
              "| failure mode | " + " | ".join(variant_names) + " |",
              "|---|" + "---|"*len(variant_names)]
    for mode in modes:
        cells = [_rate([r for r in rows if r["failure_mode"]==mode and r["variant"]==v]) for v in variant_names]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    if repeats > 1:
        # Surface per-repeat variance: same cell rerun N times shouldn't swing wildly.
        lines += ["", "## Pass-rate per repeat (variance check)", "",
                  "| repeat | pass-rate |", "|---|---|"]
        for rep in range(repeats):
            lines.append(f"| {rep+1} | {_rate([r for r in rows if r['repeat']==rep])} |")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mimo-v2.5", help="comma-sep candidate model names from cloud-models.yaml")
    ap.add_argument("--judges", default=None,
                    help="comma-sep judge PANEL; a criterion passes only on a MAJORITY of judges. "
                         "Use DIFFERENT vendors to cut self-bias.")
    ap.add_argument("--judge", default=None, help="alias for a single-judge panel (back-compat with --judges)")
    ap.add_argument("--variants", default="baseline,behavioral_core,operational_full,checklist")
    ap.add_argument("--task", default=None, help="run only this task id")
    ap.add_argument("--repeats", type=int, default=1,
                    help="rerun each (variant x model x task) N times so judge/model non-determinism "
                         "shows up as variance (temp stays 0)")
    a = ap.parse_args()
    # --judges wins; --judge is the single-judge alias; default to one cross-vendor judge.
    if a.judges:
        judge_names = [s.strip() for s in a.judges.split(",") if s.strip()]
        if a.judge:
            sys.stderr.write("WARN: both --judges and --judge given; using --judges, ignoring --judge.\n")
    elif a.judge:
        judge_names = [a.judge.strip()]
    else:
        judge_names = ["mimo-v2.5-pro"]
    run(a.models.split(","), judge_names, a.variants.split(","), a.task, a.repeats)
