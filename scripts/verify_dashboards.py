"""Run every dashboard panel query through Grafana's /api/ds/query and report empty or failing panels.

    uv run python scripts/verify_dashboards.py [--grafana http://127.0.0.1:3000] [--run RUN_ID] [--live-run RUN_ID]

Checks the dashboard definitions deployed in Grafana, including stat field selection.
History panels use the given run (default: the newest exported run) and every exported
experiment/sweep; the time range is the run's window. Live panels use the newest run
in Prometheus over the last 15 minutes. Exit code 1 if any query errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DASHBOARDS = ROOT / "observability" / "grafana" / "dashboards"
# Panels allowed to be empty for synthetic data (no GPU, or nothing to warn about).
MAY_BE_EMPTY = {"GPU utilization", "GPU memory", "GPU memory used", "Warnings", "Window spread across trials",
                "Cost so far", "Server requests and KV cache", "Prefix-cache hit rate", "Episode transitions (baseline → candidate)"}


def query(http: httpx.Client, target: dict, variables: dict, time_from: str, time_to: str,
          panel: dict | None = None) -> tuple[str, int]:
    t = json.loads(json.dumps(target))
    for key in ("expr", "rawSql"):
        if key in t:
            for name, value in variables.items():
                t[key] = t[key].replace(f"${name}", value)
    t["intervalMs"], t["maxDataPoints"] = 2000, 500
    resp = http.post("/api/ds/query", json={"queries": [t], "from": time_from, "to": time_to})
    body = resp.json()
    result = (body.get("results") or {}).get(t["refId"], {})
    if resp.status_code != 200 or result.get("error"):
        return f"ERROR {result.get('error') or body.get('message') or resp.status_code}", 0
    frames = result.get("frames", [])
    rows = sum(len(f["data"]["values"][0]) if f.get("data", {}).get("values") else 0 for f in frames)
    if rows and panel and panel["type"] == "stat" and not panel.get("options", {}).get("reduceOptions", {}).get("fields"):
        fields = [field for frame in frames for field in frame.get("schema", {}).get("fields", [])]
        if fields and not any(field.get("type") == "number" for field in fields):
            return "ERROR stat panel has text data but selects only numeric fields", rows
    return "ok", rows


def check(http: httpx.Client, dash: dict, variables: dict, time_from: str, time_to: str) -> list[str]:
    problems = []
    for panel in dash["panels"]:
        for target in panel.get("targets", []):
            status, rows = query(http, target, variables, time_from, time_to, panel=panel)
            mark = "ok" if status == "ok" and rows else ("empty" if status == "ok" else "FAIL")
            if mark == "FAIL" or (mark == "empty" and panel["title"] not in MAY_BE_EMPTY):
                problems.append(f"{dash['uid']}/{panel['title']}/{target['refId']}: {status} rows={rows}")
            print(f"  [{mark:^5}] {panel['title']} ({target['refId']}): rows={rows}" + ("" if status == "ok" else f" {status}"))
    return problems


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # panel titles use non-ASCII characters
    parser = argparse.ArgumentParser()
    parser.add_argument("--grafana", default="http://127.0.0.1:3000")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="admin")
    parser.add_argument("--anonymous", action="store_true", help="check public dashboards without login credentials")
    parser.add_argument("--run")
    parser.add_argument("--live-run")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--live-only", action="store_true", help="only the live dashboard (GPU-host Grafana)")
    args = parser.parse_args()
    http = httpx.Client(base_url=args.grafana, auth=None if args.anonymous else (args.user, args.password), timeout=30)
    pg = {"type": "grafana-postgresql-datasource", "uid": "rc-postgres"}

    def deployed(path: Path) -> dict:
        uid = json.loads(path.read_text())["uid"]
        response = http.get(f"/api/dashboards/uid/{uid}")
        response.raise_for_status()
        return response.json()["dashboard"]

    def sql_values(statement: str) -> list:
        resp = http.post("/api/ds/query", json={"queries": [{"refId": "A", "datasource": pg, "rawSql": statement,
                                                              "format": "table", "rawQuery": True}],
                                                 "from": "now-1y", "to": "now"}).json()
        frames = resp["results"]["A"].get("frames", [])
        return frames[0]["data"]["values"] if frames and frames[0]["data"]["values"] else []

    problems: list[str] = []
    run = None if args.live_only else (args.run or sql_values("SELECT run_id FROM runs ORDER BY created_utc DESC LIMIT 1")[0][0])
    start_ms, end_ms = (None, None) if args.live_only else (
        v[0] for v in sql_values(f"SELECT start_ms, end_ms FROM runs WHERE run_id = '{run}'"))
    experiments = [] if args.live_only else sql_values("SELECT name FROM experiments ORDER BY name")
    experiments = experiments[0] if experiments else []
    for name in ([] if args.live_only else ("runs-overview", "run-detail")):
        dash = deployed(DASHBOARDS / "history" / f"{name}.json")
        print(f"{dash['title']} (run {run})")
        problems += check(http, dash, {"run": run}, str(start_ms), str(end_ms))
    if experiments:
        dash = deployed(DASHBOARDS / "history" / "experiments.json")
    for experiment in experiments:
        print(f"{dash['title']} ({experiment})")
        problems += check(http, dash, {"experiment": experiment}, "now-1y", "now")
    if not args.skip_live:
        dash = deployed(DASHBOARDS / "live" / "live-run.json")
        live_run = args.live_run
        if not live_run:
            resp = http.get("/api/datasources/uid/rc-prometheus/resources/api/v1/label/run_id/values").json()
            live_run = sorted(resp.get("data") or [""])[-1]
        print(f"{dash['title']} (run {live_run})")
        problems += check(http, dash, {"run": live_run}, "now-15m", "now")
    print("\n" + ("all panels returned data" if not problems else "problems:\n  " + "\n  ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
