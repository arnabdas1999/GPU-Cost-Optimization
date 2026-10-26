"""Generate the Grafana dashboards from one definition.

    uv run python observability/build_dashboards.py

State codes, metric names and datasource UIDs come from rollout_cost.contract,
so dashboards cannot drift from the exporter or the export schema. The output
JSON is committed; tests/test_observability.py checks it matches this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout_cost.contract import DATASOURCE_UIDS, STATE_CODES  # noqa: E402

OUT = ROOT / "observability" / "grafana" / "dashboards"
PROM = {"type": "prometheus", "uid": DATASOURCE_UIDS["prometheus"]}
PG = {"type": "grafana-postgresql-datasource", "uid": DATASOURCE_UIDS["postgres"]}

STATE_COLORS = {
    "idle": "transparent", "queued": "#5c5c5c", "llm_wait": "#f2cc0c", "model_request": "#3274d9",
    "tool_wait": "#ff9830", "tool": "#a352cc", "file_op": "#8ab8ff", "grading_wait": "#ffcb7d",
    "grading": "#73bf69", "success": "#37872d", "failed_after_finish": "#e02f44",
    "failed_budget_exhausted": "#fa6400", "infra_error": "#8f3bb8",
}


class Layout:
    """Places panels left to right on Grafana's 24-column grid."""

    def __init__(self) -> None:
        self.x = self.y = self.row_h = 0
        self.next_id = 1

    def place(self, w: int, h: int) -> dict:
        if self.x + w > 24:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def newline(self) -> None:
        if self.x:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0

    def pid(self) -> int:
        self.next_id += 1
        return self.next_id - 1


def prom(expr: str, legend: str, ref: str = "A") -> dict:
    return {"datasource": PROM, "expr": expr, "legendFormat": legend, "refId": ref, "range": True}


def sql(query: str, fmt: str = "table", ref: str = "A") -> dict:
    return {"datasource": PG, "rawSql": " ".join(query.split()), "format": fmt, "rawQuery": True,
            "editorMode": "code", "refId": ref}


def panel(layout: Layout, kind: str, title: str, targets: list[dict], w: int = 8, h: int = 8, *,
          unit: str | None = None, options: dict | None = None, custom: dict | None = None,
          mappings: list | None = None, overrides: list | None = None, description: str | None = None,
          decimals: int | None = None, min_value: float | None = None, links: list | None = None) -> dict:
    defaults: dict = {"custom": custom or {}, "mappings": mappings or []}
    if unit:
        defaults["unit"] = unit
    if decimals is not None:
        defaults["decimals"] = decimals
    if min_value is not None:
        defaults["min"] = min_value
    if links:
        defaults["links"] = links
    p = {"id": layout.pid(), "type": kind, "title": title, "gridPos": layout.place(w, h),
         "datasource": targets[0]["datasource"] if targets else None, "targets": targets,
         "fieldConfig": {"defaults": defaults, "overrides": overrides or []}, "options": options or {}}
    if description:
        p["description"] = description
    return p


def text(layout: Layout, title: str, content: str, w: int = 24, h: int = 3) -> dict:
    return {"id": layout.pid(), "type": "text", "title": title, "gridPos": layout.place(w, h),
            "options": {"mode": "markdown", "content": content}}


def stat(layout: Layout, title: str, target: dict, unit: str | None = None, w: int = 4, h: int = 4, **kw) -> dict:
    return panel(layout, "stat", title, [target], w, h, unit=unit,
                 options={"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                          "colorMode": kw.pop("color_mode", "value"), "graphMode": "none",
                          "textMode": kw.pop("text_mode", "auto"), "justifyMode": "center"}, **kw)


def timeseries(layout: Layout, title: str, targets: list[dict], unit: str | None = None, w: int = 8, h: int = 8,
               step: bool = False, **kw) -> dict:
    custom = {"lineWidth": 2, "fillOpacity": 10, "showPoints": "never", "spanNulls": True}
    if step:
        custom["lineInterpolation"] = "stepAfter"
    return panel(layout, "timeseries", title, targets, w, h, unit=unit, custom=custom,
                 options={"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}, **kw)


def table(layout: Layout, title: str, target: dict, w: int = 12, h: int = 8, **kw) -> dict:
    return panel(layout, "table", title, [target], w, h, options={"showHeader": True, "cellHeight": "sm"}, **kw)


def barchart(layout: Layout, title: str, target: dict, x_field: str, unit: str | None = None, w: int = 12, h: int = 8,
             **kw) -> dict:
    return panel(layout, "barchart", title, [target], w, h, unit=unit,
                 options={"xField": x_field, "orientation": "auto", "showValue": "auto", "stacking": "none",
                          "groupWidth": 0.7, "barWidth": 0.9, "legend": {"displayMode": "list", "placement": "bottom"},
                          "xTickLabelRotation": -30}, **kw)


def value_mappings(colors: dict[str, str], codes: dict[str, int] | None = None) -> list:
    options = {}
    for index, (name, color) in enumerate(colors.items()):
        key = str(codes[name]) if codes else name
        options[key] = {"text": name, "color": color, "index": index}
    return [{"type": "value", "options": options}]


def color_cells(field: str, colors: dict[str, str]) -> dict:
    return {"matcher": {"id": "byName", "options": field},
            "properties": [{"id": "custom.cellOptions", "value": {"type": "color-background"}},
                           {"id": "mappings", "value": value_mappings(colors)}]}


def hidden(field: str) -> dict:
    return {"matcher": {"id": "byName", "options": field}, "properties": [{"id": "custom.hidden", "value": True}]}


def dashboard(uid: str, title: str, panels: list[dict], variables: list[dict], *, refresh: str = "",
              time_from: str = "now-24h", tags: list[str] | None = None, links: list | None = None) -> dict:
    return {
        "uid": uid, "title": title, "tags": ["rollout-cost", *(tags or [])], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "editable": True, "graphTooltip": 1, "refresh": refresh,
        "time": {"from": time_from, "to": "now"}, "templating": {"list": variables}, "panels": panels,
        "links": links or [], "annotations": {"list": []},
    }


NAV_LINKS = [{"title": "Rollout Cost Lab", "type": "dashboards", "tags": ["rollout-cost"], "asDropdown": True,
              "includeVars": False, "keepTime": False}]


def pg_variable(name: str, query: str, label: str) -> dict:
    return {"type": "query", "name": name, "label": label, "datasource": PG, "query": query, "definition": query,
            "refresh": 1, "sort": 0, "includeAll": False, "multi": False, "current": {}, "options": []}


# ---------------------------------------------------------------------------


def live() -> dict:
    L = Layout()
    r = '{run_id="$run"}'
    p = [
        text(L, "", "**Live run** — runner metrics (`rc_*`) and vLLM server metrics. Values for mock or scripted "
                    "runs are SYNTHETIC. The runner exports while a run is in progress; between runs the panels go quiet.", h=2),
        stat(L, "Phase", prom(f"max by (phase) (rc_run_phase{r}) == 1", "{{phase}}"), text_mode="name", w=4),
        stat(L, "Episodes graded", prom(f"sum(rc_episodes_terminal_total{r}) or vector(0)", "graded"), w=4),
        stat(L, "Submitted", prom(f"max(rc_episodes_submitted{r})", "submitted"), w=4),
        stat(L, "Successes", prom('sum(rc_episodes_terminal_total{run_id="$run",category="success"}) or vector(0)', "success"), w=4),
        stat(L, "Allocated window", prom(f"max(rc_window_seconds{r})", "window"), unit="s", w=4),
        stat(L, "Cost so far", prom(f"max(rc_allocation_cost{r})", "cost"), unit="currencyUSD", decimals=4, w=4,
             description="Window seconds × hourly rate. Empty when the price is unknown (never shown as 0)."),
    ]
    L.newline()
    p += [
        timeseries(L, "Episodes active vs cap", [prom(f"max(rc_episodes_active{r})", "active", "A"),
                                                 prom(f"max(rc_episode_cap{r})", "cap", "B")], min_value=0),
        timeseries(L, "Model requests: in flight, waiting, cap", [
            prom(f"max(rc_llm_requests_inflight{r})", "in flight", "A"),
            prom(f"max(rc_llm_requests_waiting{r})", "waiting for slot", "B"),
            prom(f"max(rc_llm_request_cap{r})", "cap", "C")], min_value=0),
        timeseries(L, "Tool slots (fixed CPU budget)", [
            prom(f"max(rc_tool_slots_busy{r})", "busy", "A"), prom(f"max(rc_tool_slots_waiting{r})", "waiting", "B"),
            prom(f"max(rc_tool_slots_total{r})", "total", "C")], min_value=0),
    ]
    L.newline()
    p += [
        timeseries(L, "vLLM requests running / waiting", [prom("sum(vllm:num_requests_running)", "running", "A"),
                                                          prom("sum(vllm:num_requests_waiting)", "waiting", "B")], min_value=0),
        timeseries(L, "KV-cache usage", [prom("max(vllm:kv_cache_usage_perc) or max(vllm:gpu_cache_usage_perc)", "kv cache")],
                   unit="percentunit", min_value=0),
        timeseries(L, "Tokens per second", [prom("sum(rate(vllm:generation_tokens_total[30s]))", "generated", "A"),
                                            prom("sum(rate(vllm:prompt_tokens_total[30s]))", "prompt", "B")], min_value=0),
    ]
    L.newline()
    p += [
        timeseries(L, "Prefix-cache hit rate", [prom("sum(rate(vllm:prefix_cache_hits_total[1m])) / "
                                                     "sum(rate(vllm:prefix_cache_queries_total[1m]))", "hit rate")],
                   unit="percentunit", min_value=0),
        timeseries(L, "Server time to first token", [
            prom("histogram_quantile(0.5, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[1m])))", "p50", "A"),
            prom("histogram_quantile(0.9, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[1m])))", "p90", "B")],
                   unit="s", min_value=0),
        timeseries(L, "Graded episodes by category", [prom(f"sum by (category) (rc_episodes_terminal_total{r})", "{{category}}")],
                   min_value=0),
    ]
    L.newline()
    p += [
        timeseries(L, "GPU utilization", [prom(f"max by (gpu) (rc_gpu_utilization_percent{r})", "GPU {{gpu}}")],
                   unit="percent", min_value=0, w=12),
        timeseries(L, "GPU memory", [prom(f"max by (gpu) (rc_gpu_memory_used_mib{r})", "used GPU {{gpu}}", "A"),
                                     prom(f"max by (gpu) (rc_gpu_memory_total_mib{r})", "total GPU {{gpu}}", "B")],
                   unit="decmbytes", min_value=0, w=12),
    ]
    variables = [{"type": "query", "name": "run", "label": "Run", "datasource": PROM,
                  "query": {"query": "label_values(rc_run_info, run_id)", "refId": "run"},
                  "definition": "label_values(rc_run_info, run_id)", "refresh": 2, "sort": 2,
                  "includeAll": False, "multi": False, "current": {}, "options": []}]
    return dashboard("rc-live", "Rollout Cost Lab · Live run", p, variables, refresh="5s", time_from="now-15m",
                     tags=["live"], links=NAV_LINKS)


def runs_overview() -> dict:
    L = Layout()
    detail = ("/d/rc-run-detail/rollout-cost-lab-run-detail?var-run=${__value.raw}"
              "&from=${__data.fields.start_ms}&to=${__data.fields.end_ms}")
    p = [
        text(L, "", "**Runs** — every exported run. Click a run id to open its detail dashboard at the run's time window. "
                    "Rows marked SYNTHETIC came from the scripted model or the mock server and are not GPU evidence. "
                    "Load data with `rollout-cost export`.", h=2),
        stat(L, "Measured runs", sql("SELECT count(*) AS measured FROM runs WHERE NOT synthetic"), w=6),
        stat(L, "Synthetic runs", sql("SELECT count(*) AS synthetic FROM runs WHERE synthetic"), w=6),
        stat(L, "Experiments", sql("SELECT count(*) AS experiments FROM experiments WHERE kind = 'paired'"), w=6),
        stat(L, "Sweeps", sql("SELECT count(*) AS sweeps FROM experiments WHERE kind = 'sweep'"), w=6),
    ]
    L.newline()
    p.append(table(L, "Runs", sql("""
        SELECT run_id, CASE WHEN synthetic THEN 'SYNTHETIC' ELSE 'measured' END AS data, config_name, label,
               max_active_episodes AS episode_cap, max_inflight_llm_requests AS request_cap, tool_workers,
               successes, valid_episodes AS valid, cohort_size AS submitted, infra_errors, wall_timeouts,
               window_s, window_cost, cost_per_success, warnings_count AS warnings, created_utc, start_ms, end_ms
        FROM runs ORDER BY created_utc DESC"""), w=24, h=10,
        overrides=[hidden("start_ms"), hidden("end_ms"),
                   {"matcher": {"id": "byName", "options": "run_id"},
                    "properties": [{"id": "links", "value": [{"title": "Open run", "url": detail}]}]},
                   color_cells("data", {"SYNTHETIC": "#8f3bb8", "measured": "#37872d"}),
                   {"matcher": {"id": "byName", "options": "window_s"}, "properties": [{"id": "unit", "value": "s"},
                                                                                         {"id": "decimals", "value": 1}]},
                   {"matcher": {"id": "byRegexp", "options": "window_cost|cost_per_success"},
                    "properties": [{"id": "unit", "value": "currencyUSD"}, {"id": "decimals", "value": 4}]}]))
    p += [
        barchart(L, "Allocated window by run (latest 15)", sql("""
            SELECT run_label, window_s FROM (SELECT config_name || ' · ' || right(run_id, 4) AS run_label, window_s,
            created_utc FROM runs ORDER BY created_utc DESC LIMIT 15) t ORDER BY created_utc"""), "run_label", unit="s"),
        barchart(L, "Successes / valid / submitted (latest 15)", sql("""
            SELECT run_label, successes, valid, submitted FROM (SELECT config_name || ' · ' || right(run_id, 4) AS run_label,
            successes, valid_episodes AS valid, cohort_size AS submitted, created_utc FROM runs
            ORDER BY created_utc DESC LIMIT 15) t ORDER BY created_utc"""), "run_label"),
    ]
    return dashboard("rc-runs", "Rollout Cost Lab · Runs", p, [], time_from="now-30d", tags=["history"], links=NAV_LINKS)


def run_detail() -> dict:
    L = Layout()
    where = "WHERE run_id = '$run'"
    jump = "/d/rc-run-detail/rollout-cost-lab-run-detail?var-run=${__value.raw}&from=${__data.fields.start_ms}&to=${__data.fields.end_ms}"
    p = [
        stat(L, "Data", sql(f"""SELECT CASE WHEN synthetic THEN 'SYNTHETIC — ' || coalesce(synthetic_reason, '')
                                ELSE 'Measured run' END AS data FROM runs {where}"""), w=8, h=3, text_mode="value",
             color_mode="background", mappings=[{"type": "regex", "options": {"pattern": "SYNTHETIC.*", "result": {"color": "#8f3bb8"}}},
                                               {"type": "value", "options": {"Measured run": {"color": "#37872d"}}}]),
        table(L, "Run window (click to zoom the dashboard to this run)", sql(f"""
            SELECT run_id, config_name, model, gpu, executor, cache_reset, start_ms, end_ms FROM runs {where}"""),
              w=16, h=3, overrides=[hidden("start_ms"), hidden("end_ms"),
                                    {"matcher": {"id": "byName", "options": "run_id"},
                                     "properties": [{"id": "links", "value": [{"title": "Zoom to run", "url": jump}]}]}]),
    ]
    L.newline()
    p += [
        stat(L, "Allocated window", sql(f"SELECT window_s FROM runs {where}"), unit="s", decimals=1),
        stat(L, "Window cost", sql(f"SELECT window_cost FROM runs {where}"), unit="currencyUSD", decimals=4,
             description="Empty when the allocation price is unknown."),
        stat(L, "Cost per success", sql(f"SELECT cost_per_success FROM runs {where}"), unit="currencyUSD", decimals=4,
             description="Undefined (empty) with no successes or unknown price."),
        stat(L, "Success / valid / submitted", sql(f"""SELECT successes || ' / ' || valid_episodes || ' / ' || cohort_size
                                                       AS outcome FROM runs {where}"""), text_mode="value"),
        stat(L, "Episode cap full", sql(f"SELECT steady_state_fraction FROM runs {where}"), unit="percentunit",
             description="Share of the window during which max_active_episodes slots were all occupied."),
        stat(L, "Tool share of episode time", sql(f"SELECT tool_time_fraction FROM runs {where}"), unit="percentunit"),
    ]
    L.newline()
    timeline_mappings = value_mappings(STATE_COLORS, STATE_CODES)
    p.append(panel(L, "state-timeline", "Episode timeline", [sql(f"""
        SELECT ts AS time, episode_id AS metric, state AS value FROM episode_states {where} ORDER BY ts""", "time_series")],
        w=24, h=14, mappings=timeline_mappings, custom={"fillOpacity": 85, "lineWidth": 0},
        options={"mergeValues": True, "showValue": "never", "alignValue": "left", "rowHeight": 0.85,
                 "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
        description="One row per episode: queued, waiting for a model-request slot, in a model request, waiting for a "
                    "tool slot, running tests, file actions, grading, then the outcome."))
    L.newline()

    def series(names: list[str], cap_column: str | None, cap_label: str) -> list[dict]:
        targets = [sql(f"""SELECT ts AS time, series AS metric, value FROM concurrency {where}
                           AND series IN ({", ".join(f"'{n}'" for n in names)}) ORDER BY ts""", "time_series", "A")]
        if cap_column:
            targets.append(sql(f"""SELECT t AS time, '{cap_label}' AS metric, v AS value FROM (
                SELECT window_start AS t, {cap_column} AS v FROM runs {where}
                UNION ALL SELECT window_end, {cap_column} FROM runs {where}) caps ORDER BY t""", "time_series", "B"))
        return targets

    p += [
        timeseries(L, "Episodes active vs cap", series(["active_episodes"], "max_active_episodes", "cap"), step=True, min_value=0),
        timeseries(L, "Model requests vs cap", series(["llm_inflight", "llm_waiting"], "max_inflight_llm_requests", "cap"),
                   step=True, min_value=0),
        timeseries(L, "Tool slots vs workers", series(["tool_busy", "tool_waiting"], "tool_workers", "workers"),
                   step=True, min_value=0),
    ]
    L.newline()
    p += [
        barchart(L, "Where episode time went", sql(f"""SELECT component, share FROM attribution {where} ORDER BY share DESC"""),
                 "component", unit="percentunit", w=8),
        table(L, "Latency percentiles (s)", sql(f"SELECT metric, n, p50, p90, p99, max FROM latency {where}"), w=8,
              overrides=[{"matcher": {"id": "byRegexp", "options": "p50|p90|p99|max"},
                          "properties": [{"id": "decimals", "value": 2}]}]),
        panel(L, "histogram", "Model request duration", [sql(f"SELECT duration_s FROM requests {where} AND status = 'ok'")],
              w=8, unit="s", options={"bucketCount": 20, "legend": {"showLegend": False}}),
    ]
    L.newline()
    p += [
        table(L, "Outcomes", sql(f"""SELECT category, count(*) AS episodes FROM episodes {where}
                                     GROUP BY category ORDER BY category"""), w=6,
              overrides=[color_cells("category", {k: STATE_COLORS[k] for k in
                                                  ("success", "failed_after_finish", "failed_budget_exhausted", "infra_error")})]),
        table(L, "Stop reason × grader outcome", sql(f"""SELECT stop_reason, grader_outcome, count(*) AS episodes
                                                         FROM episodes {where} GROUP BY 1, 2 ORDER BY 1, 2"""), w=8),
        table(L, "Per task", sql(f"""SELECT task_id, count(*) AS episodes,
                                     sum(CASE WHEN category = 'success' THEN 1 ELSE 0 END) AS passes,
                                     round(avg(service_s)::numeric, 2) AS mean_service_s, round(avg(steps)::numeric, 1) AS mean_steps,
                                     sum(protocol_errors) AS protocol_errors FROM episodes {where} GROUP BY task_id ORDER BY task_id"""),
              w=10),
    ]
    L.newline()
    p += [
        timeseries(L, "GPU utilization", [sql(f"""SELECT ts AS time, 'GPU ' || gpu AS metric, utilization AS value
                                                  FROM gpu_samples {where} ORDER BY ts""", "time_series")],
                   unit="percent", min_value=0, w=8),
        timeseries(L, "GPU memory used", [sql(f"""SELECT ts AS time, 'GPU ' || gpu AS metric, memory_used AS value
                                                  FROM gpu_samples {where} ORDER BY ts""", "time_series")],
                   unit="decmbytes", min_value=0, w=8),
        timeseries(L, "Server requests and KV cache", [sql(f"""SELECT ts AS time, signal AS metric, value FROM server_samples
                    {where} AND signal IN ('requests_running', 'requests_waiting', 'kv_cache_usage') ORDER BY ts""",
                                                           "time_series")], min_value=0, w=8),
    ]
    L.newline()
    p += [
        table(L, "Warnings", sql(f"SELECT message FROM run_warnings {where} ORDER BY ordinal"), w=24, h=5),
        table(L, "Episodes", sql(f"""SELECT idx, episode_id, category, stop_reason, stop_detail, infrastructure_status,
                                     grader_outcome, steps, protocol_errors, round(service_s::numeric, 2) AS service_s,
                                     round(llm_wait_s::numeric, 2) AS llm_wait_s, round(request_s::numeric, 2) AS request_s,
                                     round(tool_wait_s::numeric, 2) AS tool_wait_s, round(tool_s::numeric, 2) AS tool_s,
                                     prompt_tokens, completion_tokens FROM episodes {where} ORDER BY idx"""), w=24, h=10),
    ]
    variables = [pg_variable("run", "SELECT run_id FROM runs ORDER BY created_utc DESC", "Run")]
    return dashboard("rc-run-detail", "Rollout Cost Lab · Run detail", p, variables, time_from="now-24h",
                     tags=["history"], links=NAV_LINKS)


def experiments() -> dict:
    L = Layout()
    where = "WHERE experiment = '$experiment'"
    verdict_colors = [{"type": "value", "options": {"ACCEPTED": {"color": "#37872d"}, "NOT ACCEPTED": {"color": "#e02f44"},
                                                    "A/A CALIBRATION": {"color": "#3274d9"}}},
                      {"type": "regex", "options": {"pattern": "CHOSEN.*", "result": {"color": "#3274d9"}}}]
    pivot = """SELECT x_label,
                      max({col}) FILTER (WHERE role = 'baseline') AS baseline,
                      max({col}) FILTER (WHERE role = 'candidate') AS candidate
               FROM experiment_trials {where} GROUP BY x_label ORDER BY min(ordinal)"""
    p = [
        stat(L, "Decision", sql("SELECT verdict FROM experiments WHERE name = '$experiment'"), w=8, h=4,
             text_mode="value", color_mode="background", mappings=verdict_colors),
        stat(L, "Data", sql("""SELECT CASE WHEN synthetic THEN 'SYNTHETIC' ELSE 'measured' END AS data
                               FROM experiments WHERE name = '$experiment'"""), w=4, h=4, text_mode="value",
             color_mode="background", mappings=[{"type": "value", "options": {"SYNTHETIC": {"color": "#8f3bb8"},
                                                                               "measured": {"color": "#37872d"}}}]),
        stat(L, "Kind / status", sql("SELECT kind || ' · ' || coalesce(status, '') AS kind FROM experiments WHERE name = '$experiment'"),
             w=4, h=4, text_mode="value"),
        stat(L, "Trials", sql("SELECT trials FROM experiments WHERE name = '$experiment'"), w=4, h=4),
        stat(L, "Window spread across trials", sql("SELECT window_spread_pct / 100 AS spread FROM experiments WHERE name = '$experiment'"),
             unit="percentunit", w=4, h=4, description="Noise floor: the pooled reduction must exceed this."),
    ]
    L.newline()
    p += [
        barchart(L, "Allocated window", sql(pivot.format(col="window_s", where=where)), "x_label", unit="s", w=8),
        barchart(L, "Window cost", sql(pivot.format(col="window_cost", where=where)), "x_label", unit="currencyUSD", w=8),
        barchart(L, "Successes", sql(pivot.format(col="successes", where=where)), "x_label", w=8),
    ]
    L.newline()
    p += [
        table(L, "Acceptance checks", sql(f"""SELECT candidate, check_name AS check,
                                              CASE WHEN ok THEN 'pass' ELSE 'FAIL' END AS result, detail
                                              FROM acceptance_checks {where} ORDER BY candidate, ordinal"""), w=24, h=10,
              overrides=[color_cells("result", {"pass": "#37872d", "FAIL": "#e02f44"})]),
        table(L, "Episode transitions (baseline → candidate)", sql(f"""SELECT candidate, trial, transition, episodes
                                                                       FROM transitions {where} ORDER BY candidate, trial, transition"""), w=12),
        table(L, "Pooled passes per task", sql(f"""SELECT candidate, task_id, baseline_passes, candidate_passes,
                                                   candidate_passes - baseline_passes AS change
                                                   FROM task_passes {where} ORDER BY candidate, task_id"""), w=12),
        table(L, "Runs in this experiment", sql(f"""SELECT x_label, role, candidate, run_id, max_active_episodes AS episode_cap,
                                                    max_inflight_llm_requests AS request_cap, window_s, window_cost, successes,
                                                    valid_episodes AS valid, cohort_size AS submitted, infra_errors
                                                    FROM experiment_trials {where} ORDER BY ordinal"""), w=24),
    ]
    variables = [pg_variable("experiment", "SELECT name FROM experiments ORDER BY kind, name", "Experiment / sweep")]
    return dashboard("rc-experiments", "Rollout Cost Lab · Experiments and sweeps", p, variables, time_from="now-30d",
                     tags=["history"], links=NAV_LINKS)


DASHBOARDS = {"live/live-run.json": live, "history/runs-overview.json": runs_overview,
              "history/run-detail.json": run_detail, "history/experiments.json": experiments}


def render() -> dict[str, str]:
    return {path: json.dumps(build(), indent=2) + "\n" for path, build in DASHBOARDS.items()}


def main() -> None:
    for path, content in render().items():
        target = OUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
