-- Rollout Cost Lab historical store for Grafana. Version: contract.EXPORT_SCHEMA_VERSION.
-- Created by `rollout-cost export`; rows are replaced per run / experiment on every export.

CREATE TABLE IF NOT EXISTS schema_version (version integer NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
    run_id text PRIMARY KEY,
    label text,
    config_name text,
    split text,
    synthetic boolean NOT NULL,
    synthetic_reason text,
    status text,
    format_version integer,
    created_utc timestamptz,
    start_at timestamptz,          -- run start (server preparation)
    window_start timestamptz,      -- first admission
    window_end timestamptz,        -- last terminal episode
    end_at timestamptz,            -- drain finished
    start_ms bigint,               -- epoch ms of start_at / end_at, for dashboard links
    end_ms bigint,
    model text,
    model_revision text,
    gpu text,
    executor text,
    cache_reset text,
    max_active_episodes integer,
    max_inflight_llm_requests integer,
    tool_workers integer,
    tasks integer,
    repetitions integer,
    cohort_size integer,
    valid_episodes integer,
    successes integer,
    failed_after_finish integer,
    failed_budget_exhausted integer,
    infra_errors integer,
    wall_timeouts integer,
    window_s double precision,
    run_inclusive_s double precision,
    overhead_s double precision,
    hourly_rate double precision,
    currency text,
    window_cost double precision,
    run_inclusive_cost double precision,
    cost_per_valid double precision,
    cost_per_success double precision,
    steady_state_fraction double precision,
    tool_time_fraction double precision,
    protocol_error_rate double precision,
    prompt_tokens bigint,
    completion_tokens bigint,
    prefix_cache_hit_rate double precision,
    cancelled_requests integer,
    drain_verified boolean,
    warnings_count integer,
    config_fingerprint text,
    protected_fingerprint text
);

CREATE TABLE IF NOT EXISTS run_warnings (
    run_id text NOT NULL,
    ordinal integer NOT NULL,
    message text NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    run_id text NOT NULL,
    episode_id text NOT NULL,
    idx integer,
    task_id text,
    family text,
    rep integer,
    category text,
    stop_reason text,
    stop_detail text,
    infrastructure_status text,
    infrastructure_detail text,
    grader_outcome text,
    grader_detail text,
    steps integer,
    protocol_errors integer,
    tool_errors integer,
    prompt_tokens bigint,
    completion_tokens bigint,
    llm_wait_s double precision,
    request_s double precision,
    tool_wait_s double precision,
    tool_s double precision,
    file_op_s double precision,
    grading_s double precision,
    service_s double precision,
    latency_s double precision,
    admission_wait_s double precision,
    admitted_at timestamptz,
    agent_end_at timestamptz,
    graded_at timestamptz
);

CREATE TABLE IF NOT EXISTS spans (
    run_id text NOT NULL,
    episode_id text NOT NULL,
    kind text NOT NULL,
    start_at timestamptz NOT NULL,
    end_at timestamptz NOT NULL,
    duration_s double precision NOT NULL,
    request_id text,
    detail text
);

CREATE TABLE IF NOT EXISTS episode_states (
    run_id text NOT NULL,
    episode_id text NOT NULL,
    ts timestamptz NOT NULL,
    state integer NOT NULL
);

CREATE TABLE IF NOT EXISTS concurrency (
    run_id text NOT NULL,
    ts timestamptz NOT NULL,
    series text NOT NULL,
    value double precision NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    run_id text NOT NULL,
    episode_id text NOT NULL,
    request_id text NOT NULL,
    step integer,
    status text,
    duration_s double precision,
    wait_s double precision,
    ttft_s double precision,
    prompt_tokens bigint,
    completion_tokens bigint,
    finish_reason text,
    attempts integer
);

CREATE TABLE IF NOT EXISTS gpu_samples (
    run_id text NOT NULL,
    ts timestamptz NOT NULL,
    gpu integer,
    utilization double precision,
    memory_used double precision,
    memory_total double precision,
    power_draw double precision
);

CREATE TABLE IF NOT EXISTS server_samples (
    run_id text NOT NULL,
    ts timestamptz NOT NULL,
    signal text NOT NULL,
    value double precision NOT NULL
);

CREATE TABLE IF NOT EXISTS attribution (
    run_id text NOT NULL,
    component text NOT NULL,
    share double precision
);

CREATE TABLE IF NOT EXISTS latency (
    run_id text NOT NULL,
    metric text NOT NULL,
    n integer,
    p50 double precision,
    p90 double precision,
    p99 double precision,
    max double precision
);

CREATE TABLE IF NOT EXISTS experiments (
    name text PRIMARY KEY,
    kind text NOT NULL,            -- paired | sweep
    split text,
    status text,
    synthetic boolean NOT NULL,
    verdict text,
    accepted boolean,
    aa_calibration boolean,
    trials integer,
    window_spread_pct double precision,
    chosen text,
    source_path text
);

CREATE TABLE IF NOT EXISTS experiment_trials (
    experiment text NOT NULL,
    ordinal integer NOT NULL,
    x_label text NOT NULL,         -- "trial 1" (paired) or the variant name (sweep)
    role text NOT NULL,            -- baseline | candidate
    candidate text,                -- the candidate this row belongs to ("candidate" or a variant name)
    run_id text NOT NULL,
    max_active_episodes integer,
    max_inflight_llm_requests integer,
    window_s double precision,
    window_cost double precision,
    successes integer,
    valid_episodes integer,
    cohort_size integer,
    infra_errors integer
);

CREATE TABLE IF NOT EXISTS acceptance_checks (
    experiment text NOT NULL,
    candidate text NOT NULL,
    ordinal integer NOT NULL,
    check_name text NOT NULL,
    ok boolean NOT NULL,
    detail text
);

CREATE TABLE IF NOT EXISTS transitions (
    experiment text NOT NULL,
    candidate text NOT NULL,
    trial integer NOT NULL,
    transition text NOT NULL,
    episodes integer NOT NULL
);

CREATE TABLE IF NOT EXISTS task_passes (
    experiment text NOT NULL,
    candidate text NOT NULL,
    task_id text NOT NULL,
    baseline_passes integer NOT NULL,
    candidate_passes integer NOT NULL
);

CREATE INDEX IF NOT EXISTS episodes_run ON episodes (run_id);
CREATE INDEX IF NOT EXISTS spans_run ON spans (run_id);
CREATE INDEX IF NOT EXISTS states_run ON episode_states (run_id, ts);
CREATE INDEX IF NOT EXISTS concurrency_run ON concurrency (run_id, ts);
CREATE INDEX IF NOT EXISTS requests_run ON requests (run_id);
CREATE INDEX IF NOT EXISTS gpu_run ON gpu_samples (run_id, ts);
CREATE INDEX IF NOT EXISTS server_run ON server_samples (run_id, ts);
