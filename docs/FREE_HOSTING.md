# Free public Grafana hosting with Render and Neon

Your public URL stays the same when you import a new run. The Runs dashboard lists
every imported run; clicking a run opens its detail dashboard at the recorded time
window. Those individual run links differ, while the site and dashboard URLs remain
stable. Keep the same Render service and the provisioned dashboard UIDs.

This deployment hosts the three **history** dashboards (Runs, Run detail, Experiments
and sweeps). Run the GPU experiments separately in Colab. Importing their downloaded
archives requires no GPU, Docker or running Colab session.

## 1. Create two free Neon projects

Sign in at [Neon](https://console.neon.tech/) and use its **Free** plan. Create:

| Project | Purpose | Database |
| --- | --- | --- |
| `rollout-history` | Imported experiment results | Keep the default `neondb` |
| `grafana-state` | Grafana settings, users, sessions and encrypted datasource credentials | Keep the default `neondb` |

Use the same provider/nearby region for both. Keeping state separate prevents a
public history query from reading Grafana's own tables. Neon Free currently includes
1 GB and 100 CU-hours per month **per project**, with no time limit or credit card.
Leave scale-to-zero enabled and remain on the Free plan.

For each project, select **Connect** and turn **connection pooling off**. Record the
hostname, database, role and password. Keep credentials in a password manager.
The hostname looks like `ep-example-123.us-east-2.aws.neon.tech`.

For `rollout-history`, also copy the complete owner connection string (beginning
`postgresql://` and including `sslmode=require`) for the local import command below.
Do not publish that string or put it in GitHub.

## 2. Create the history reader through SQL

Open the **SQL Editor in `rollout-history`**, using `neondb` and its owner role.
Choose a new, unique password (at least 24 random characters; using letters and
digits makes pasting into SQL easier). Replace the password below and run:

```sql
CREATE ROLE grafana_reader WITH LOGIN PASSWORD 'REPLACE_WITH_A_RANDOM_PASSWORD'
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
GRANT CONNECT ON DATABASE neondb TO grafana_reader;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO grafana_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO grafana_reader;
ALTER ROLE grafana_reader SET statement_timeout = '15s';
```

Run this once, using the **same owner role that will perform imports**; default
privileges apply to tables that role creates. To change an existing reader's
password later, use `ALTER ROLE grafana_reader PASSWORD 'NEW_RANDOM_PASSWORD';`.

Create this role through SQL, **not Neon's Add role button**: roles made through its
console/API inherit administrative privileges. Never use the history owner for the
public Grafana datasource. The state project uses its own owner, separately.

## 3. Import your downloaded GPU results from Windows

Open PowerShell in the repository. Update the archive filename to your latest
download. Extract it at the repository root so experiment paths resolve correctly.

```powershell
cd C:\Users\arnab\Documents\GPU_Cost_Optimization
uv sync --extra dashboard --locked

$archivePath = Join-Path $HOME 'Downloads\notebook-results.tgz'
tar -xzf $archivePath --exclude 'configs' --exclude 'uv.lock'
if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed' }

# Paste the rollout-history OWNER connection string at the hidden prompt.
# It is not written into PowerShell's command history.
$secretUrl = Read-Host 'rollout-history owner connection string' -AsSecureString
$env:RC_DB_URL = [System.Net.NetworkCredential]::new('', $secretUrl).Password
try {
    uv run --extra dashboard --locked rollout-cost export
    if ($LASTEXITCODE -ne 0) { throw 'Database export failed' }
} finally {
    Remove-Item Env:RC_DB_URL -ErrorAction SilentlyContinue
    Remove-Variable secretUrl -ErrorAction SilentlyContinue
}
```

The exporter loads run events, outcomes, GPU samples and experiment metadata into
PostgreSQL; Grafana does not consume `report.md` directly. Check that it reports
at least one exported run and inspect any skipped-artifact warnings. This command
uses all local runs/experiments. If you only want selected results public, use
`--runs-dir` and `--experiments-dir` to point to a separate collection while keeping
its experiment references valid.

Keep your archives as backups. Re-exporting the same run updates its rows without
duplicating them; importing new run IDs keeps existing runs. Do not use `--reset`.
Avoid overlapping imports from two processes.

## 4. Deploy Grafana on Render Free

Make sure `render.yaml` and `observability/render/` are pushed to your GitHub branch.
Sign in at [Render](https://dashboard.render.com/) with GitHub, then:

1. Select **New > Blueprint** and connect `arnabdas1999/GPU-Cost-Optimization`.
2. Choose the branch containing `render.yaml` (normally `main`). Render reads the
   blueprint from the repository root.
3. Confirm there is **one web service with plan Free**, with **no Render database
   or persistent disk**. No GPU or runner is deployed.
4. Supply the prompted environment variables using the table below.
5. Deploy the Blueprint. Wait until the service is healthy; initial database
   migrations and the Docker image build can take several minutes.

| Render variable | Value |
| --- | --- |
| `GF_DATABASE_HOST` | **grafana-state** hostname followed by `:5432` |
| `GF_DATABASE_USER` | **grafana-state** owner role, usually `neondb_owner` |
| `GF_DATABASE_PASSWORD` | **grafana-state** owner password |
| `RC_HISTORY_HOST` | **rollout-history** hostname followed by `:5432` |
| `RC_HISTORY_PASSWORD` | Password you chose for **grafana_reader** |

The Blueprint sets both database names to `neondb` and history user to
`grafana_reader`. Change `GF_DATABASE_NAME` or `RC_HISTORY_DATABASE` in Render if you
used different names. Supply raw passwords, not URI-encoded values. Put only
`hostname:5432` in the host fields, with no `postgresql://` or query string.

Render generates `GF_SECURITY_ADMIN_PASSWORD` and `GF_SECURITY_SECRET_KEY` and keeps
them as service environment variables. Copy the generated admin password from
**Service > Environment** to your password manager. Keep the secret key stable:
Grafana uses it to encrypt credentials in its state database.

The image configures HTTPS-aware links from Render's assigned external URL and
listens on its `PORT`. Both PostgreSQL connections use TLS with certificate
verification. History queries use the read-only account. Dashboards are provisioned
from Git, and no live Prometheus datasource is needed.

## 5. Verify and share the stable link

Render displays a URL similar to:

```text
https://rollout-cost-dashboard.onrender.com
```

The exact subdomain depends on availability. Use the URL Render actually assigned.
Open it in an incognito window without signing in. You should see **Runs**. Click
the pilot's run ID, verify its outcomes and GPU graphs, and try the run selector.
For admin access, open `/login` and use `admin` plus the generated password.

Share the **base URL** to let visitors browse every imported run. Copy a run's
detail link when you want to share that specific result. Anonymous users have
Viewer access; anyone with the URL can read all history tables made available to
the datasource. Import only results intended to be public.

Changing run data does **not** require a Render deploy. Restarting/redeploying the
same service keeps its URL, and the two Neon projects retain results and Grafana
state. Provisioned dashboards should be edited in the repository and redeployed.
Admin changes to those provisioned dashboards are overwritten by provisioning.

## 6. Publish each new GPU run

1. In Colab, download the updated archive after your run:

   ```python
   from google.colab import files
   files.download(str(archive_results(ROOT)))
   ```

2. On Windows, repeat **step 3** with the new archive and the **same rollout-history
   owner connection string**. If the browser added `(1)` or a timestamp to its
   filename, update `$archivePath`.
3. Refresh the existing Grafana site. The new run appears in Runs; clicking it
   selects the correct historical time window. Existing run links keep working.

There is no automatic upload from Colab in this setup. Downloading alone does not
update the hosted dashboard; the database export is the publishing step. You do
not need another Neon project, Render service or deployment for a new report.

## Free-tier behavior and troubleshooting

- Render sleeps after **15 minutes without inbound traffic**. Opening the same
  link wakes it; the first load can take about a minute. Do not add keepalive
  traffic to defeat sleeping. The free service is for a small demo, not a service
  with an uptime guarantee.
- Render provides **750 free instance hours per workspace per calendar month**,
  shared by its free web services. Stay on Free; without a payment method,
  bandwidth/build overages suspend services or builds instead of billing.
- Neon has monthly compute/storage limits. Leave dashboard auto-refresh off, and
  close unused tabs so idle compute can sleep. Watch each project's usage. Keep
  archive backups in case you reach a quota.
- **Do not use Render Free Postgres here**: it expires after 30 days. Neon Free
  currently has no fixed time limit.
- **No data / permission denied:** check that you imported into `rollout-history`,
  that `RC_HISTORY_*` points there, and that step 2 was run as the importer owner.
  After changing the owner, rerun the `GRANT SELECT` and `ALTER DEFAULT PRIVILEGES`
  statements as that owner.
- **Database connection error:** check the project/hostname/password mapping,
  `:5432`, database names and Neon quota. Use direct endpoints, not pooling.
- **Run-detail graphs empty:** open the run from Runs to select its historical
  time window. The default last-30-days range may exclude an older run.
- **Container runs out of memory:** the profile disables alerting, Grafana Live
  and automatic plugin installation and limits database connections. Free Render
  still has only 512 MB; this is a low-traffic deployment, not a capacity guarantee.
- Keep the Grafana image patched; the public profile pins the current 13.2.4 OSS
  release rather than exposing the older local-development image.

Provider terms and quotas can change. References checked October 26, 2026:
[Render Free](https://render.com/docs/free),
[Render Blueprints](https://render.com/docs/infrastructure-as-code),
[Neon pricing](https://neon.com/pricing),
[Neon role privileges](https://neon.com/docs/manage/roles),
[Grafana PostgreSQL permissions](https://grafana.com/docs/grafana/latest/datasources/postgres/configure/),
[Grafana downloads](https://grafana.com/grafana/download?edition=oss).
