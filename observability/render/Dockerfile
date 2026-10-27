# Current OSS release, checked October 26, 2026. Keep this public image patched.
FROM grafana/grafana:13.2.4

COPY observability/render/grafana.ini /etc/grafana/rollout-public.ini
COPY observability/render/provisioning/ /etc/grafana/provisioning/
COPY observability/grafana/dashboards/history/ /etc/rollout-cost/dashboards/history/
COPY --chmod=755 observability/render/start.sh /start-rollout-grafana.sh

ENV GF_PATHS_CONFIG=/etc/grafana/rollout-public.ini
EXPOSE 10000
ENTRYPOINT ["/start-rollout-grafana.sh"]
