# Monitoring

This directory contains time-series dashboards and alerting templates for the prediction oracle.

## Dashboards

- `dashboards/grafana_prediction_oracle.json`: Grafana dashboard for real-time views of model scores, signals, and fill health.
- `dashboards/superset_prediction_oracle.yaml`: Superset dashboard export covering the same metrics with simple time-series charts.

## Alerts

- `alerts/alert_rules.yaml`: Example Prometheus-style alert rules for score degradation, signal drops, and fill anomalies. Alerts are wired to webhook/Slack endpoints by default.

## Deployment notes

1. Import the desired dashboard into Grafana or Superset.
2. Update the data source names if your metrics are published under a different Prometheus job or Superset dataset.
3. Adjust alert thresholds to match production tolerances and point the webhook/Slack receivers at the correct URLs.
