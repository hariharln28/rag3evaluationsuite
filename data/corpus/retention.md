# Data retention

Raw event retention depends on plan. Starter retains raw events for 90 days. Team retains raw events for 13 months. Enterprise retains raw events for 36 months.

Aggregated Pulse metrics at 1-minute granularity are retained for 24 months on every paid plan.

Helix offline feature snapshots are retained for 24 months on Team and Enterprise. Starter does not include Helix.

Soft-deleted workspaces remain recoverable for 30 days, after which object storage and ClickHouse partitions are purged. Legal holds can extend retention on Enterprise only, in 12-month increments.
