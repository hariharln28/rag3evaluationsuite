# Architecture

Ingestion uses Apache Kafka. Stream processors written in Go write compressed columns into ClickHouse for analytics and into a Redis-backed online store for Pulse alerts.

The control plane is written in Go and exposed behind Envoy. The query path has an internal objective of 140 milliseconds at p99 for cached Pulse dashboard queries.

Helix online features are served from a separately sharded Redis cluster. Offline feature snapshots land in customer-owned S3 or GCS buckets, or in Northstar-managed object storage on Starter and Team.

Vector search for Atlas metric synonym matching uses pgvector on PostgreSQL 16. Northstar does not use Pinecone in production.

Batch jobs run on Kubernetes in each residency region. Customers cannot deploy the control plane into their own Kubernetes clusters; Northstar is a multi-tenant SaaS only.
