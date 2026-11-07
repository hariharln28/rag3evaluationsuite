# Products

Northstar ships three commercial products.

Pulse is a real-time monitoring and alerting product. It launched in March 2019. Pulse ingests metrics, logs, and traces, and evaluates alert rules with a default evaluation interval of 15 seconds.

Helix is a managed feature store. It launched in June 2021. Helix stores point-in-time correct training datasets and low-latency online features. Custom training-time transformers are available only on Enterprise.

Atlas is a semantic layer for metrics definitions. It launched in October 2023. Atlas lets teams declare canonical metrics once and query them from Pulse dashboards, notebooks, and the GraphQL API. Team plans may define up to 12 semantic domains; Enterprise has no published domain cap.

Pulse can run without Helix or Atlas. Helix requires Pulse for the built-in monitoring of feature freshness. Atlas requires Pulse, and can optionally read online features from Helix.
