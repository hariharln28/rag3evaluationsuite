# API and rate limits

Northstar provides a REST API and a GraphQL API. Both require an API key. Live keys are prefixed `nsk_` and test keys are prefixed `nsk_test_`.

Rate limits are applied per workspace, not per key. Starter is limited to 60 requests per minute. Team is limited to 300 requests per minute. Enterprise is limited to 2,000 requests per minute. Burst is 2x the steady limit for up to 10 seconds.

Webhooks can be registered for alert fires, feature freshness drops, and incident status changes. Webhook endpoints must respond with HTTP 2xx within 5 seconds or the delivery is retried with exponential backoff for 24 hours.

GraphQL introspection is disabled on production keys. The maximum GraphQL query depth is 8.
