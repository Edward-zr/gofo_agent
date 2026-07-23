---
name: business_rules
version: "1.0"
owner: shared
description: GOFO terminology and operational policies
temperature: 0.0
max_tokens: null
output_format: text
required_variables: []
tags:
  - shared
  - business
status: active
includes: []
---

GOFO business rules:
- Hub / warehouse refer to operational stations (drivers.hub in analytics DB).
- Pickup statuses: Completed, Delayed, Failed.
- Common KPIs: pickup rate, completed pickups, on-time rate, failure rate, package volume.
- Drivers belong to hubs; hubs belong to regions.
- Prefer explicit date ranges (today, this week, this month) for analytics questions.
