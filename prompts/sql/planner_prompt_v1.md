---
name: planner_prompt
version: "1.0"
owner: sql
description: SQL strategy notes before generation
temperature: 0.0
max_tokens: 1200
output_format: text
required_variables:
  - question
tags:
  - sql
  - strategy
status: active
includes: []
---

Produce a brief SQL strategy for the question (not SQL itself):
- target metric
- grain (driver/hub/day)
- filters / date range
- joins needed

Question:
{{question}}

Schema summary:
{{schema}}
