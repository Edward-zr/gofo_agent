---
name: validator_prompt
version: "1.0"
owner: sql
description: Validate SQL against schema allowlist (LLM assist)
temperature: 0.0
max_tokens: 1200
output_format: json
required_variables:
  - sql
  - schema
tags:
  - sql
  - validation
status: active
includes: []
---

Validate the SQL against the schema allowlist.
Return ONLY JSON:
{"valid": true, "issues": [], "rewritten_sql": null}

SQL:
{{sql}}

Schema:
{{schema}}
