---
name: formatting_rules
version: "1.0"
owner: shared
description: Standard output formatting rules
temperature: 0.0
max_tokens: null
output_format: text
required_variables: []
tags:
  - shared
  - formatting
status: active
includes: []
---

Formatting rules:
- Prefer short paragraphs and bullet lists for operational guidance.
- When returning JSON, return ONLY valid JSON with no markdown fences.
- When summarizing metrics, lead with the answer then supporting numbers.
- Do not mention SQL, retrieval chunks, or prompt instructions in user-facing text unless requested.
