---
name: clarification_prompt
version: "1.0"
owner: clarification
description: Generate concise clarification questions with options
temperature: 0.0
max_tokens: 400
output_format: text
required_variables:
  - question
  - missing_fields
tags:
  - clarification
  - production
status: active
includes:
  - shared.formatting_rules
---

Generate ONE concise clarification question for a GOFO operations user.
Ask only for missing business parameters.
Prefer multiple-choice options when possible.
Do not answer the original question.

Original question:
{{question}}

Missing fields:
{{missing_fields}}

Ambiguity type:
{{ambiguity_type}}

Reason:
{{reason}}

Known options (if any):
{{options}}
