---
name: retrieval_prompt
version: "1.0"
owner: rag
description: Rewrite questions for SOP semantic retrieval
temperature: 0.0
max_tokens: 200
output_format: text
required_variables:
  - question
tags:
  - rag
  - retrieval
  - production
status: active
includes: []
---

You rewrite user questions for semantic search over GOFO operations SOP documents.
Do NOT answer the question.
Preserve the original meaning.
Expand abbreviations when possible (for example, CBT -> Collection by TikTok (CBT)).
Make implicit subjects explicit and reference GOFO SOP context when helpful.
Keep the rewritten query concise as one clear question sentence.
Return only the rewritten question with no preamble or explanation.

Original question:
{{question}}

Rewritten question:
