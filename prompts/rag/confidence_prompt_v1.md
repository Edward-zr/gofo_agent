---
name: confidence_prompt
version: "1.0"
owner: rag
description: Optional LLM assist for retrieval confidence rationale
temperature: 0.0
max_tokens: 400
output_format: json
required_variables:
  - question
  - similarity_scores
tags:
  - rag
  - confidence
status: active
includes: []
---

Evaluate whether retrieved evidence is sufficient for a grounded SOP answer.
Return ONLY JSON:
{"confidence_level":"HIGH|MEDIUM|LOW","reason":"...","should_clarify":false}

Question:
{{question}}

Similarity scores:
{{similarity_scores}}

Sources:
{{retrieved_sources}}

Chunk count:
{{retrieved_chunk_count}}
