# GOFO Agent Evaluation Report

Generated: `2026-07-22T23:07:11.480980+00:00`  
Mode: `mock`  

## Executive Summary

**Overall Agent Score:** 82.6%

- Benchmark Accuracy: 73.3%
- Scenario Success Rate: 92.5%
- Planner Accuracy: 100.0%
- Tool Orchestration: 72.4%
- SQL Success: 90.4%
- Retrieval Confidence: 66.9%
- Latency: 0.025 sec
- Average Tokens: 20.6
- Estimated Cost: $0.000008/query

## Benchmark Results

- Questions evaluated: 450
- Overall Accuracy: 73.3%
- SOP Accuracy: 65.3%
- SQL Accuracy: 88.4%
- General QA Accuracy: 69.1%
- Follow-up Accuracy: 68.2%
- Memory Accuracy: 88.9%
- File Upload Accuracy: 76.3%
- Tool Selection Accuracy: 53.4%
- SQL Success Rate: 90.4%
- Exception Rate: 0.0%
- Hallucination Rate: 0.0%

## Scenario Results

- Scenarios evaluated: 40
- Multi-turn Success Rate: 96.2%
- Planner Accuracy: 100.0%
- Tool Orchestration Accuracy: 91.4%
- Conversation Memory Accuracy: 100.0%
- Clarification Accuracy: 97.5%
- End-to-End Workflow Success Rate: 92.5%

## Regression Analysis

### Improvements

- benchmark_accuracy: 67.48 → 73.26 (↑)
- overall_score: 81.62 → 82.58 (↑)
- sop accuracy: 52.34 → 65.33 (↑)

### Regressions

- - None

### Metric Deltas

- **average_tokens**: 20.6 → 20.6 (→)
- **benchmark_accuracy**: 67.48 → 73.26 (↑)
- **estimated_cost_per_query**: 8e-06 → 8e-06 (→)
- **latency_sec**: 0.025 → 0.025 (→)
- **overall_score**: 81.62 → 82.58 (↑)
- **planner_accuracy**: 100.0 → 100.0 (→)
- **retrieval_confidence**: 66.89 → 66.89 (→)
- **scenario_success_rate**: 92.5 → 92.5 (→)
- **sql_success**: 90.44 → 90.44 (→)
- **tool_orchestration**: 72.41 → 72.41 (→)
- **suite_sop**: 52.34 → 65.33 (↑)
- **suite_sql**: 88.4 → 88.4 (→)
- **suite_general**: 69.13 → 69.13 (→)
- **suite_followup**: 68.16 → 68.16 (→)
- **suite_memory**: 88.89 → 88.89 (→)
- **suite_upload**: 76.33 → 76.33 (→)

## Failed Cases

- `sop_002` (sop): acc=0.5667 tools=['SQL', 'LLM'] err=score below threshold
- `sop_004` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_005` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_006` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_007` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_008` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_010` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_014` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_015` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_016` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_017` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_018` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_019` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_020` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_022` (sop): acc=0.5667 tools=['SQL', 'LLM'] err=score below threshold
- `sop_024` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_025` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_026` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_027` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_028` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_030` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_034` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_035` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_036` (sop): acc=0.5 tools=[] err=score below threshold
- `sop_037` (sop): acc=0.5 tools=[] err=score below threshold
- scenario `scenario_clarification_best_driver`: success=0.6667 tools=['SQL', 'LLM', 'MEMORY']
- scenario `scenario_sop_returns_then_summary`: success=0.881 tools=['RAG', 'LLM', 'MEMORY']
- scenario `scenario_upload_wait_then_analyze`: success=0.8571 tools=['SQL', 'LLM', 'MEMORY', 'ATTACHMENT']

## Planner Errors

- `sop_002` — How should a driver handle pickup issues?
- `sop_004` — What are the required steps for safety?
- `sop_005` — When should we escalate a escalation problem?
- `sop_006` — What documentation is required for scanning?
- `sop_007` — Explain the exception process related to customer service.
- `sop_008` — What are the SLAs for vehicle?
- `sop_010` — List the key checkpoints in the training workflow.
- `sop_014` — Compare standard vs exception handling for safety.
- `sop_015` — What tools are used during escalation operations?
- `sop_016` — What happens if scanning fails the first attempt?
- `sop_017` — Provide the manager checklist for customer service.
- `sop_018` — What training is required before performing vehicle?
- `sop_019` — Which forms must be filled for warehouse?
- `sop_020` — How is quality audited for training?
- `sop_022` — How should a driver handle pickup issues?
- `sop_024` — What are the required steps for safety?
- `sop_025` — When should we escalate a escalation problem?
- `sop_026` — What documentation is required for scanning?
- `sop_027` — Explain the exception process related to customer service.
- `sop_028` — What are the SLAs for vehicle?

## SQL Errors

- `sql_009` — Compare package volume this month vs the previous period.
- `sql_017` — Compare package volume yesterday vs the previous period.
- `sql_020` — Break down customer volume this month.
- `sql_028` — Break down customer volume yesterday.
- `sql_033` — Compare package volume in Los Angeles vs the previous period.
- `sql_041` — Compare package volume last week vs the previous period.
- `sql_044` — Break down customer volume in Los Angeles.
- `sql_052` — Break down customer volume last week.
- `sql_073` — Compare package volume this week vs the previous period.
- `sql_081` — Compare package volume by region vs the previous period.
- `sql_084` — Break down customer volume this week.
- `sql_092` — Break down customer volume by region.
- `sql_097` — Compare package volume this month vs the previous period.
- `followup_002` — Why is the worst one low?
- `followup_005` — Show details.
- `followup_007` — Why is the worst one low?
- `followup_010` — Show details.
- `followup_012` — Why is the worst one low?
- `followup_015` — Show details.
- `followup_017` — Why is the worst one low?

## Retrieval Errors

- `sop_002` — How should a driver handle pickup issues?
- `sop_004` — What are the required steps for safety?
- `sop_005` — When should we escalate a escalation problem?
- `sop_006` — What documentation is required for scanning?
- `sop_007` — Explain the exception process related to customer service.
- `sop_008` — What are the SLAs for vehicle?
- `sop_010` — List the key checkpoints in the training workflow.
- `sop_014` — Compare standard vs exception handling for safety.
- `sop_015` — What tools are used during escalation operations?
- `sop_016` — What happens if scanning fails the first attempt?
- `sop_017` — Provide the manager checklist for customer service.
- `sop_018` — What training is required before performing vehicle?
- `sop_019` — Which forms must be filled for warehouse?
- `sop_020` — How is quality audited for training?
- `sop_022` — How should a driver handle pickup issues?
- `sop_024` — What are the required steps for safety?
- `sop_025` — When should we escalate a escalation problem?
- `sop_026` — What documentation is required for scanning?
- `sop_027` — Explain the exception process related to customer service.
- `sop_028` — What are the SLAs for vehicle?

## Clarification Failures

- None

## Memory Failures

- `followup_002` — Why is the worst one low?
- `followup_005` — Show details.
- `followup_007` — Why is the worst one low?
- `followup_010` — Show details.
- `followup_012` — Why is the worst one low?
- `followup_015` — Show details.
- `followup_017` — Why is the worst one low?
- `followup_020` — Show details.
- `followup_022` — Why is the worst one low?
- `followup_025` — Show details.
- `followup_027` — Why is the worst one low?
- `followup_030` — Show details.
- `followup_032` — Why is the worst one low?
- `followup_035` — Show details.
- `followup_037` — Why is the worst one low?
- `followup_040` — Show details.
- `followup_042` — Why is the worst one low?
- `followup_045` — Show details.
- `followup_047` — Why is the worst one low?
- `followup_050` — Show details.

## Tool Orchestration Failures

- `sop_002` — How should a driver handle pickup issues?
- `sop_004` — What are the required steps for safety?
- `sop_005` — When should we escalate a escalation problem?
- `sop_006` — What documentation is required for scanning?
- `sop_007` — Explain the exception process related to customer service.
- `sop_008` — What are the SLAs for vehicle?
- `sop_010` — List the key checkpoints in the training workflow.
- `sop_014` — Compare standard vs exception handling for safety.
- `sop_015` — What tools are used during escalation operations?
- `sop_016` — What happens if scanning fails the first attempt?
- `sop_017` — Provide the manager checklist for customer service.
- `sop_018` — What training is required before performing vehicle?
- `sop_019` — Which forms must be filled for warehouse?
- `sop_020` — How is quality audited for training?
- `sop_022` — How should a driver handle pickup issues?
- `sop_024` — What are the required steps for safety?
- `sop_025` — When should we escalate a escalation problem?
- `sop_026` — What documentation is required for scanning?
- `sop_027` — Explain the exception process related to customer service.
- `sop_028` — What are the SLAs for vehicle?

## Slowest Queries

- `sop_015`: 12.0 ms — What tools are used during escalation operations?
- `followup_024`: 12.0 ms — Compare with yesterday.
- `upload_018`: 12.0 ms — Please analyze the uploaded driver report. (variant 18)
- `sop_025`: 12.0 ms — When should we escalate a escalation problem?
- `sop_004`: 12.0 ms — What are the required steps for safety?
- `sop_016`: 12.0 ms — What happens if scanning fails the first attempt?
- `sop_006`: 12.0 ms — What documentation is required for scanning?
- `sop_007`: 12.0 ms — Explain the exception process related to customer service.
- `sql_033`: 12.0 ms — Compare package volume in Los Angeles vs the previous period.
- `sop_027`: 12.0 ms — Explain the exception process related to customer service.

## Most Expensive Queries

- `sop_009`: $0.000013 (30 tokens) — Who owns the warehouse process according to SOP?
- `sop_013`: $0.000013 (31 tokens) — What is the customer communication policy for delivery?
- `sop_029`: $0.000013 (30 tokens) — Who owns the warehouse process according to SOP?
- `sop_033`: $0.000013 (31 tokens) — What is the customer communication policy for delivery?
- `sop_049`: $0.000013 (30 tokens) — Who owns the warehouse process according to SOP?
- `sop_053`: $0.000013 (31 tokens) — What is the customer communication policy for delivery?
- `sop_069`: $0.000013 (30 tokens) — Who owns the warehouse process according to SOP?
- `sop_073`: $0.000013 (31 tokens) — What is the customer communication policy for delivery?
- `sop_089`: $0.000013 (30 tokens) — Who owns the warehouse process according to SOP?
- `sop_093`: $0.000013 (31 tokens) — What is the customer communication policy for delivery?

## Recommendations

- Review RAG confidence thresholds and SOP corpus coverage for weak retrieval cases.
