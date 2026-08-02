# Prompt Experiment: planner.planner_prompt

Experiment ID: `exp_planner_prompt`  
Mode: `mock`  
Generated: `2026-07-22T23:48:34.783293+00:00`  

**Recommended version:** `v1`  
Reason: Selected v1 using priorities: accuracy, sql_success, scenario_success, latency, cost

## Version Comparison

### v1

- Accuracy: 65.33%
- Planner Accuracy: 100.0%
- Tool Selection: 58.09%
- SQL Success: 100.0%
- Retrieval Accuracy: 25.5%
- Scenario Success: 70.0%
- Latency: 0.027 sec
- Avg Tokens: 21.8
- Cost: $8e-06/query
- Overall Score: 69.82%

### v2

- Accuracy: 65.33%
- Planner Accuracy: 100.0%
- Tool Selection: 58.09%
- SQL Success: 100.0%
- Retrieval Accuracy: 25.5%
- Scenario Success: 70.0%
- Latency: 0.027 sec
- Avg Tokens: 21.8
- Cost: $8e-06/query
- Overall Score: 69.82%

## Priorities

1. accuracy
2. sql_success
3. scenario_success
4. latency
5. cost

## Next step

Activate the recommended version without code changes:

```bash
python -m evaluation.prompt_experiments --activate planner.planner_prompt:v1
```
