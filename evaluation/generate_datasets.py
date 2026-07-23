"""Generate benchmark + scenario JSON datasets at the target scale.

Run:
  python -m evaluation --generate-datasets
  # or
  python -m evaluation.generate_datasets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASETS = Path(__file__).resolve().parent / "datasets"

SOP_TOPICS = [
    ("returns", ["return", "30 days", "refund", "customer"]),
    ("pickup", ["pickup", "driver", "hub", "scan"]),
    ("delivery", ["delivery", "attempt", "exception", "proof"]),
    ("safety", ["safety", "PPE", "incident", "report"]),
    ("escalation", ["escalate", "manager", "SLA", "priority"]),
    ("scanning", ["scan", "barcode", "package", "manifest"]),
    ("customer_service", ["customer", "complaint", "resolution", "contact"]),
    ("vehicle", ["vehicle", "inspection", "fuel", "maintenance"]),
    ("warehouse", ["warehouse", "sortation", "staging", "dock"]),
    ("training", ["training", "onboarding", "certification", "SOP"]),
]

SOP_TEMPLATES = [
    "What is the SOP for {topic}?",
    "How should a driver handle {topic} issues?",
    "Summarize the {topic} policy.",
    "What are the required steps for {topic}?",
    "When should we escalate a {topic} problem?",
    "What documentation is required for {topic}?",
    "Explain the exception process related to {topic}.",
    "What are the SLAs for {topic}?",
    "Who owns the {topic} process according to SOP?",
    "List the key checkpoints in the {topic} workflow.",
    "What safety rules apply during {topic}?",
    "How do I verify completion for {topic}?",
    "What is the customer communication policy for {topic}?",
    "Compare standard vs exception handling for {topic}.",
    "What tools are used during {topic} operations?",
    "What happens if {topic} fails the first attempt?",
    "Provide the manager checklist for {topic}.",
    "What training is required before performing {topic}?",
    "Which forms must be filled for {topic}?",
    "How is quality audited for {topic}?",
]

SQL_METRICS = [
    ("pickup rate", ["pickup", "rate"]),
    ("completed pickups", ["completed", "pickup"]),
    ("on-time rate", ["on-time", "rate"]),
    ("failure rate", ["fail", "rate"]),
    ("package volume", ["volume", "package"]),
    ("driver performance", ["driver", "performance"]),
    ("hub performance", ["hub", "performance"]),
    ("customer volume", ["customer", "volume"]),
]

SQL_SCOPES = [
    "today",
    "yesterday",
    "this week",
    "last week",
    "this month",
    "in Chicago",
    "in Los Angeles",
    "for Atlanta Hub",
    "by driver",
    "by hub",
    "by region",
]

SQL_TEMPLATES = [
    "Show {metric} {scope}.",
    "What is the {metric} {scope}?",
    "Rank hubs by {metric} {scope}.",
    "Rank drivers by {metric} {scope}.",
    "Compare {metric} {scope} vs the previous period.",
    "Which hub has the worst {metric} {scope}?",
    "Which driver has the best {metric} {scope}?",
    "Break down {metric} {scope}.",
]

GENERAL_TEMPLATES = [
    ("Hello", "greeting", ["hello", "gofo"], ["LLM"]),
    ("Hi there", "greeting", ["hello", "gofo"], ["LLM"]),
    ("What can you help me with?", "capabilities", ["pickup", "sop", "analyze"], ["LLM"]),
    ("Who are you?", "identity", ["gofo", "operations"], ["LLM"]),
    ("Thanks", "chitchat", ["welcome", "help"], ["LLM"]),
    ("What is last-mile logistics?", "general_knowledge", ["delivery", "logistics"], ["LLM"]),
    ("Explain pickup rate in logistics", "general_knowledge", ["pickup", "rate"], ["LLM"]),
    ("How do hubs and drivers relate?", "general_knowledge", ["hub", "driver"], ["LLM"]),
    ("Write a Python snippet to compute a rate", "coding", ["python", "rate"], ["LLM"]),
    ("Help me draft an ops standup agenda", "general_knowledge", ["agenda", "ops"], ["LLM"]),
]

FOLLOWUP_PAIRS = [
    (
        "Show today's pickup performance.",
        "Show only Chicago.",
        ["chicago", "pickup"],
        ["SQL", "MEMORY", "LLM"],
        "follow_up_filter",
    ),
    (
        "Rank hubs by pickup rate this week.",
        "Why is the worst one low?",
        ["hub", "rate"],
        ["SQL", "MEMORY", "LLM"],
        "follow_up_explain",
    ),
    (
        "Show top drivers by completed pickups.",
        "Which hub does the top driver belong to?",
        ["hub", "driver"],
        ["SQL", "MEMORY", "KNOWLEDGE_GRAPH", "LLM"],
        "follow_up_entity",
    ),
    (
        "Show Los Angeles Hub performance today.",
        "Compare with yesterday.",
        ["compare", "yesterday"],
        ["SQL", "MEMORY", "LLM"],
        "follow_up_compare",
    ),
    (
        "Show pickup failures this week.",
        "Show details.",
        ["fail", "detail"],
        ["SQL", "MEMORY", "LLM"],
        "follow_up_details",
    ),
]

MEMORY_PROMPTS = [
    (
        "Remember we are analyzing Chicago hubs.",
        "What region are we focused on?",
        ["chicago"],
        ["MEMORY", "LLM"],
        "memory_recall_region",
    ),
    (
        "Use pickup rate as the metric.",
        "Rank drivers with the current metric.",
        ["pickup", "rate", "driver"],
        ["SQL", "MEMORY", "LLM"],
        "memory_metric_carry",
    ),
    (
        "Focus on this week.",
        "Show hub performance for the current date range.",
        ["week", "hub"],
        ["SQL", "MEMORY", "LLM"],
        "memory_date_carry",
    ),
]

UPLOAD_PROMPTS = [
    ("I want to analyze an Excel file of pickups.", ["upload", "file", "excel"], ["ATTACHMENT", "LLM"]),
    ("Can you read my CSV attachment?", ["upload", "csv", "attach"], ["ATTACHMENT", "LLM"]),
    ("Please analyze the uploaded driver report.", ["upload", "analyze", "driver"], ["ATTACHMENT", "LLM"]),
    ("Chart the attached hub performance spreadsheet.", ["chart", "attach", "hub"], ["ATTACHMENT", "VISUALIZATION", "LLM"]),
    ("Summarize the PDF I uploaded.", ["upload", "pdf", "summar"], ["ATTACHMENT", "LLM"]),
]


def _write(name: str, payload: Any) -> Path:
    DATASETS.mkdir(parents=True, exist_ok=True)
    path = DATASETS / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def build_sop_questions(target: int = 200) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    n = 0
    while len(items) < target:
        topic, keywords = SOP_TOPICS[n % len(SOP_TOPICS)]
        template = SOP_TEMPLATES[n % len(SOP_TEMPLATES)]
        question = template.format(topic=topic.replace("_", " "))
        n += 1
        items.append(
            {
                "id": f"sop_{len(items) + 1:03d}",
                "question": question,
                "expected_behavior": "rag_sop_answer",
                "expected_tools": ["RAG", "LLM"],
                "expected_answer_keywords": [topic.replace("_", " ").split()[0], keywords[0]],
                "notes": f"SOP topic={topic}",
            }
        )
    return items


def build_sql_questions(target: int = 100) -> list[dict[str, Any]]:
    clarifications = [
        ("Show the best driver.", ["metric"], "clarification_metric"),
        ("Show performance.", ["hub", "driver", "region"], "clarification_scope"),
        ("Compare pickup rates.", ["today", "week", "month"], "clarification_periods"),
        ("Show top customers.", ["volume", "revenue", "pickup"], "clarification_customer_metric"),
    ]
    items: list[dict[str, Any]] = []
    for question, keywords, behavior in clarifications:
        items.append(
            {
                "id": f"sql_{len(items) + 1:03d}",
                "question": question,
                "expected_behavior": behavior,
                "expected_tools": [],
                "expected_answer_keywords": keywords,
                "notes": "Ambiguous prompt should clarify before tools when Clarification Manager is enabled.",
            }
        )
    n = 0
    while len(items) < target:
        metric, keywords = SQL_METRICS[n % len(SQL_METRICS)]
        scope = SQL_SCOPES[n % len(SQL_SCOPES)]
        template = SQL_TEMPLATES[n % len(SQL_TEMPLATES)]
        question = template.format(metric=metric, scope=scope)
        n += 1
        items.append(
            {
                "id": f"sql_{len(items) + 1:03d}",
                "question": question,
                "expected_behavior": "sql_analytics",
                "expected_tools": ["SQL", "LLM"],
                "expected_answer_keywords": keywords,
                "notes": f"metric={metric}; scope={scope}",
            }
        )
    return items[:target]


def build_general_questions(target: int = 50) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    base = list(GENERAL_TEMPLATES)
    extras = [
        (f"Tell me about operational KPI number {i}", "general_knowledge", ["kpi", "operations"], ["LLM"])
        for i in range(1, 30)
    ]
    for question, behavior, keywords, tools in base + extras:
        if len(items) >= target:
            break
        items.append(
            {
                "id": f"general_{len(items) + 1:03d}",
                "question": question,
                "expected_behavior": behavior,
                "expected_tools": tools,
                "expected_answer_keywords": keywords,
                "notes": "",
            }
        )
    while len(items) < target:
        i = len(items) + 1
        items.append(
            {
                "id": f"general_{i:03d}",
                "question": f"Give a brief tip for improving hub productivity #{i}.",
                "expected_behavior": "general_knowledge",
                "expected_tools": ["LLM"],
                "expected_answer_keywords": ["hub", "productiv"],
                "notes": "Synthetic general prompt",
            }
        )
    return items[:target]


def build_followup_questions(target: int = 50) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    n = 0
    while len(items) < target:
        prior, follow, keywords, tools, behavior = FOLLOWUP_PAIRS[n % len(FOLLOWUP_PAIRS)]
        n += 1
        items.append(
            {
                "id": f"followup_{len(items) + 1:03d}",
                "question": follow,
                "expected_behavior": behavior,
                "expected_tools": tools,
                "expected_answer_keywords": keywords,
                "notes": f"Assumes prior context: {prior}",
                "prior_question": prior,
            }
        )
    return items


def build_memory_questions(target: int = 30) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    n = 0
    while len(items) < target:
        setup, ask, keywords, tools, behavior = MEMORY_PROMPTS[n % len(MEMORY_PROMPTS)]
        n += 1
        items.append(
            {
                "id": f"memory_{len(items) + 1:03d}",
                "question": ask,
                "expected_behavior": behavior,
                "expected_tools": tools,
                "expected_answer_keywords": keywords,
                "notes": f"Setup turn: {setup}",
                "setup_question": setup,
            }
        )
    return items


def build_upload_questions(target: int = 20) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    n = 0
    while len(items) < target:
        question, keywords, tools = UPLOAD_PROMPTS[n % len(UPLOAD_PROMPTS)]
        n += 1
        items.append(
            {
                "id": f"upload_{len(items) + 1:03d}",
                "question": question if n <= len(UPLOAD_PROMPTS) else f"{question} (variant {n})",
                "expected_behavior": "upload_or_attachment_handling",
                "expected_tools": tools,
                "expected_answer_keywords": keywords,
                "notes": "May wait for upload if no attachment_ids provided.",
            }
        )
    return items


def build_scenarios(target: int = 40) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []

    scenarios.append(
        {
            "scenario_id": "scenario_pickup_chicago_decline_chart",
            "description": "Pickup performance drill-down with compare, explain, chart, recommend.",
            "conversation": [
                {
                    "user": "Show today's pickup performance.",
                    "expected_behavior": "sql_analytics",
                    "expected_tools": ["SQL", "LLM"],
                    "expected_answer_keywords": ["pickup"],
                },
                {
                    "user": "Show only Chicago.",
                    "expected_behavior": "follow_up_filter",
                    "expected_tools": ["SQL", "MEMORY", "LLM"],
                    "expected_answer_keywords": ["chicago"],
                },
                {
                    "user": "Compare with yesterday.",
                    "expected_behavior": "follow_up_compare",
                    "expected_tools": ["SQL", "MEMORY", "LLM"],
                    "expected_answer_keywords": ["yesterday", "compare"],
                },
                {
                    "user": "Explain the biggest decline.",
                    "expected_behavior": "follow_up_explain",
                    "expected_tools": ["SQL", "MEMORY", "LLM"],
                    "expected_answer_keywords": ["decline"],
                },
                {
                    "user": "Generate a bar chart.",
                    "expected_behavior": "visualization",
                    "expected_tools": ["SQL", "VISUALIZATION", "LLM"],
                    "expected_answer_keywords": ["chart"],
                },
                {
                    "user": "Recommend operational improvements.",
                    "expected_behavior": "recommendation",
                    "expected_tools": ["SQL", "RECOMMENDATION", "LLM"],
                    "expected_answer_keywords": ["recommend"],
                },
            ],
            "expected_behaviors": [
                "sql_analytics",
                "memory",
                "visualization",
                "recommendation",
            ],
            "expected_tools": [
                "SQL",
                "MEMORY",
                "VISUALIZATION",
                "RECOMMENDATION",
                "LLM",
            ],
        }
    )

    hubs = ["Chicago", "Los Angeles", "Atlanta", "Dallas", "Seattle"]
    for hub in hubs:
        scenarios.append(
            {
                "scenario_id": f"scenario_hub_{hub.lower().replace(' ', '_')}_weekly",
                "description": f"Weekly hub review for {hub}.",
                "conversation": [
                    {
                        "user": f"Show this week's pickup rate for {hub}.",
                        "expected_behavior": "sql_analytics",
                        "expected_tools": ["SQL", "LLM"],
                        "expected_answer_keywords": ["pickup", "rate"],
                    },
                    {
                        "user": "Compare with last week.",
                        "expected_behavior": "follow_up_compare",
                        "expected_tools": ["SQL", "MEMORY", "LLM"],
                        "expected_answer_keywords": ["week"],
                    },
                    {
                        "user": "Which drivers are underperforming?",
                        "expected_behavior": "sql_analytics",
                        "expected_tools": ["SQL", "MEMORY", "LLM"],
                        "expected_answer_keywords": ["driver"],
                    },
                    {
                        "user": "Recommend actions.",
                        "expected_behavior": "recommendation",
                        "expected_tools": ["RECOMMENDATION", "LLM"],
                        "expected_answer_keywords": ["recommend"],
                    },
                ],
                "expected_behaviors": ["sql_analytics", "memory", "recommendation"],
                "expected_tools": ["SQL", "MEMORY", "RECOMMENDATION", "LLM"],
            }
        )

    scenarios.append(
        {
            "scenario_id": "scenario_clarification_best_driver",
            "description": "Ambiguous ranking triggers clarification then resumes.",
            "conversation": [
                {
                    "user": "Show the best driver.",
                    "expected_behavior": "clarification_metric",
                    "expected_tools": [],
                    "expected_answer_keywords": ["metric"],
                },
                {
                    "user": "2",
                    "expected_behavior": "sql_analytics",
                    "expected_tools": ["SQL", "LLM"],
                    "expected_answer_keywords": ["pickup", "rate"],
                    "notes": "Clarification option for Pickup rate",
                },
            ],
            "expected_behaviors": ["clarification", "sql_analytics"],
            "expected_tools": ["SQL", "LLM"],
        }
    )

    scenarios.append(
        {
            "scenario_id": "scenario_sop_returns_then_summary",
            "description": "SOP Q&A followed by summary request.",
            "conversation": [
                {
                    "user": "What is the SOP for customer returns?",
                    "expected_behavior": "rag_sop_answer",
                    "expected_tools": ["RAG", "LLM"],
                    "expected_answer_keywords": ["return"],
                },
                {
                    "user": "Summarize the key points.",
                    "expected_behavior": "rag_sop_answer",
                    "expected_tools": ["RAG", "LLM", "MEMORY"],
                    "expected_answer_keywords": ["return"],
                },
            ],
            "expected_behaviors": ["rag_sop_answer", "memory"],
            "expected_tools": ["RAG", "LLM"],
        }
    )

    scenarios.append(
        {
            "scenario_id": "scenario_upload_wait_then_analyze",
            "description": "User asks to analyze a file without uploading first.",
            "conversation": [
                {
                    "user": "Analyze my pickup Excel file.",
                    "expected_behavior": "upload_or_attachment_handling",
                    "expected_tools": ["ATTACHMENT", "LLM"],
                    "expected_answer_keywords": ["upload", "file"],
                },
                {
                    "user": "What columns should the file include?",
                    "expected_behavior": "general_knowledge",
                    "expected_tools": ["LLM"],
                    "expected_answer_keywords": ["column", "driver"],
                },
            ],
            "expected_behaviors": ["upload_or_attachment_handling"],
            "expected_tools": ["ATTACHMENT", "LLM"],
        }
    )

    # Synthetic scenario variants to reach ~40
    metrics = ["pickup rate", "completed pickups", "failure rate", "on-time rate"]
    for i, metric in enumerate(metrics):
        for j, hub in enumerate(hubs[:3]):
            if len(scenarios) >= target:
                break
            scenarios.append(
                {
                    "scenario_id": f"scenario_metric_{i}_{j}",
                    "description": f"Multi-turn {metric} investigation for {hub}.",
                    "conversation": [
                        {
                            "user": f"Show {metric} today for {hub}.",
                            "expected_behavior": "sql_analytics",
                            "expected_tools": ["SQL", "LLM"],
                            "expected_answer_keywords": metric.split()[:2],
                        },
                        {
                            "user": "Break it down by driver.",
                            "expected_behavior": "follow_up_filter",
                            "expected_tools": ["SQL", "MEMORY", "LLM"],
                            "expected_answer_keywords": ["driver"],
                        },
                        {
                            "user": "Plot a bar chart.",
                            "expected_behavior": "visualization",
                            "expected_tools": ["VISUALIZATION", "LLM"],
                            "expected_answer_keywords": ["chart"],
                        },
                    ],
                    "expected_behaviors": ["sql_analytics", "memory", "visualization"],
                    "expected_tools": ["SQL", "MEMORY", "VISUALIZATION", "LLM"],
                }
            )

    while len(scenarios) < target:
        idx = len(scenarios) + 1
        scenarios.append(
            {
                "scenario_id": f"scenario_synthetic_{idx:02d}",
                "description": f"Synthetic ops workflow #{idx}.",
                "conversation": [
                    {
                        "user": "Show today's pickup performance.",
                        "expected_behavior": "sql_analytics",
                        "expected_tools": ["SQL", "LLM"],
                        "expected_answer_keywords": ["pickup"],
                    },
                    {
                        "user": "Show only the worst hub.",
                        "expected_behavior": "follow_up_filter",
                        "expected_tools": ["SQL", "MEMORY", "LLM"],
                        "expected_answer_keywords": ["hub"],
                    },
                    {
                        "user": "Recommend improvements.",
                        "expected_behavior": "recommendation",
                        "expected_tools": ["RECOMMENDATION", "LLM"],
                        "expected_answer_keywords": ["recommend"],
                    },
                ],
                "expected_behaviors": ["sql_analytics", "memory", "recommendation"],
                "expected_tools": ["SQL", "MEMORY", "RECOMMENDATION", "LLM"],
            }
        )
    return scenarios[:target]


def main() -> None:
    files = {
        "sop_questions.json": build_sop_questions(200),
        "sql_questions.json": build_sql_questions(100),
        "general_questions.json": build_general_questions(50),
        "followup_questions.json": build_followup_questions(50),
        "memory_questions.json": build_memory_questions(30),
        "upload_questions.json": build_upload_questions(20),
        "scenarios.json": build_scenarios(40),
    }
    for name, payload in files.items():
        path = _write(name, payload)
        count = len(payload) if isinstance(payload, list) else 0
        print(f"Wrote {count:>4} → {path}")


if __name__ == "__main__":
    main()
