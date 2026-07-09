# GOFO Operations Intelligence Dashboard

## Start API

```bash
uvicorn api.server:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Ask the agent:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How are operations today?"}'
```

## Start UI

```bash
streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

## CLI

```bash
python query.py
python query.py --debug
```
