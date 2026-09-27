# Harrier RunPod Serverless Worker

RunPod Serverless worker that embeds text with [`microsoft/harrier-oss-v1-27b`](https://huggingface.co/microsoft/harrier-oss-v1-27b) via Sentence Transformers.

Uses a **new enough Transformers stack** (`transformers>=4.57`) so model type `gemma3_text` is recognized (the stock `runpod/worker-infinity-embedding:1.1.4` image fails on this architecture).

## Environment

| Variable     | Default                         | Notes                                      |
|--------------|---------------------------------|--------------------------------------------|
| `MODEL_NAME` | `microsoft/harrier-oss-v1-27b`  | Any compatible SentenceTransformer id/path |
| `HF_HOME`    | (optional)                      | Prefer a RunPod volume cache if mounted    |

Pre-configured query prompts: `web_search_query`, `sts_query`, `bitext_query`. Documents should be encoded **without** a prompt.

## Example `/runsync` body

```json
{
  "input": {
    "texts": [
      "how much protein should a female eat",
      "summit define"
    ],
    "prompt_name": "web_search_query"
  }
}
```

Or a single string via `"input": "some text"`, optional custom `"prompt": "Instruct: ...\nQuery: "`.

Response shape:

```json
{
  "embeddings": [[0.01, -0.02, "..."]],
  "dimensions": 5376,
  "model": "microsoft/harrier-oss-v1-27b"
}
```

## Container image

Published to GHCR (via Actions on `main`):

`ghcr.io/justindowding555-lgtm/harrier-runpod-worker:latest`

## Build (local)

```bash
docker build -t harrier-runpod-worker .
```
