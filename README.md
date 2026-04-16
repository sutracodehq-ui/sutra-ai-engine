# 🧠 SutraAI Engine

**A standalone, multi-tenant AI microservice** that powers intelligent content generation, multi-agent orchestration, and continuous self-learning — designed to serve any product via a simple REST API.

> Built with Python 3.12 · FastAPI · SQLAlchemy · Celery · Redis · PostgreSQL · Qdrant

---

## Why SutraAI?

Instead of embedding AI logic inside every product, **SutraAI is a shared brain** that any application can call:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Tryambaka   │     │  E-Commerce  │     │  Mobile App   │
│  (PHP)       │     │  (Node.js)   │     │  (Swift)      │
└──────┬───────┘     └──────┬───────┘     └───────┬───────┘
       │                    │                     │
       └────────────┬───────┘─────────────────────┘
                    │
              ┌─────▼──────┐
              │  SutraAI   │  ← API Key Auth
              │  Engine    │  ← Tenant-Scoped
              │  (Python)  │  ← Self-Learning
              └────────────┘
```

**One API. Any product. Any language.**

---

## Core Features

### 🤖 Multi-Agent System
7 specialized AI agents, each with domain expertise:

| Agent | Focus | Output |
|---|---|---|
| **Copywriter** | Headlines, body copy, CTAs | Persuasive content |
| **SEO** | Meta tags, keywords, content outlines | Search optimization |
| **Social Media** | Platform-optimized posts, hashtags | Social content |
| **Email Campaign** | Subject lines, email body, A/B variants | Email marketing |
| **WhatsApp** | Message templates, quick-reply flows | WhatsApp Business |
| **SMS** | 160-char messages, variants | SMS marketing |
| **Ad Creative** | Ad copy, hooks, CTA | Paid advertising |

### 🔌 Multi-LLM Drivers
Switch between LLM providers with zero code changes:

- **OpenAI** (GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude Sonnet, Haiku)
- **Google Gemini** (Flash, Pro)
- **Groq** (Llama 3.3 70B)
- **Ollama** (Local models — zero cost)
- **Mock** (Development/testing)

Automatic fallback chains: if OpenAI fails → try Gemini → try Ollama.

### 🧠 Intelligence Layer
- **Smart Router** — Routes simple tasks to cheap models, complex ones to powerful models
- **Quality Gate** — Auto-scores outputs, regenerates if below threshold
- **Prompt Cache** — Exact-match Redis cache for repeated prompts (0ms response)
- **Semantic Cache** — Vector similarity cache for "similar enough" prompts (40-60% fewer LLM calls)
- **Circuit Breaker** — Stops calling dead LLM services
- **Token Budget** — Per-tenant monthly limits with graceful degradation
- **Thinking Middleware** — Chain-of-thought reasoning for complex tasks

### 📚 Continuous Self-Learning
The engine gets smarter over time — automatically:

| Pipeline | Method | Schedule |
|---|---|---|
| **OPRO** | LLM analyzes accepted vs rejected → generates better instructions | Daily 2 AM |
| **TextGrad** | Reverse-engineers user edit patterns → learns preferences | Daily 3 AM |
| **A/B Testing** | Epsilon-greedy prompt variant selection | Daily 3:30 AM |
| **Training Sync** | Persists feedback data to R2 for archival | Daily 4 AM |
| **Fine-Tuning** | Regenerates Ollama Modelfiles from training data | Weekly |

### 🏢 Multi-Tenancy
- Complete data isolation per tenant (product)
- API key authentication (no JWT, no user sessions)
- Per-tenant rate limits, token budgets, and voice profiles
- Webhook callbacks for async task completion

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.12+ (for local development)

### Run Locally

```bash
# Clone
git clone https://github.com/sutracodehq/sutracode-ai-engine.git
cd sutracode-ai-engine

# Copy environment
cp .env.example .env

# Start all services (Docker **or** Podman — same compose file)
docker compose up -d --build
# Podman: podman compose up -d --build

# Run migrations
docker compose exec sutra-ai-api alembic upgrade head
# Podman: podman compose exec sutra-ai-api alembic upgrade head

# Create your first tenant (API is published on host port 8090 in docker-compose.yml)
curl -X POST http://localhost:8090/v1/tenants \
  -H "Authorization: Bearer sk_master_your_admin_key" \
  -H "Content-Type: application/json" \
  -d '{"name": "Tryambaka", "slug": "tryambaka"}'

# Response: { "api_key": "sk_live_xxxxxxxx" }
```

### Your First AI Call

```bash
curl -X POST http://localhost:8090/v1/chat \
  -H "Authorization: Bearer sk_live_xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "social_media",
    "prompt": "Write a LinkedIn post about launching our new AI product"
  }'
```

---

## API Reference

### Authentication
All requests require a tenant API key:
```
Authorization: Bearer sk_live_xxxxxxxx
```

### Endpoints

#### Chat
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/chat` | Single-turn agent completion |
| `POST` | `/v1/chat/stream` | SSE streaming completion |

#### Conversations
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/conversations` | Create new conversation |
| `GET` | `/v1/conversations/{id}` | Get conversation + messages |
| `POST` | `/v1/conversations/{id}/messages` | Send message in conversation |

#### Agents
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/agents` | List available agents |
| `POST` | `/v1/agents/{type}/run` | Run agent task (async) |
| `POST` | `/v1/agents/batch` | Parallel multi-agent run |

#### Tasks
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/tasks/{id}` | Get task status + result |
| `POST` | `/v1/tasks/{id}/feedback` | Submit feedback (accepted/edited/rejected) |

#### Tenants (Admin)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/tenants` | Register new tenant |
| `GET` | `/v1/tenants/{id}` | Get tenant info |
| `GET` | `/v1/tenants/{id}/usage` | Token usage + cost analytics |
| `POST` | `/v1/tenants/{id}/api-keys` | Rotate API key |

#### Health
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness (checks DB + Redis + Qdrant, informational) |

---

## Architecture

```
sutracode-ai-engine/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Environment config (Pydantic Settings)
│   ├── dependencies.py         # DI: auth, DB session, Redis
│   ├── api/v1/                 # Route modules
│   ├── models/                 # SQLAlchemy models
│   ├── schemas/                # Pydantic request/response
│   └── services/
│       ├── drivers/            # LLM drivers (OpenAI, Anthropic, etc.)
│       ├── agents/             # AI agents + hub orchestrator
│       ├── chat/               # Chat pipeline + steps
│       ├── intelligence/       # Cache, routing, quality, circuit breaker
│       ├── learning/           # OPRO, TextGrad, A/B testing
│       └── rag/                # Vector store, web crawler, extractors
├── agent_config/               # Agent prompts + schemas (YAML)
├── alembic/                    # Database migrations
├── docker/                     # Dockerfiles + nginx
├── kubernetes/                 # K8s manifests
├── tests/                      # pytest test suite
├── docker-compose.yml          # Local dev (Compose — Docker or Podman)
└── pyproject.toml              # Dependencies
```

### Service Map

| Service | Port | Purpose |
|---|---|---|
| `sutra-ai-api` | 8000 | FastAPI + Uvicorn |
| `sutra-ai-worker` | — | Celery worker (queue processing) |
| `sutra-ai-beat` | — | Celery Beat (cron scheduler) |
| `sutra-ai-postgres` | 5432 | PostgreSQL 16 |
| `sutra-ai-redis` | 6379 | Redis 7 (cache + queues) |
| `sutra-ai-qdrant` | 6333 | Qdrant (vector store) |
| `sutra-ai-ollama` | 11434 | Ollama (local LLM) |

---

## Integration Examples

### PHP (Tryambaka / Laravel)
```php
$response = Http::withToken('sk_live_xxx')
    ->post('https://ai.sutracodehq.com/v1/chat', [
        'agent_type' => 'copywriter',
        'prompt' => 'Write a landing page headline for...',
    ]);

$result = $response->json();
```

### Node.js
```javascript
const res = await fetch('https://ai.sutracodehq.com/v1/chat', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer sk_live_xxx',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    agent_type: 'social_media',
    prompt: 'Create an Instagram post about...',
  }),
});
const data = await res.json();
```

### Python
```python
import httpx

response = httpx.post(
    "https://ai.sutracodehq.com/v1/chat",
    headers={"Authorization": "Bearer sk_live_xxx"},
    json={"agent_type": "seo", "prompt": "Audit this URL..."},
)
result = response.json()
```

### cURL
```bash
curl -X POST https://ai.sutracodehq.com/v1/chat \
  -H "Authorization: Bearer sk_live_xxx" \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "email_campaign", "prompt": "Welcome email for..."}'
```

---

## Deployment

### Compose (Dev/VPS — Docker or Podman)
```bash
docker compose up -d --build
# Podman:
podman compose up -d --build
```
Use the same `docker-compose.yml`. On rootless Podman, bind-mounted project dirs work like Docker; if SELinux blocks volume access on Fedora/RHEL, add `:z` to those bind mounts (see Podman docs).

### Kubernetes
```bash
kubectl apply -f kubernetes/
```

See [kubernetes/](kubernetes/) for full manifests including HPA, CronJobs, and Ingress.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `AI_DRIVER` | `mock` | Default LLM driver |
| `AI_FALLBACK_DRIVER` | `gemini` | Fallback LLM driver |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GROQ_API_KEY` | — | Groq API key |
| `OLLAMA_BASE_URL` | `http://sutra-ai-ollama:11434` | Ollama endpoint |
| `QDRANT_URL` | `http://sutra-ai-qdrant:6333` | Qdrant HTTP API |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for vector embeddings (must match `EMBEDDING_VECTOR_SIZE`) |
| `R2_ACCESS_KEY` | — | Cloudflare R2 access key |
| `R2_SECRET_KEY` | — | Cloudflare R2 secret key |
| `R2_BUCKET` | `sutra-ai-storage` | R2 bucket name |
| `R2_ENDPOINT` | — | R2 endpoint URL |
| `MASTER_API_KEY` | — | Admin key for tenant management |

**Vector store:** Qdrant replaces Chroma; after upgrade **re-index** brand/FAQ data (e.g. `scripts/seed_knowledge.py` or `brand_import_faq`). Old Chroma vectors are not portable.

See [.env.example](.env.example) for the full list.

---

## License

Private — © SutraCodeHQ
