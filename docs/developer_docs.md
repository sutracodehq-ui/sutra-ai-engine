# SutraCode AI Engine — Developer Documentation

> **187 agents · 32 products · 40 phases · India-first multi-sector AI platform**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Core Concepts](#core-concepts)
4. [Agent System](#agent-system)
5. [Security Layer](#security-layer)
6. [Billing & Metering](#billing--metering)
7. [Product Catalog](#product-catalog)
8. [Intelligence Layer](#intelligence-layer)
9. [Multi-Modal Output](#multi-modal-output)
10. [API Reference](#api-reference)
11. [Configuration Files](#configuration-files)
12. [Adding a New Agent](#adding-a-new-agent)
13. [Inter-Agent Communication](#inter-agent-communication)
14. [Deployment](#deployment)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                          │
│                  (API Key in X-API-Key header)                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    SECURE GATEWAY (8 Layers)                    │
│                                                                 │
│  1. API Key Auth     →  Validates sc_live_xxx / sc_test_xxx     │
│  2. IP Whitelist     →  Per-tenant allowed IPs                  │
│  3. Anti-Replay      →  Nonce + timestamp (5-min window)        │
│  4. HMAC Verify      →  SHA-256 request signing                 │
│  5. Injection Guard  →  Jailbreak / prompt leak detection       │
│  6. PII Redaction    →  Masks Aadhaar, PAN, phone before LLM   │
│  7. Rate Limiter     →  Tier-based daily limits                 │
│  8. Response Cache   →  LRU cache with domain-specific TTLs    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                       AGENT HUB                                 │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Learning │  │ Memory   │  │ Prompt   │  │ Multi-   │       │
│  │ System   │  │ (RAG)    │  │ Engine   │  │ Lingual  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       └──────────────┴─────────────┴──────────────┘             │
│                          │                                      │
│                   ┌──────▼──────┐                               │
│                   │ BASE AGENT  │                               │
│                   │  (execute)  │                               │
│                   └──────┬──────┘                               │
│                          │                                      │
│  ┌───────────────────────┼───────────────────────────┐         │
│  │  149 Specialized Agents across 31 Phases          │         │
│  │  Marketing │ Health │ Legal │ Finance │ ...       │         │
│  └───────────────────────────────────────────────────┘         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   MULTI-MODAL OUTPUT                            │
│                                                                 │
│  Text │ Voice (Edge-TTS) │ Image │ Video Script │ Steps │ Chart│
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│              USAGE TRACKING + AUDIT LOG                         │
│         (Per-tenant, per-agent, per-day counters)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
sutracode-ai-engine/
│
├── agent_config/                    # 147 YAML agent configurations
│   ├── copywriter.yaml
│   ├── seo.yaml
│   ├── tax_planner.yaml
│   └── ... (one per agent)
│
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── billing.py          # API key CRUD, usage stats, metered execute
│   │   │   ├── secure.py           # Secure gateway endpoints + smart routing
│   │   │   ├── chatbot.py          # Chatbot integration endpoints
│   │   │   ├── chatbot_ws.py       # WebSocket chatbot
│   │   │   ├── feedback.py         # Feedback collection
│   │   │   └── voip.py             # Voice/WebRTC endpoints
│   │   └── v1/
│   │       ├── agents.py           # Agent management
│   │       ├── billing.py          # V1 billing
│   │       ├── chat.py             # Chat endpoints
│   │       ├── content.py          # Content generation
│   │       ├── conversations.py    # Conversation management
│   │       ├── intelligence.py     # Intelligence endpoints
│   │       ├── rag.py              # RAG endpoints
│   │       ├── url_analyzer.py     # URL analysis
│   │       └── voice.py            # Voice endpoints
│   │
│   ├── services/
│   │   ├── agents/
│   │   │   ├── base.py             # BaseAgent (all agents extend this)
│   │   │   ├── hub.py              # AiAgentHub — registry + dispatch + delegation
│   │   │   ├── copywriter.py       # 149 agent Python files
│   │   │   ├── seo.py
│   │   │   └── ...
│   │   │
│   │   ├── billing/
│   │   │   ├── api_keys.py         # API key generation, validation, rotation
│   │   │   ├── usage_tracker.py    # Per-tenant usage counters
│   │   │   ├── rate_limiter.py     # Tier-based rate limiting
│   │   │   ├── gateway.py          # Billing gateway (metered execution)
│   │   │   └── product_registry.py # Product catalog manager
│   │   │
│   │   ├── security/
│   │   │   ├── injection_guard.py  # Prompt injection detection
│   │   │   ├── pii_redactor.py     # PII masking (Aadhaar, PAN, etc.)
│   │   │   ├── request_auth.py     # HMAC + IP whitelist + anti-replay
│   │   │   ├── audit_logger.py     # Immutable audit log
│   │   │   └── secure_gateway.py   # 8-layer security orchestrator
│   │   │
│   │   ├── intelligence/
│   │   │   ├── agent_learning.py   # Feedback → learning → quality scoring
│   │   │   ├── multimodal_engine.py# Voice, image, video, steps output
│   │   │   ├── agent_memory.py     # RAG-based agent memory
│   │   │   ├── prompt_engine.py    # A/B testing prompt optimization
│   │   │   └── multilingual.py     # Multi-language support
│   │   │
│   │   ├── optimization/
│   │   │   ├── response_cache.py   # LRU cache with domain TTLs
│   │   │   └── smart_router.py     # Auto-detect best agent from prompt
│   │   │
│   │   ├── chat/                   # Chat session management
│   │   ├── connectivity/           # External integrations
│   │   ├── drivers/                # LLM provider drivers
│   │   ├── rag/                    # Retrieval-Augmented Generation
│   │   └── voice/                  # Voice/TTS services
│   │
│   ├── models/                     # SQLAlchemy database models
│   ├── schemas/                    # Pydantic request/response schemas
│   ├── middleware/                  # FastAPI middleware
│   ├── workers/tasks/              # Background task workers
│   └── config.py                   # Application settings
│
├── config/
│   ├── languages.yaml              # Supported languages
│   ├── openapi.yaml                # OpenAPI spec
│   └── voice.yaml                  # Voice configuration
│
├── intelligence_config.yaml        # Master AI configuration
├── product_catalog.yaml            # 23 products, 3 bundles
├── scanner_feeds.yaml              # External data feeds
├── docker-compose.yml              # Compose (Docker / Podman)
└── pyproject.toml                  # Python dependencies
```

---

## Core Concepts

### Software Factory Principle

Every component follows the **Software Factory** approach:

- **Config-driven**: Agent behavior is defined in YAML, not hardcoded
- **Polymorphic**: One `BaseAgent` class handles all 149 agents
- **Self-learning**: Agents improve from user feedback automatically
- **Metered**: Every API call is tracked, rate-limited, and auditable

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| YAML configs over hardcoded prompts | A/B test prompts without code deploys |
| Singleton services | One instance per process, lazy initialization |
| Async everywhere | Non-blocking I/O for concurrent agent calls |
| India-first | All agents have Indian context (₹, GST, Aadhaar, etc.) |
| Sector-wise products | Sell what each customer needs, not the whole engine |

---

## Agent System

### BaseAgent (`app/services/agents/base.py`)

Every agent extends `BaseAgent`. It provides:

```python
class BaseAgent:
    identifier = "copywriter"  # Unique agent ID — matches YAML filename

    async def get_system_prompt(db, context)
    # 1. Try PromptEngine (A/B testing from DB)
    # 2. Fallback to YAML config
    # Auto-injects: Chain-of-Thought, JSON mode, Rules, Capabilities

    async def build_messages(prompt, history, db, context)
    # 1. Resolve system prompt
    # 2. Inject multilingual instructions
    # 3. Inject RAG memory (past good responses)
    # 4. Add conversation history
    # 5. Add user prompt

    async def execute(prompt, db, context)
    # Build messages → Call LLM → Store in memory → Return response

    async def execute_in_conversation(prompt, history, db, context)
    # Same as execute but with full conversation history
```

### Agent YAML Config (`agent_config/*.yaml`)

```yaml
name: "Tax Planner"
identifier: tax_planner              # Must match Python class identifier
domain: "finance"                     # Used for product grouping
description: "Indian income tax planning..."

system_prompt: |                      # The heart of the agent
  You are an Indian Income Tax Planning Expert.
  ...

capabilities: [tax_planning, regime_comparison, 80c_deductions]
rules: [india_tax_law, current_rates, disclaimer]

# Optional:
response_schema:
  format: json                        # Forces JSON output
  fields: [recommendation, savings, comparison]
```

### AiAgentHub (`app/services/agents/hub.py`)

Central registry and dispatcher:

```python
from app.services.agents.hub import get_agent_hub

hub = get_agent_hub()

# Run a single agent
response = await hub.run("tax_planner", "How to save tax under new regime?")

# Run multiple agents in parallel
results = await hub.batch("Analyze this brand", ["seo", "brand_auditor", "competitor_analyst"])

# Safe inter-agent delegation
result = await hub.delegate(
    from_agent="trip_planner",
    to_agent="visa_guide",
    prompt="What's the visa process for Japan?",
)

# Multi-delegation (parallel)
results = await hub.multi_delegate(
    from_agent="daily_briefing",
    to_agents=["market_trend_analyzer", "threat_briefing", "weather_planting"],
    prompt="What's happening today?",
)
```

### Inter-Agent Delegation Safety

Agents communicate but **never deadlock**:

| Guard | Protection |
|-------|-----------|
| **Max depth = 3** | A→B→C→D stops — no infinite chains |
| **Cycle detection** | A→B→A breaks immediately |
| **Timeout = 30s** | No agent waits forever |
| **Fallback** | Always returns a response, even on failure |

---

## Security Layer

### Secure Gateway (`app/services/security/secure_gateway.py`)

8-layer security pipeline — single entry point for all external API calls:

```python
from app.services.security.secure_gateway import get_secure_gateway

gateway = get_secure_gateway()
result = await gateway.execute(
    api_key="sc_live_xxx",
    agent_id="copywriter",
    prompt="Write a tagline for my chai brand",
    request_ip="203.0.113.42",
    nonce="abc123",                     # Anti-replay
    timestamp="1711065600",              # Unix timestamp
    signature="v1=hmac_sha256_hex",      # HMAC signature
)
```

### Layer Details

#### 1. Prompt Injection Guard (`injection_guard.py`)

Three severity levels with base64 bypass detection:

```python
from app.services.security.injection_guard import get_injection_guard

guard = get_injection_guard()
result = guard.check("Ignore all previous instructions and...")
# → InjectionResult(is_safe=False, risk_score=0.9, triggers=["HIGH: ignore..."])
```

**Detects:**
- System prompt override ("ignore all previous instructions")
- Role manipulation ("you are now DAN")
- Prompt leaking ("show me your system prompt")
- Base64/rot13 encoded attacks
- Unicode homoglyph/zero-width character obfuscation

#### 2. PII Redactor (`pii_redactor.py`)

Masks Indian PII before the LLM ever sees it:

```python
from app.services.security.pii_redactor import get_pii_redactor

redactor = get_pii_redactor()
result = redactor.redact("My Aadhaar is 1234 5678 9012 and PAN is ABCDE1234F")
# → "My [Aadhaar Number: XXXXXXXX9012] and [PAN Card: XXXXX1234F]"
```

**Supported PII:**

| Type | Pattern | Context-aware |
|------|---------|--------------|
| Aadhaar | 12-digit | No |
| PAN | ABCDE1234F | No |
| Phone | +91 / 10-digit starting 6-9 | No |
| Email | user@domain.com | No |
| Bank Account | 9-18 digits | Yes — only when "account"/"bank" present |
| IFSC | XXXX0XXXXXX | No |
| Credit Card | 16 digits | No |
| UPI | user@upi | Yes — only when "upi"/"pay" present |
| Passport | X1234567 | Yes — only when "passport" present |

#### 3. Request Authenticator (`request_auth.py`)

```python
from app.services.security.request_auth import get_request_authenticator

auth = get_request_authenticator()

# Set up tenant security
auth.set_ip_whitelist("acme", ["203.0.113.42", "198.51.100.0"])
auth.set_signing_secret("acme", "whsec_super_secret_key")

# Full check (IP + HMAC + anti-replay)
result = auth.full_check(
    tenant_id="acme",
    request_ip="203.0.113.42",
    payload='{"agent_id":"copywriter","prompt":"..."}',
    timestamp="1711065600",
    signature="v1=abc123...",
    nonce="unique_request_id",
)
# → {"passed": True} or {"passed": False, "check": "ip_whitelist", ...}
```

#### 4. Audit Logger (`audit_logger.py`)

Immutable JSONL log of every API call:

```python
from app.services.security.audit_logger import get_audit_logger

audit = get_audit_logger()

# Logged automatically by SecureGateway, but can be used directly:
event_id = audit.log_request(
    tenant_id="acme", agent_id="copywriter", prompt="...",
    ip_address="203.0.113.42", tier="pro", latency_ms=450,
)

# Query logs
logs = audit.get_tenant_log("acme", limit=50)
stats = audit.get_stats("acme")
security_events = audit.get_security_events()
```

**Stored per entry:**
- Request/response SHA-256 hashes (not raw content — privacy-safe)
- Latency, tokens used, agent ID, tier
- Security flags, PII count, injection risk score
- Rate limit status, remaining calls

**Storage:** `/tmp/sutracode_audit/audit_YYYY-MM-DD.jsonl` (append-only)

---

## Billing & Metering

### API Keys (`app/services/billing/api_keys.py`)

```python
from app.services.billing.api_keys import get_api_key_manager

manager = get_api_key_manager()

# Generate key (raw key shown ONCE, stored as SHA-256 hash)
raw_key = manager.generate(tenant_id="acme", tier="pro", name="Production")
# → "sc_live_AbCdEf12xxxxxxxxxxxxxxxxxxxxxxxxx"

# Test mode key (no billing)
test_key = manager.generate(tenant_id="acme", tier="pro", is_test=True)
# → "sc_test_XyZ789xxxxxxxxxxxxxxxxxxxxxxxxx"

# Validate
key_info = manager.validate(raw_key)  # → ApiKey dataclass or None

# Rotate (generate new + revoke old)
new_key = manager.rotate("acme", "sc_live_AbCdEf12")

# Upgrade tier
manager.update_tier("acme", "enterprise")
```

### Rate Limiter (`app/services/billing/rate_limiter.py`)

Four subscription tiers:

| Tier | ₹/mo | Daily Limit | Agents | Voice | WebSocket | Priority |
|------|------|------------|--------|-------|-----------|----------|
| **Free** | ₹0 | 50 | 5 basic | ❌ | ❌ | ❌ |
| **Starter** | ₹999 | 500 | 25 (marketing + productivity) | ❌ | ✅ | ❌ |
| **Pro** | ₹2,999 | 2,000 | All 149 | ✅ | ✅ | ❌ |
| **Enterprise** | ₹9,999 | Unlimited | All + custom | ✅ | ✅ | ✅ |

```python
from app.services.billing.rate_limiter import get_rate_limiter

limiter = get_rate_limiter()
result = limiter.check(tenant_id="acme", tier="starter",
                       agent_id="tax_planner", current_daily_usage=499)
# → {"allowed": True, "remaining": 0}

result = limiter.check(tenant_id="acme", tier="starter",
                       agent_id="tax_planner", current_daily_usage=500)
# → {"allowed": False, "reason": "daily_limit_exceeded",
#     "message": "Upgrade to Pro for more.", "upgrade_to": "Pro"}
```

### Usage Tracker (`app/services/billing/usage_tracker.py`)

```python
from app.services.billing.usage_tracker import get_usage_tracker

tracker = get_usage_tracker()

# Track a call
tracker.track(tenant_id="acme", agent_id="copywriter", latency_ms=340)

# Get today's usage
today = tracker.get_daily_usage("acme")
# → {"total_calls": 42, "by_agent": {"copywriter": 15, "seo": 27}}

# Get 30-day summary
summary = tracker.get_usage_summary("acme", days=30)
# → {"total_calls": 1250, "top_agents": {"copywriter": 300, ...}, "daily": [...]}
```

---

## Product Catalog

### Overview (`product_catalog.yaml`)

149 agents divided into **23 deployable products** + **3 bundles**:

| Product | Code | Agents | ₹/mo | Target |
|---------|------|--------|------|--------|
| Tryambaka Marketing | `tryambaka_marketing` | 39 | 2,999 | Agencies, D2C |
| VoiceFlow | `voiceflow` | 5 | 1,999 | Call centers |
| VideoMind | `videomind` | 5 | 1,499 | YouTubers |
| EdBrain | `edbrain` | 5 | 999 | Students |
| FinWise | `finwise` | 10 | 2,499 | Investors, CAs |
| HealthMate | `healthmate` | 9 | 1,499 | Clinics |
| LegalEase | `legalease` | 4 | 1,999 | Lawyers |
| HireGenius | `hiregenius` | 5 | 1,999 | HR teams |
| ShopBrain | `shopbrain` | 5 | 1,999 | E-com sellers |
| KisanAI | `kisan_ai` | 5 | 499 | Farmers |
| PropertyGuru | `propertyguru` | 4 | 1,499 | Home buyers |
| TravelBuddy | `travelbuddy` | 5 | 999 | Travelers |
| LogiSmart | `logismart` | 4 | 2,999 | Logistics |
| JanSeva | `janseva` | 4 | 299 | Citizens |
| SuccessIQ | `successiq` | 5 | 2,499 | SaaS companies |
| WorkPilot | `workpilot` | 6 | 999 | Freelancers |
| CreatorStudio | `creatorstudio` | 5 | 1,499 | Creators |
| CyberShield | `cybershield` | 5 | 2,999 | IT teams |
| LaunchPad | `launchpad` | 5 | 1,999 | Founders |
| GreenIQ | `greeniq` | 4 | 1,999 | ESG compliance |
| SportsIQ | `sportsiq` | 4 | 999 | Sports fans |
| EcoMotion | `ecomotion` | 5 | 999 | EV buyers |
| DataForge | `dataforge` | 3 | 2,499 | Data scientists |

### Product Registry (`app/services/billing/product_registry.py`)

```python
from app.services.billing.product_registry import get_product_registry

registry = get_product_registry()

# List all products (pricing page)
products = registry.list_products()

# Get agents in a product
agents = registry.get_agents_for_product("finwise")
# → ["stock_analyzer", "tax_planner", "sip_calculator", ...]

# Check agent access
registry.is_agent_in_product("copywriter", "tryambaka_marketing")  # True
registry.is_agent_in_product("copywriter", "kisan_ai")             # False

# Get bundle agents
agents = registry.get_bundle_agents("business_suite")
# → All agents from Marketing + HR + Legal + Finance + Productivity
```

---

## Intelligence Layer

### Agent Learning (`app/services/intelligence/agent_learning.py`)

Continuous improvement through user feedback:

```python
from app.services.intelligence.agent_learning import get_agent_learning

learning = get_agent_learning()

# Submit feedback (user rates a response)
learning.submit_feedback(
    agent_id="tax_planner",
    tenant_id="acme",
    prompt="How to save tax?",
    response="Under Section 80C...",
    rating=5,                          # 1-5 scale
    correction="",                      # Correction if bad response
)

# Get learnings to inject into prompt (called automatically by BaseAgent)
context = learning.get_learnings_for_prompt("tax_planner", "How to save tax?")
# → Past good examples + corrections injected into system prompt

# Monitor quality
quality = learning.get_quality("tax_planner")
# → {"avg_rating": 4.2, "trend": "improving", "positive_rate": "85%"}

# Admin: get degrading agents
alerts = learning.get_degrading_agents()
# → Agents with avg_rating < 3.0 or trend == "degrading"
```

**Learning flow:**

```
Rating ≥ 4 → Stored as good example (Qdrant)
Rating ≤ 2 + correction → Stored as training data
Next call → Relevant past learnings injected into system prompt
Agent improves over time 📈
```

### Smart Router (`app/services/optimization/smart_router.py`)

Auto-detects the best agent from a natural language prompt:

```python
from app.services.optimization.smart_router import get_smart_router

router = get_smart_router()

result = router.route("What's my income tax liability under old regime?")
# → RouteResult(agent_id="tax_planner", confidence=0.9,
#               alternatives=["gst_compliance", "loan_comparator"])

result = router.route("My tomato crop has yellow leaves")
# → RouteResult(agent_id="crop_advisor", confidence=0.8)

result = router.route("Write a viral reel script about fitness")
# → RouteResult(agent_id="reel_script_writer", confidence=0.85)
```

### Response Cache (`app/services/optimization/response_cache.py`)

Same prompt + same agent = cached response (saves LLM calls):

```python
from app.services.optimization.response_cache import get_response_cache

cache = get_response_cache()

# Domain-specific TTLs:
# marketing: 1 hour, finance: 15 min, health: NEVER cached, logistics: 5 min

# Agents that NEVER cache (safety-critical or real-time):
# symptom_triage, mental_health_companion, medicine_info,
# dynamic_pricing, shipment_tracker, daily_briefing, reminder_agent

stats = cache.stats()
# → {"size": 1234, "hits": 500, "misses": 200, "hit_rate": "71.4%"}
```

---

## Multi-Modal Output

### MultiModalEngine (`app/services/intelligence/multimodal_engine.py`)

Transforms text responses into human-like outputs:

```python
from app.services.intelligence.multimodal_engine import get_multimodal_engine, OutputMode

engine = get_multimodal_engine()

response = await engine.generate(
    text="Here's your diet plan: 1. Oats with almonds...",
    agent_id="diet_planner",
    modes=[OutputMode.TEXT, OutputMode.VOICE, OutputMode.STEPS],
    voice="hi-female",  # Hindi female voice
)

response.text           # Original text
response.voice_audio    # Base64 MP3 (Edge-TTS)
response.steps          # [{"step": 1, "title": "Oats with almonds", "icon": "📌"}, ...]
response.video_script   # Scene-by-scene narration
response.image_prompts  # AI image generation prompts
response.chart_data     # JSON for frontend charts
```

### Available Voices (Edge-TTS)

| Key | Voice | Language |
|-----|-------|----------|
| `hi-male` | MadhurNeural | Hindi |
| `hi-female` | SwaraNeural | Hindi |
| `en-male` | PrabhatNeural | English (Indian) |
| `en-female` | NeerjaNeural | English (Indian) |
| `ta-male` | ValluvarNeural | Tamil |
| `ta-female` | PallaviNeural | Tamil |
| `te-male` | MohanNeural | Telugu |
| `te-female` | ShrutiNeural | Telugu |
| `bn-male` | BashkarNeural | Bengali |
| `bn-female` | TanishaaNeural | Bengali |
| `mr-male` | ManoharNeural | Marathi |
| `mr-female` | AarohiNeural | Marathi |

### Output Modes

| Mode | Technology | When to use |
|------|-----------|-------------|
| `TEXT` | Markdown | Always (default) |
| `VOICE` | Edge-TTS | Chatbots, accessibility, hands-free |
| `IMAGE` | AI prompts | Visual content (diet plans, designs) |
| `VIDEO_SCRIPT` | Scene structuring | Video creators |
| `STEPS` | Auto-extraction | Tutorials, processes, guides |
| `CHART` | JSON extraction | Financial data, analytics |
| `FULL` | All of the above | Premium tier |

---

## API Reference

### Billing Endpoints (`/v1/billing`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/v1/billing/keys` | Admin | Generate new API key |
| `GET` | `/v1/billing/keys?tenant_id=x` | Admin | List tenant's keys |
| `POST` | `/v1/billing/keys/revoke` | Admin | Revoke a key |
| `POST` | `/v1/billing/keys/rotate` | Admin | Rotate a key |
| `GET` | `/v1/billing/usage?tenant_id=x` | Admin | Usage summary |
| `GET` | `/v1/billing/usage/today?tenant_id=x` | Admin | Today's usage |
| `GET` | `/v1/billing/tiers` | Public | All tier info |
| `POST` | `/v1/billing/execute` | X-API-Key | Metered agent call |

### Secure Endpoints (`/v1/secure`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/v1/secure/execute` | X-API-Key | Full 8-layer secure execution |
| `POST` | `/v1/secure/auto` | X-API-Key | Smart-routed (auto-detect agent) |
| `POST` | `/v1/secure/multimodal` | X-API-Key | Multi-modal output |
| `POST` | `/v1/secure/feedback` | X-API-Key | Submit learning feedback |
| `GET` | `/v1/secure/quality` | Admin | Agent quality dashboard |
| `GET` | `/v1/secure/quality/alerts` | Admin | Degrading agents |
| `GET` | `/v1/secure/audit?tenant_id=x` | Admin | Audit log |
| `GET` | `/v1/secure/audit/security` | Admin | Security events |
| `GET` | `/v1/secure/cache/stats` | Admin | Cache performance |

### Example: Secure Execute

```bash
curl -X POST https://api.sutracode.ai/v1/secure/execute \
  -H "X-API-Key: sc_live_AbCdEf12..." \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "tax_planner",
    "prompt": "Compare old vs new tax regime for ₹15L salary",
    "context": {"language": "hi"}
  }'
```

Response:
```json
{
  "success": true,
  "response": "## Old vs New Regime Comparison\n\n...",
  "agent": "tax_planner",
  "cached": false,
  "usage": {
    "remaining": 1999,
    "tier": "pro",
    "latency_ms": 1240
  },
  "security": {
    "pii_redacted": 0,
    "injection_risk": 0.0
  }
}
```

### Example: Auto-Route

```bash
curl -X POST https://api.sutracode.ai/v1/secure/auto \
  -H "X-API-Key: sc_live_AbCdEf12..." \
  -d '{"prompt": "My rice crop has brown spots on leaves"}'
```

Response:
```json
{
  "success": true,
  "response": "Brown spots on rice leaves indicate...",
  "routing": {
    "agent_selected": "crop_advisor",
    "confidence": 0.85,
    "alternatives": ["soil_report_interpreter", "weather_planting"]
  }
}
```

### Example: Multi-Modal

```bash
curl -X POST https://api.sutracode.ai/v1/secure/multimodal \
  -H "X-API-Key: sc_live_AbCdEf12..." \
  -d '{
    "agent_id": "diet_planner",
    "prompt": "Give me a diabetic-friendly Indian diet plan",
    "modes": ["text", "voice", "steps"],
    "voice": "hi-female"
  }'
```

---

## Configuration Files

### `intelligence_config.yaml`

Master configuration for AI behavior, including:

- LLM model settings
- Agent-specific parameters
- Health severity mappings
- **Billing tiers and pricing**

### `product_catalog.yaml`

23 products + 3 bundles with agent-to-product mapping.

### `agent_config/*.yaml`

One YAML per agent. Required fields:

```yaml
name: "Human-readable name"
identifier: "snake_case_id"        # Must match Python class
domain: "sector_name"
description: "What this agent does"
system_prompt: |
  Multi-line system prompt...
capabilities: [list, of, capabilities]
rules: [list, of, rules]
```

---

## Adding a New Agent

### Step 1: Create YAML config

```bash
# agent_config/my_new_agent.yaml
```

```yaml
name: "My New Agent"
identifier: my_new_agent
domain: "my_sector"
description: "What it does."

system_prompt: |
  You are a specialist in...

capabilities: [capability_1, capability_2]
rules: [rule_1, rule_2]
```

### Step 2: Create Python class

```bash
# app/services/agents/my_new_agent.py
```

```python
"""My New Agent — my_sector domain agent."""
from app.services.agents.base import BaseAgent


class MyNewAgentAgent(BaseAgent):
    identifier = "my_new_agent"
```

### Step 3: Register in hub

Edit `app/services/agents/hub.py`:

```python
# Add import
from app.services.agents.my_new_agent import MyNewAgentAgent

# Add to the agent list
for agent_cls in [
    # ... existing agents ...
    MyNewAgentAgent,  # ← Add here
]:
```

### Step 4: Add to product catalog

Edit `product_catalog.yaml`:

```yaml
my_product:
  agents:
    - my_new_agent    # ← Add here
```

### Step 5: Add to smart router (optional)

Edit `app/services/optimization/smart_router.py`:

```python
ROUTING_RULES.append({
    "keywords": ["relevant", "keywords", "for", "this", "agent"],
    "agent": "my_new_agent",
    "domain": "my_sector",
})
```

---

## Deployment

### Docker or Podman Compose

```bash
docker compose up -d --build
# or:
podman compose up -d --build
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis for caching/queues | `redis://localhost:6379` |
| `OLLAMA_HOST` | Ollama LLM server | `http://localhost:11434` |
| `AI_AGENT_MEMORY_ENABLED` | Enable RAG memory | `true` |
| `QDRANT_URL` | Qdrant HTTP API for vector search | `http://localhost:6333` |

### Health Check

```bash
curl http://localhost:8090/health
```

---

## Complete Agent Catalog (149 Agents)

All agents grouped by product. Each agent entry includes its identifier, description, capabilities, and real-world use cases.

---

### 📢 Product 1: Tryambaka Marketing Suite (39 agents)

> The flagship marketing intelligence platform. Covers everything from copywriting to attribution analysis.

#### Core Marketing Agents

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 1 | **Copywriter** | `copywriter` | Generates high-converting marketing copy: taglines, headlines, ad copy, product descriptions, CTAs, and brand stories. |
| 2 | **SEO Specialist** | `seo` | On-page and off-page SEO optimization: keyword research, meta descriptions, content structure, internal linking, and SERP analysis. |
| 3 | **Social Media Manager** | `social` | Creates platform-native social media posts for Instagram, Twitter/X, LinkedIn, and Facebook with hashtag strategy. |
| 4 | **Email Campaign Manager** | `email_campaign` | Designs email sequences: welcome series, drip campaigns, newsletters, re-engagement flows, and subject line optimization. |
| 5 | **WhatsApp Marketer** | `whatsapp` | Creates WhatsApp Business API templates, broadcast messages, and conversational flows compliant with Meta policies. |
| 6 | **SMS Campaign Writer** | `sms` | Writes concise SMS campaigns within 160-character limits with trackable short links and action-oriented CTAs. |
| 7 | **Ad Creative Designer** | `ad_creative` | Generates ad creatives for Google, Meta, and LinkedIn with A/B test variations, targeting suggestions, and budget allocation. |
| 8 | **Brand Auditor** | `brand_auditor` | Audits brand consistency across channels: voice, tone, visual identity, messaging alignment, and competitive positioning. |
| 9 | **Content Repurposer** | `content_repurposer` | Transforms long-form content into multiple formats: blogs → social posts → email snippets → video scripts → infographics. |
| 10 | **Click Shield** | `click_shield` | Detects and prevents click fraud on paid advertising campaigns by analyzing traffic patterns and flagging suspicious clicks. |

**Use Cases:**
- A D2C brand uses Copywriter + Social + Email Campaign to launch a product in 1 day
- An agency uses Content Repurposer to turn one blog post into 15 social posts
- A startup uses Brand Auditor before investor pitch to ensure consistent messaging

#### Marketing Intelligence Agents

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 11 | **Persona Builder** | `persona_builder` | Creates detailed buyer personas from demographics, psychographics, pain points, and buying behavior data. |
| 12 | **Campaign Strategist** | `campaign_strategist` | Designs full-funnel campaign strategies: awareness → consideration → conversion → retention with budget splits. |
| 13 | **A/B Test Advisor** | `ab_test_advisor` | Recommends what to A/B test (headlines, CTAs, images, layouts), calculates sample sizes, and interprets statistical significance. |
| 14 | **Competitor Analyst** | `competitor_analyst` | Deep-dives into competitor strategies: pricing, messaging, channel mix, content themes, and market positioning. |
| 15 | **URL Analyzer** | `url_analyzer` | Analyzes any website: SEO health, tech stack, page speed, mobile-friendliness, social signals, and competitive intelligence. |

**Use Cases:**
- A SaaS founder uses Persona Builder before launching to understand their ICP
- A marketing head uses Competitor Analyst to prepare a quarterly strategy

#### Analytics & Optimization Agents

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 16 | **Performance Reporter** | `performance_reporter` | Generates weekly/monthly marketing performance reports with KPIs, trends, and actionable recommendations. |
| 17 | **Budget Optimizer** | `budget_optimizer` | Optimizes marketing budget allocation across channels based on ROAS, CAC, and LTV projections. |
| 18 | **Anomaly Alerter** | `anomaly_alerter` | Detects unusual patterns in marketing data: sudden traffic drops, CTR spikes, cost anomalies, and conversion rate changes. |
| 19 | **ROI Calculator** | `roi_calculator` | Calculates marketing ROI across channels with attribution modeling, comparing spend vs revenue generated. |
| 20 | **Content Grader** | `content_grader` | Scores content quality on readability, SEO optimization, engagement potential, and brand voice consistency. |
| 21 | **Attribution Analyst** | `attribution_analyst` | Multi-touch attribution modeling: first-click, last-click, linear, time-decay, and data-driven models. |
| 22 | **Pricing Strategist** | `pricing_strategist` | Recommends pricing strategies: penetration, skimming, value-based, competitive; with price elasticity analysis. |

**Use Cases:**
- A CMO uses Performance Reporter for board-ready monthly reports
- An e-commerce team uses Anomaly Alerter to catch a sudden drop in conversion rate

#### Creative & Media Agents

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 23 | **Visual Designer** | `visual_designer` | Creates design briefs for banners, social graphics, and brand-consistent image specifications. |
| 24 | **Video Scriptwriter** | `video_scriptwriter` | Writes video scripts: reels, YouTube intros, ad storyboards, explainer videos, and voiceover scripts. |
| 25 | **Landing Page Builder** | `landing_page_builder` | Generates complete landing page copy and structure: hero section, benefits, social proof, FAQ, and CTA. |

#### Autonomous Operations Agents

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 26 | **Auto Publisher** | `auto_publisher` | Schedules and auto-publishes content across platforms with optimal timing based on audience activity patterns. |
| 27 | **Lead Scorer** | `lead_scorer` | Scores leads based on engagement signals, demographics, and behavior patterns to prioritize sales outreach. |
| 28 | **Chatbot Trainer** | `chatbot_trainer` | Generates training data and conversation flows for customer-facing chatbots. |

#### Reputation & Growth Agents

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 29 | **Review Reputation Manager** | `review_reputation` | Monitors and responds to online reviews (Google, Trustpilot, G2), generates professional response templates. |
| 30 | **Trend Spotter** | `trend_spotter` | Identifies trending topics, hashtags, and content formats in real-time for timely brand engagement. |
| 31 | **Funnel Analyzer** | `funnel_analyzer` | Diagnoses conversion funnel bottlenecks: where users drop off and why, with fix recommendations. |
| 32 | **Influencer Matcher** | `influencer_matcher` | Matches brands with relevant influencers based on niche, audience demographics, and engagement rates. |
| 33 | **Journey Mapper** | `journey_mapper` | Maps complete customer journeys from awareness to advocacy, identifying touchpoints and emotion curves. |

#### Smart Automation Agents

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 34 | **Auto Scheduler** | `auto_scheduler` | AI-powered content calendar that auto-schedules posts for optimal engagement times. |
| 35 | **Audience Segmenter** | `audience_segmenter` | Segments audiences by behavior, demographics, and purchase history for targeted campaigns. |
| 36 | **Churn Predictor** | `churn_predictor` | Predicts which customers are about to churn based on engagement decline and usage patterns. |

#### Platform-Specific Agents

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 37 | **Google Ads Optimizer** | `google_ads_optimizer` | Optimizes Google Ads: keyword bids, ad copy, quality score, negative keywords, and campaign structure. |
| 38 | **Meta Ads Optimizer** | `meta_ads_optimizer` | Optimizes Facebook/Instagram ads: audience targeting, creative testing, budget pacing, and lookalike audiences. |
| 39 | **LinkedIn Growth** | `linkedin_growth` | LinkedIn organic growth: post optimization, connection strategy, thought leadership content, and SSI improvement. |

**Use Cases:**
- A performance marketer uses Google Ads Optimizer + Meta Ads Optimizer to manage ₹50L/month ad spend
- A CEO uses LinkedIn Growth for personal branding and lead generation

---

### 📞 Product 2: VoiceFlow (5 agents)

> AI agents for voice, calls, and conversational intelligence.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 40 | **Cold Call Scripter** | `cold_call_scripter` | Generates cold call scripts with objection handling, hooks, and closing techniques tailored to Indian B2B sales. |
| 41 | **Call Sentiment Analyzer** | `call_sentiment_analyzer` | Analyzes call transcripts for customer sentiment: positive, negative, neutral with emotion scoring and key moments. |
| 42 | **WhatsApp Bot Builder** | `whatsapp_bot_builder` | Designs conversational WhatsApp bot flows with menu trees, quick replies, and API integration points. |
| 43 | **Call Summarizer** | `call_summarizer` | Summarizes sales/support calls into action items, key decisions, follow-ups, and customer sentiment scores. |
| 44 | **IVR Designer** | `ivr_designer` | Designs IVR menu flows with multilingual support, intelligent call routing, and customer-friendly navigation. |

**Use Cases:**
- A BPO uses Call Sentiment Analyzer to monitor 10,000 daily calls for quality
- A startup's sales team uses Cold Call Scripter for B2B outreach to Indian enterprises
- A hospital uses IVR Designer for Hindi/English patient routing

---

### 🎬 Product 3: VideoMind (5 agents)

> AI-powered video analysis, scripting, and content creation.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 45 | **YouTube Analyzer** | `youtube_analyzer` | Analyzes YouTube videos: extracts transcripts, engagement signals, SEO tags, content structure, and competitor benchmarking. |
| 46 | **Video Summarizer** | `video_summarizer` | Summarizes long videos into executive summaries, chapter markers, timestamped key moments, and TL;DR bullet points. |
| 47 | **Caption Generator** | `caption_generator` | Generates captions and subtitles in any Indian language from video transcripts. Outputs SRT/VTT formatted files. |
| 48 | **Audio Dubber** | `audio_dubber` | Translates transcripts to target Indian languages and generates TTS audio dubs. Returns translated text + base64 audio. |
| 49 | **Social Clip Maker** | `social_clip_maker` | Identifies viral-worthy moments from long videos, suggests short-form clips for Reels/Shorts with hooks and hashtags. |

**Use Cases:**
- A YouTube educator uses Video Summarizer to create chapter markers automatically
- A media house uses Audio Dubber to dub Hindi content into Tamil, Telugu, Bengali
- A content agency uses Social Clip Maker to extract 10 reels from one 30-min podcast

---

### 📚 Product 4: EdBrain (5 agents)

> AI tutoring, notes, quizzes, and learning tools.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 50 | **Note Generator** | `note_generator` | Generates structured study notes from any topic: organized sections, bullet points, key terms, and summary boxes. |
| 51 | **Key Points Extractor** | `key_points_extractor` | Extracts the most important points from text, lectures, or documents. Prioritizes exam-relevant content. |
| 52 | **Quiz Generator** | `quiz_generator` | Creates MCQ, true/false, fill-in-the-blank, and short-answer questions from any content with answer keys. |
| 53 | **Flashcard Creator** | `flashcard_creator` | Generates spaced-repetition flashcards from any content: front (question), back (answer), with difficulty tags. |
| 54 | **Lecture Planner** | `lecture_planner` | Plans complete lecture sessions: learning objectives, content flow, activities, assessments, and time allocation. |

**Use Cases:**
- A UPSC aspirant uses Note Generator + Quiz Generator for daily preparation
- A coaching institute uses Lecture Planner for standardized class delivery across branches
- A medical student uses Flashcard Creator for anatomy and pharmacology revision

---

### 💹 Product 5: FinWise (10 agents)

> Stock analysis, tax planning, and personal finance for India.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 55 | **Stock Analyzer** | `stock_analyzer` | Fundamental and technical analysis of Indian stocks: P/E ratio, debt, promoter holding, chart patterns, support/resistance. |
| 56 | **Stock Predictor** | `stock_predictor` | Uses historical patterns and market indicators to predict short-term price movements with confidence scores. |
| 57 | **Market Trend Analyzer** | `market_trend_analyzer` | Analyzes broader market trends: sectoral rotation, FII/DII flows, global cues, and macroeconomic indicators. |
| 58 | **AI Trend Tracker** | `ai_trend_tracker` | Tracks the latest AI industry trends, breakthrough research, new models, and market developments. |
| 59 | **Crypto Analyzer** | `crypto_analyzer` | Analyzes cryptocurrencies: on-chain metrics, market cap, volume, social sentiment, and regulatory impact for Indian investors. |
| 60 | **Tax Planner** | `tax_planner` | Indian income tax planning: Old vs New regime comparison, Section 80C/80D deductions, HRA, LTA, and ITR filing guidance. |
| 61 | **Loan Comparator** | `loan_comparator` | Compares home, personal, car, and education loans across Indian banks: interest rates, EMI, processing fees, foreclosure charges. |
| 62 | **Insurance Advisor** | `insurance_advisor` | Recommends health, term, motor, and travel insurance: coverage analysis, claim settlement ratios, premium comparison. |
| 63 | **SIP Calculator** | `sip_calculator` | Calculates SIP returns: CAGR projections, ELSS tax savings, goal-based investment planning, and fund comparison. |
| 64 | **Retirement Planner** | `retirement_planner` | Plans retirement corpus: NPS vs PPF vs EPF comparison, inflation-adjusted goals, withdrawal strategies, and pension planning. |

**Use Cases:**
- A salaried professional uses Tax Planner to save ₹1.5L under 80C
- A first-time investor uses SIP Calculator to plan ₹10K/month for 15 years
- A CA firm uses Stock Analyzer + Market Trend Analyzer for client advisory

---

### 🏥 Product 6: HealthMate (9 agents)

> AI health assistant: lab reports, diet, Ayurveda, fitness.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 65 | **Lab Report Interpreter** | `lab_report_interpreter` | Interprets blood tests, urine tests, and other lab reports: explains values, flags abnormalities, and suggests follow-ups. |
| 66 | **Symptom Triage** | `symptom_triage` | Assesses symptoms and classifies urgency: emergency (go to ER), urgent (see doctor today), or routine (schedule appointment). |
| 67 | **Diet Planner** | `diet_planner` | Creates personalized Indian diet plans for health goals: diabetes management, weight loss, PCOS, pregnancy, and sports. |
| 68 | **Mental Health Companion** | `mental_health_companion` | Provides empathetic mental health support: stress management, anxiety coping techniques, and guided exercises. |
| 69 | **Medicine Info** | `medicine_info` | Explains medications: uses, dosage, side effects, drug interactions, and Indian brand name equivalents. |
| 70 | **Patient Follow-Up** | `patient_followup` | Generates patient follow-up reminders, post-discharge care instructions, and medication adherence tracking. |
| 71 | **Ayurveda Advisor** | `ayurveda_advisor` | Provides Ayurvedic health guidance: dosha assessment, herbal remedies, dietary recommendations, and yoga prescriptions. |
| 72 | **Sports Nutrition** | `sports_nutrition` | Sports nutrition for Indian athletes: macros, meal timing, Indian food-based plans, WADA-compliant supplements. |
| 73 | **Fitness Coach** | `fitness_coach` | Personalized workout plans: home workouts for Indian apartments, gym programs, yoga integration, seasonal adaptation. |

**Use Cases:**
- A patient uses Lab Report Interpreter to understand their CBC report before a doctor visit
- A diabetic uses Diet Planner for a month-long Indian meal plan with glycemic index tracking
- An Ayurveda clinic uses Ayurveda Advisor for initial dosha assessment of new patients

> ⚠️ **Safety:** Health agents include medical disclaimers and always recommend consulting a qualified doctor. Symptom Triage, Mental Health Companion, and Medicine Info responses are NEVER cached.

---

### ⚖️ Product 7: LegalEase (4 agents)

> Indian legal docs, contracts, GST, and compliance.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 74 | **Contract Analyzer** | `contract_analyzer` | Reviews contracts and agreements: identifies risky clauses, missing protections, and suggests amendments. |
| 75 | **RTI Drafter** | `rti_drafter` | Drafts Right to Information applications for Indian government departments with proper format and legal backing. |
| 76 | **GST Compliance** | `gst_compliance` | GST compliance guidance: HSN codes, return filing, ITC claims, e-invoicing, and reconciliation. |
| 77 | **Legal Document Writer** | `legal_document_writer` | Drafts legal documents: NDAs, MOUs, partnership deeds, legal notices, affidavits, and power of attorney. |

**Use Cases:**
- A startup founder uses Contract Analyzer before signing a vendor agreement
- A citizen uses RTI Drafter to request information about road construction delays
- A small business uses GST Compliance for monthly GSTR-3B filing assistance

---

### 👔 Product 8: HireGenius (5 agents)

> AI-powered hiring, JDs, resume screening, and onboarding.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 78 | **Resume Screener** | `resume_screener` | Screens resumes against job requirements: skill matching, experience scoring, red flag detection, and shortlist ranking. |
| 79 | **Interview Q Generator** | `interview_q_generator` | Generates structured interview questions by role, level, and competency: technical, behavioral, and situational. |
| 80 | **JD Writer** | `jd_writer` | Writes professional job descriptions: role summary, responsibilities, requirements, culture fit, and compensation range. |
| 81 | **Salary Benchmarker** | `salary_benchmarker` | Benchmarks salaries for Indian market: CTC ranges by role, city, experience level, and industry comparisons. |
| 82 | **Onboarding Guide** | `onboarding_guide` | Creates structured onboarding plans: first day, first week, first 30/60/90 days with checklists and milestones. |

**Use Cases:**
- An HR team uses Resume Screener to filter 500 applications for a backend developer role
- A startup uses JD Writer + Salary Benchmarker to create competitive listings on Naukri
- A new hire gets a personalized 90-day onboarding plan from Onboarding Guide

---

### 🛒 Product 9: ShopBrain (5 agents)

> Product listings, pricing, reviews, and catalog management.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 83 | **Product Description Writer** | `product_description_writer` | Writes SEO-optimized product descriptions for Amazon, Flipkart, and Shopify with bullet points and keywords. |
| 84 | **Review Analyzer** | `review_analyzer` | Analyzes customer reviews: sentiment breakdown, common complaints, feature requests, and competitive insights. |
| 85 | **Dynamic Pricing** | `dynamic_pricing` | Recommends pricing strategies based on competition, demand, seasonality, and margin targets. |
| 86 | **Returns Predictor** | `returns_predictor` | Predicts return probability for products based on category, price, description quality, and customer segment. |
| 87 | **Catalog Enricher** | `catalog_enricher` | Enriches product catalogs with missing attributes, standardized categories, and enhanced descriptions. |

**Use Cases:**
- An Amazon seller uses Product Description Writer for 200 product listings
- A D2C brand uses Review Analyzer to identify the top 3 complaints across 10,000 reviews
- An e-commerce platform uses Returns Predictor to flag high-return-risk orders

---

### 🌾 Product 10: KisanAI (5 agents)

> Crop advisory, soil reports, MSP tracking for Indian farmers.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 88 | **Crop Advisor** | `crop_advisor` | Advises on crop selection, pest management, irrigation, and harvest timing based on region, soil, and season. |
| 89 | **Soil Report Interpreter** | `soil_report_interpreter` | Interprets soil test reports: pH, NPK levels, micronutrients, and recommends fertilizer schedules. |
| 90 | **Weather Planting** | `weather_planting` | Advises planting schedules based on monsoon predictions, temperature, and rainfall patterns for Indian regions. |
| 91 | **MSP Tracker** | `msp_tracker` | Tracks Minimum Support Prices for crops, nearby mandi rates, and best selling strategies for farmers. |
| 92 | **Subsidy Finder** | `subsidy_finder` | Finds eligible government subsidies and schemes: PM-KISAN, PMFBY, KCC, soil health card, and state schemes. |

**Use Cases:**
- A farmer in Punjab uses Crop Advisor for wheat rotation planning
- An FPO uses MSP Tracker to advise 500 member farmers on best selling time
- A progressive farmer uses Soil Report Interpreter after getting soil test from KVK

> 💡 **Pricing:** KisanAI is priced at ₹499/mo — the lowest — because farmers are the priority.

---

### 🏠 Product 11: PropertyGuru (4 agents)

> Property valuation, RERA check, rental yields for India.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 93 | **Property Valuator** | `property_valuator` | Estimates property values based on location, carpet area, floor, amenities, and market comparables. |
| 94 | **Rental Yield Calculator** | `rental_yield_calculator` | Calculates rental yield: gross vs net yield, rent vs EMI comparison, and investment ROI projections. |
| 95 | **RERA Compliance** | `rera_compliance` | Checks builder RERA compliance: registration status, carpet area calculations, and buyer rights. |
| 96 | **Area Comparator** | `area_comparator` | Compares localities: connectivity, infrastructure, appreciation trends, schools, hospitals, and livability index. |

**Use Cases:**
- A home buyer uses Property Valuator to check if ₹85L for a 3BHK in Pune is fair
- An NRI uses Rental Yield Calculator to evaluate investment in Hyderabad vs Bangalore
- A buyer uses RERA Compliance to verify a builder's registration before booking

---

### ✈️ Product 12: TravelBuddy (5 agents)

> Trip planning, visa guide, and budget optimization.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 97 | **Trip Planner** | `trip_planner` | Plans Indian and international trips: itineraries, budgets, best travel times, food, and booking recommendations. |
| 98 | **Visa Guide** | `visa_guide` | Guides Indian passport holders on visa requirements, documents, application process, and interview tips for any country. |
| 99 | **Travel Budget Optimizer** | `travel_budget_optimizer` | Optimizes travel budgets: finds deals, alternative routes, cost-saving tips, and free activities at destinations. |
| 100 | **Cultural Advisor** | `cultural_advisor` | Provides cultural etiquette, dress codes, tipping customs, and local phrases for international destinations. |
| 101 | **Itinerary Generator** | `itinerary_generator` | Auto-generates day-wise optimized itineraries with routes, timings, restaurant suggestions, and backup plans. |

**Use Cases:**
- A family uses Trip Planner for a 7-day Rajasthan road trip with hotel and food suggestions
- A first-time traveler uses Visa Guide for Japan tourist visa application step-by-step
- A backpacker uses Travel Budget Optimizer for a Europe trip under ₹1.5L

---

### 🚛 Product 13: LogiSmart (4 agents)

> Route optimization, shipment tracking, warehouse planning.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 102 | **Route Optimizer** | `route_optimizer` | Plans optimal delivery routes considering traffic, fuel costs, time windows, and driver constraints. |
| 103 | **Shipment Tracker** | `shipment_tracker` | Tracks shipments across carriers, predicts delays, and generates customer notifications. |
| 104 | **Warehouse Planner** | `warehouse_planner` | Designs warehouse layouts: zone planning, pick path optimization, capacity planning, and seasonal scaling. |
| 105 | **Last Mile Optimizer** | `last_mile_optimizer` | Optimizes last-mile delivery: clustering deliveries, rider assignment, and failed delivery prediction. |

**Use Cases:**
- A logistics company uses Route Optimizer for 500-vehicle fleet daily planning
- An e-commerce fulfillment center uses Last Mile Optimizer to reduce delivery failures
- A 3PL uses Warehouse Planner for a new 50,000 sq ft facility layout

---

### 🏛️ Product 14: JanSeva (4 agents)

> Scheme eligibility, complaint drafting, form filling.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 106 | **Scheme Eligibility** | `scheme_eligibility` | Checks eligibility for government schemes: PM-KISAN, Ayushman Bharat, PM Awas, Ujjwala, and state schemes. |
| 107 | **Complaint Drafter** | `complaint_drafter` | Drafts formal complaints for consumer forums, CPGRAMS, traffic violations, and municipal issues in proper format. |
| 108 | **Document Translator** | `document_translator` | Translates government documents between Hindi, English, and regional Indian languages. |
| 109 | **Form Filler** | `form_filler` | Guides users through government form filling: passport, PAN card, Aadhaar update, voter ID, and driving license. |

**Use Cases:**
- A rural citizen uses Scheme Eligibility to discover they're eligible for PM Awas Yojana
- A consumer uses Complaint Drafter to file a complaint on the National Consumer Helpline
- An elderly person uses Form Filler guidance for Aadhaar address update

> 💡 **Pricing:** JanSeva at ₹299/mo — civic service agents priced for universal access.

---

### 🎯 Product 15: SuccessIQ (5 agents)

> NPS, retention, churn prevention, and upsell intelligence.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 110 | **NPS Analyzer** | `nps_analyzer` | Analyzes Net Promoter Score surveys: categorizes promoters/detractors, identifies themes, and recommends actions. |
| 111 | **Retention Strategist** | `retention_strategist` | Creates customer retention strategies: loyalty programs, engagement campaigns, and win-back sequences. |
| 112 | **Feedback Synthesizer** | `feedback_synthesizer` | Aggregates and synthesizes customer feedback from multiple channels into actionable insight reports. |
| 113 | **Churn Reversal** | `churn_reversal` | Generates personalized save offers, escalation scripts, and win-back campaigns for at-risk customers. |
| 114 | **Upsell Advisor** | `upsell_advisor` | Identifies upsell and cross-sell opportunities based on usage patterns, tier analysis, and timing signals. |

**Use Cases:**
- A SaaS company uses NPS Analyzer to understand why enterprise customers rate them 6/10
- A subscription business uses Churn Reversal to reduce monthly churn by 30%
- A B2B sales team uses Upsell Advisor to prioritize expansion revenue opportunities

---

### ⚡ Product 16: WorkPilot (6 agents)

> Email summaries, meeting notes, invoices, reminders.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 115 | **Email Summarizer** | `email_summarizer` | Summarizes long email threads: key decisions, action items, deadlines, and who's responsible for what. |
| 116 | **Meeting Notes** | `meeting_notes` | Generates structured meeting notes from transcripts: agenda items, decisions, action items, and follow-ups. |
| 117 | **Invoice Generator** | `invoice_generator` | Creates GST-compliant invoices with automatic tax calculation, HSN codes, and professional formatting. |
| 118 | **Expense Tracker** | `expense_tracker` | Tracks and categorizes business expenses: auto-categorization, GST input claims, and monthly summaries. |
| 119 | **Daily Briefing** | `daily_briefing` | Generates personalized morning briefings: pending tasks, calendar events, market updates, and weather. |
| 120 | **Reminder Agent** | `reminder_agent` | Creates smart reminders with follow-up scheduling, escalation rules, and priority-based nudges. |

**Use Cases:**
- A freelancer uses Invoice Generator for ₹10K/month client billing with GST
- A team lead uses Meeting Notes after every standup to auto-send action items
- A solopreneur uses Daily Briefing for a 2-minute morning context download

---

### 🎙️ Product 17: CreatorStudio (5 agents)

> YouTube optimization, reels, sponsorships, monetization.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 121 | **YouTube Revenue Optimizer** | `youtube_revenue_optimizer` | Optimizes YouTube for max revenue: title formulas, thumbnail psychology, SEO tags, AdSense RPM, and upload scheduling. |
| 122 | **Reel Script Writer** | `reel_script_writer` | Writes viral scripts for Reels, YouTube Shorts, and TikTok: 1.5-second hooks, pattern interrupts, CTAs, and trending audio. |
| 123 | **Sponsorship Matcher** | `sponsorship_matcher` | Matches creators with brand sponsorships by niche, engagement rate, audience demographics, and rate card guidance. |
| 124 | **Audience Growth Strategist** | `audience_growth_strategist` | Data-driven strategies: content pillars, posting frequency, engagement tactics, cross-platform funnels, and community building. |
| 125 | **Content Monetizer** | `content_monetizer` | Monetization beyond ads: online courses, memberships, digital products, merchconuslting, and newsletter strategies for Indian creators. |

**Use Cases:**
- A YouTuber with 50K subs uses YouTube Revenue Optimizer to double RPM from ₹30 to ₹60
- An Instagram creator uses Reel Script Writer to produce 30 reels/month
- A creator with 100K followers uses Sponsorship Matcher to land ₹50K brand deals

---

### 🛡️ Product 18: CyberShield (5 agents)

> Phishing detection, compliance, threat briefings.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 126 | **Phishing Detector** | `phishing_detector` | Analyzes emails and URLs for phishing: lookalike domains, spoofed senders, UPI/KYC scams specific to India. Risk scoring 0-100. |
| 127 | **Password Auditor** | `password_auditor` | Audits password policies: strength scoring, breach detection awareness, 2FA/MFA setup, and passkey migration advice. |
| 128 | **Incident Reporter** | `incident_reporter` | Drafts CERT-In compliant cybersecurity incident reports with 6-hour reporting format and containment steps. |
| 129 | **Compliance Checker** | `compliance_checker` | Checks compliance: DPDP Act 2023, IT Act, CERT-In Directions 2022, RBI cybersecurity framework, PCI DSS, ISO 27001. |
| 130 | **Threat Briefing** | `threat_briefing` | Daily cybersecurity threat landscape briefing: active threats to Indian orgs, vulnerability advisories, CERT-In alerts. |

**Use Cases:**
- An IT admin uses Phishing Detector to analyze a suspicious "SBI KYC update" email
- A CISO uses Compliance Checker before a DPDP Act audit
- A security team uses Incident Reporter to file a CERT-In report within the mandatory 6-hour window

---

### 🚀 Product 19: LaunchPad (5 agents)

> Pitch decks, valuation, funding, cap tables.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 131 | **Pitch Deck Analyzer** | `pitch_deck_analyzer` | Scores pitch decks section-by-section (problem, solution, market, traction, team, ask) with improvement suggestions. |
| 132 | **Funding Tracker** | `funding_tracker` | Tracks Indian startup funding rounds, active VCs, sector trends, and accelerator programs (Y Combinator, Sequoia Surge). |
| 133 | **Startup Valuation** | `startup_valuation` | Estimates valuation using 5 methods: revenue multiple, comparable analysis, DCF, Berkus, and Rule of 40. |
| 134 | **Cap Table Manager** | `cap_table_manager` | Manages cap tables: ESOP pool sizing, vesting schedules, dilution modeling, and Indian ESOP taxation (Section 17(2)). |
| 135 | **Market Sizing** | `market_sizing` | Calculates TAM/SAM/SOM using top-down and bottom-up approaches with Indian market data (MOSPI, RBI, NASSCOM). |

**Use Cases:**
- A pre-seed founder uses Pitch Deck Analyzer before approaching angel investors
- A Series A startup uses Cap Table Manager to plan a 10% ESOP pool without over-dilution
- A fintech startup uses Market Sizing for a ₹2Cr seed round deck

---

### 🌱 Product 20: GreenIQ (4 agents)

> Carbon footprint, ESG reports, solar ROI, green buildings.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 136 | **Carbon Footprint** | `carbon_footprint` | Calculates carbon emissions from energy, transport, food, and waste. Suggests reduction strategies and Indian offsetting programs. |
| 137 | **ESG Report Writer** | `esg_report_writer` | Drafts ESG reports in BRSR format (SEBI mandatory for top 1000 listed companies), GRI Standards, and UN SDG mapping. |
| 138 | **Solar ROI Calculator** | `solar_roi_calculator` | Calculates rooftop solar ROI: system sizing, PM Surya Ghar Yojana subsidy (40-60%), payback period, and 25-year savings. |
| 139 | **Green Certification** | `green_certification` | Guides toward IGBC, GRIHA, LEED, and BEE certifications for buildings with checklist generation and cost-benefit analysis. |

**Use Cases:**
- A listed company uses ESG Report Writer for mandatory BRSR filing with SEBI
- A homeowner uses Solar ROI Calculator: discovers ₹3L system pays back in 3 years with government subsidy
- A real estate developer uses Green Certification for IGBC Platinum rating on new project

---

### 🏏 Product 21: SportsIQ (4 agents)

> Cricket analytics, fantasy teams, sports nutrition.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 140 | **Cricket Analyst** | `cricket_analyst` | Player performance analysis, head-to-head matchups, pitch reports, IPL auction analytics, and match predictions. |
| 141 | **Fantasy Team Builder** | `fantasy_team_builder` | Builds optimal Dream11/MPL teams considering form, venue history, credits, and captain/VC selection. |
| 142 | **Sports Nutrition** | `sports_nutrition` | Indian athlete nutrition: macros, pre/post workout meals using Indian foods, vegetarian protein optimization. |
| 143 | **Fitness Coach** | `fitness_coach` | Personalized workouts: small apartment routines, terrace workouts, gym programs, yoga integration, and seasonal plans. |

**Use Cases:**
- An IPL fan uses Cricket Analyst for CSK vs MI head-to-head stats and pitch report
- A fantasy cricket player uses Fantasy Team Builder for optimal Dream11 team with differential picks
- A gym-goer uses Fitness Coach for a home workout plan during monsoon season

---

### ⚡🚗 Product 22: EcoMotion (5 agents)

> EV comparison, charging planning, solar systems.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 144 | **EV Comparator** | `ev_comparator` | Compares Indian EVs: ARAI range vs real-world, FAME subsidy, TCO, battery warranty, state-wise subsidies (Delhi, Maharashtra, Gujarat). |
| 145 | **Charging Station Planner** | `charging_station_planner` | Plans charging network placement: traffic corridors, demand hotspots, grid capacity, BIS standards, and revenue projections. |
| 146 | **Battery Health** | `battery_health` | EV battery analysis: State of Health estimation, degradation prediction, optimal charging habits, and Indian climate impact. |
| 147 | **Solar Planner** | `solar_planner` | Designs complete home/commercial solar systems: load analysis, panel/inverter selection, net metering, and PM Surya Ghar subsidy. |

**Use Cases:**
- A buyer uses EV Comparator to decide between Tata Nexon EV and MG ZS EV
- A CPSE uses Charging Station Planner for 100 DC fast chargers on NH-48
- A homeowner uses Solar Planner: 6kWp system design with Tata Solar panels and net metering application

> ℹ️ Solar ROI Calculator from GreenIQ is also available in this product for convenience.

---

### 🔬 Product 23: DataForge (3 agents)

> Data curation, ML pipelines, dataset optimization.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 148 | **Data Curator** | `data_curator` | Guides data collection, cleaning, normalization, and labeling strategies for ML datasets. |
| 149 | **Dataset Optimizer** | `dataset_optimizer` | Optimizes datasets: balancing classes, removing duplicates, augmentation strategies, and quality scoring. |
| — | **ML Pipeline** | `ml_pipeline` | Designs end-to-end ML pipelines: feature engineering, model selection, training, evaluation, and deployment strategies. |

**Use Cases:**
- A data team uses Data Curator to standardize 500K messy customer records
- An ML engineer uses ML Pipeline to design a churn prediction model pipeline
- A startup uses Dataset Optimizer to balance an imbalanced fraud detection dataset

---

### 💒 Product 24: ShaadiAI (5 agents)

> AI-powered Indian wedding planning and management.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 150 | **Wedding Budget Planner** | `wedding_budget_planner` | Plans budgets by city tier: venue, catering, decor, photography allocation with shagun recovery estimates. |
| 151 | **Vendor Matcher** | `vendor_matcher` | Matches vendors (photographers, caterers, decorators) by budget, location, availability, and reviews. |
| 152 | **Invitation Writer** | `invitation_writer` | Wedding card text in Hindi/English/regional languages — traditional, modern, digital, and bilingual formats. |
| 153 | **Event Timeline** | `event_timeline` | Minute-by-minute event timelines: mehendi → sangeet → haldi → wedding → reception with vendor coordination. |
| 154 | **Guest List Manager** | `guest_list_manager` | Guest categorization, seating plans, RSVP tracking, dietary preferences, and return gift planning. |

**Use Cases:**
- A couple uses Wedding Budget Planner to plan a ₹15L wedding in Jaipur
- A wedding planner uses Event Timeline for a 3-day destination wedding schedule
- Parents use Invitation Writer for bilingual (Hindi + English) wedding cards

> 💡 **India Insight:** ₹5 lakh crore wedding industry — every family is a customer.

---

### 🔮 Product 25: JyotishAI (5 agents)

> Vedic astrology, Vastu, Panchang, and spiritual guidance.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 155 | **Kundli Analyzer** | `kundli_analyzer` | Birth chart analysis: planetary positions, dasha periods, yogas, Mangal Dosha, gemstone recommendations. |
| 156 | **Muhurat Finder** | `muhurat_finder` | Finds auspicious timings for marriage, griha pravesh, business launch, vehicle purchase, and ceremonies. |
| 157 | **Vastu Advisor** | `vastu_advisor` | Vastu Shastra for homes/offices: room placement, directions, colors, and no-demolition remedies. |
| 158 | **Panchang Reader** | `panchang_reader` | Daily panchang: tithi, nakshatra, yoga, karana, Rahu Kaal, festival calendar, and regional variations. |
| 159 | **Mantra Guide** | `mantra_guide` | Recommends mantras and meditation by planetary condition and life goal. Chanting count, timing, mala selection. |

**Use Cases:**
- A family uses Muhurat Finder for auspicious griha pravesh timing
- A business owner uses Vastu Advisor for office layout without structural changes
- A devotee uses Mantra Guide for Navagraha healing mantras

> ⚠️ **Disclaimer:** All astrology agents include cultural/traditional guidance disclaimers.

---

### 🍽️ Product 26: FoodBrain (5 agents)

> Recipe generation, restaurant optimization, kitchen management.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 160 | **Recipe Generator** | `recipe_generator` | Indian recipes by available ingredients, dietary restrictions, regional cuisine, cooking time, and skill level. |
| 161 | **Menu Designer** | `menu_designer` | Restaurant menu creation: pricing psychology, food cost analysis, menu engineering (Stars/Plowhorses/Dogs matrix). |
| 162 | **Food Cost Optimizer** | `food_cost_optimizer` | Portion control, waste reduction, APMC/mandi supplier comparison, seasonal ingredient substitution. |
| 163 | **Zomato Listing Optimizer** | `zomato_listing_optimizer` | Optimizes Zomato/Swiggy listings: descriptions, tags, photo guidelines, review management, promotional strategy. |
| 164 | **Kitchen Inventory** | `kitchen_inventory` | Tracks perishables, auto-reorder suggestions, FIFO enforcement, expiry alerts, and monsoon storage tips. |

**Use Cases:**
- A home cook uses Recipe Generator: "I have paneer, spinach, and cream — what can I make?"
- A restaurant owner uses Menu Designer to redesign their thali pricing
- A cloud kitchen uses Zomato Listing Optimizer to improve ratings from 3.8 to 4.3

---

### 🎮 Product 27: GameIQ (4 agents)

> Game strategies, esports analytics, stream optimization.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 165 | **Game Strategy** | `game_strategy` | BGMI, Valorant, Free Fire strategies: drop spots, weapon loadouts, team compositions, and rank climbing tips. |
| 166 | **Esports Analyst** | `esports_analyst` | Tournament analysis: player stats, meta trends, roster changes, Indian esports scene (BGMI Masters, Valorant Challengers). |
| 167 | **Stream Optimizer** | `stream_optimizer` | Twitch/YouTube Gaming optimization: OBS settings for Indian internet, engagement tactics, growth strategies. |
| 168 | **Gaming PC Builder** | `gaming_pc_builder` | PC/laptop build recommendations: ₹30K to ₹2L+ budgets with Indian retailer pricing (MD Computers, PrimeABGB). |

**Use Cases:**
- A BGMI player uses Game Strategy for Erangel drop spots and rotation strategies
- A gaming streamer uses Stream Optimizer for 720p streaming on Indian broadband
- A teen uses Gaming PC Builder for best ₹50K gaming setup

---

### 🏗️ Product 28: BuildSmart (4 agents)

> Construction estimation, interior design, material planning.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 169 | **Construction Estimator** | `construction_estimator` | Per sq ft costs by city tier and finish quality: ₹1200-3500+/sqft with phase-wise budget breakdowns. |
| 170 | **Interior Designer** | `interior_designer` | Room-by-room design: style selection, color palettes, furniture from Urban Ladder/Pepperfry/IKEA, budget tiers. |
| 171 | **Material Calculator** | `material_calculator` | Calculates cement bags, steel tonnes, bricks, sand, aggregate quantities from floor plan dimensions. |
| 172 | **Contractor Checker** | `contractor_checker` | Contractor evaluation checklists, milestone payment schedules, quality checkpoints, and dispute guidance. |

**Use Cases:**
- A homeowner uses Construction Estimator for a 1500 sqft house in Pune
- A couple uses Interior Designer for a 2BHK makeover under ₹8L
- A builder uses Material Calculator for a 4-floor residential project

---

### 👶 Product 29: ParentSquad (4 agents)

> Child milestones, school selection, vaccination, nutrition.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 173 | **Child Milestone Tracker** | `child_milestone_tracker` | Developmental milestones by age: motor, cognitive, speech with delay detection and pediatrician referral flags. |
| 174 | **School Selector** | `school_selector` | Compares CBSE/ICSE/IB/State boards: fees, results, infrastructure, admission process with RTE quota guidance. |
| 175 | **Vaccination Scheduler** | `vaccination_scheduler` | NIS + IAP vaccination schedule, catch-up plans for missed vaccines, government vs private options. |
| 176 | **Child Nutrition** | `child_nutrition` | Age-appropriate Indian meals: weaning foods, lunch box ideas, brain foods, and picky eating solutions. |

**Use Cases:**
- New parents use Child Milestone Tracker for their 6-month-old's development
- Parents use School Selector to compare 5 schools near them for Class 1 admission
- A mother uses Vaccination Scheduler to catch up on missed IAP vaccines

> ⚠️ **Safety:** Parenting health agents always recommend consulting a pediatrician.

---

### 📰 Product 30: NewsRadar (4 agents)

> Fact-checking, news digests, media monitoring, PR.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 177 | **Fake News Detector** | `fake_news_detector` | Fact-checks claims using source verification, logical analysis, and known misinformation patterns. Confidence scoring. |
| 178 | **News Summarizer** | `news_summarizer` | Daily top-10 news digest: politics, business, tech, sports with market summary (Sensex, Nifty, gold). |
| 179 | **Media Monitor** | `media_monitor` | Brand/person mention tracking: sentiment analysis, share of voice, crisis detection, and coverage reports. |
| 180 | **Press Release Writer** | `press_release_writer` | Professional press releases: AP style, inverted pyramid format, with PTI/ANI distribution guidance. |

**Use Cases:**
- A user forwards a WhatsApp message to Fake News Detector for verification
- A CEO uses News Summarizer for a 2-minute morning briefing
- A PR agency uses Media Monitor to track client mentions during a product launch

---

### 🐕 Product 31: PetPal (3 agents)

> Pet health, nutrition, and training for Indian pet owners.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 181 | **Pet Health** | `pet_health` | Symptom checker for dogs/cats: vaccination schedules, breed-specific care, monsoon/tick prevention, emergency vs routine. |
| 182 | **Pet Nutrition** | `pet_nutrition` | Diet plans: homemade Indian recipes (chicken + rice), commercial brand comparison (Royal Canin, Drools, Farmina). |
| 183 | **Pet Trainer** | `pet_trainer` | Positive reinforcement training: housetraining, commands, leash training, Indian apartment living, Indie dog care. |

**Use Cases:**
- A new puppy owner uses Pet Health for the vaccination schedule
- A Labrador owner uses Pet Nutrition for homemade diet vs Pedigree comparison
- A rescue adopter uses Pet Trainer for housetraining an Indian Pariah dog

---

### 🧓 Product 32: SeniorSafe (4 agents)

> Medication, pension, will drafting, and caregiver support.

| # | Agent | Identifier | Description |
|---|-------|-----------|-------------|
| 184 | **Elder Health Monitor** | `elder_health_monitor` | Medication schedule management, chronic condition tracking (BP, sugar), fall prevention, emergency protocols. |
| 185 | **Pension Advisor** | `pension_advisor` | EPF/EPS withdrawal, SCSS rates, PMVVY, senior citizen FD rates, Section 80TTB, and income generation strategies. |
| 186 | **Will Drafter** | `will_drafter` | Simple will drafting under Indian Succession Act 1925: asset categories, beneficiaries, witness requirements, registration. |
| 187 | **Caregiver Guide** | `caregiver_guide` | Daily care routines, physiotherapy exercises, mental stimulation, Alzheimer's/Parkinson's care, and caregiver burnout prevention. |

**Use Cases:**
- A senior citizen uses Pension Advisor to compare SCSS vs Senior Citizen FD rates
- A family uses Will Drafter to create a simple will for aging parents
- A caregiver uses Caregiver Guide for daily routines for a post-stroke patient

> 💡 **Pricing:** SeniorSafe at ₹499/mo — affordable elder care for every family.

---

## License

Proprietary — SutraCode Technologies. All rights reserved.
