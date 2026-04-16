# SutraAI Engine — Software Factory Architecture Plan

## Vision

The SutraAI Engine is a **Software Factory for AI Agents**. The core engine is not a collection of tools — it is a **sentient kernel** that understands context, sentiment, logic, and intent. Agents are lightweight extensions that inherit all of the kernel's intelligence and simply define *what domain* to apply it to.

> **Analogy**: The core engine is like a **human brain**. Agents are like **job titles**. A person can be a "copywriter" or a "click fraud analyst" — the brain (intelligence) is the same, only the task changes.

---

## 1. The Core Kernel — "The Brain"

The kernel is the foundation. Every agent, every API call, and every background job routes through these layers. They are **not optional** — they are the engine's nervous system.

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE SUTRA KERNEL                            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. PERCEPTION LAYER (Understand)                         │  │
│  │     Sentiment · Language · Intent · Entity Extraction     │  │
│  │     PII Detection · Toxicity · User Frustration           │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                            ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  2. MEMORY LAYER (Remember)                               │  │
│  │     Conversation History · RAG Knowledge Base             │  │
│  │     Voice Profiles · Tenant Preferences                   │  │
│  │     Semantic Cache · Cross-Tenant Learnings               │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                            ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  3. REASONING LAYER (Think)                               │  │
│  │     Smart Router · Complexity Analysis · Chain-of-Thought │  │
│  │     Quality Gate · Token Budget · Cost Optimization       │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                            ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  4. EXECUTION LAYER (Act)                                 │  │
│  │     Driver Manager · Fallback Chains · Circuit Breaker    │  │
│  │     Retry Strategy · Streaming · Batch Processing         │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                            ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  5. LEARNING LAYER (Evolve)                               │  │
│  │     OPRO Meta-Prompts · Edit Diff Analysis · A/B Testing  │  │
│  │     Feedback Loops · Anomaly Model Retraining             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### What Makes the Kernel "Smart"

| Capability | How It Works | Status |
|---|---|---|
| **Sentiment Awareness** | Detects frustrated/angry users, adjusts tone, triggers webhook alerts | ✅ Built |
| **Language Intelligence** | Auto-detects user language, responds natively | ✅ Built |
| **Contextual Memory** | RAG + Conversation History + Voice Profiles injected into every call | ✅ Built |
| **Cost-Aware Routing** | SmartRouter selects cheapest model that can handle the complexity | ✅ Built |
| **Self-Healing** | CircuitBreaker + Retry + Fallback automatically recover from failures | ✅ Built |
| **Self-Learning** | OPRO meta-prompts + edit analysis + A/B testing evolve prompts over time | ✅ Built |
| **Safety-First** | PII redaction + Moderation + Competitor Lock on every request | ✅ Built |
| **Anomaly Detection** | Isolation Forest models trained per-tenant for click fraud | ✅ Built |

---

## 2. The Agent Extension Model — "Job Titles for the Brain"

An agent is a **thin wrapper** that tells the kernel what domain to focus on. The agent itself contains almost zero logic — all intelligence comes from the kernel.

### Agent Anatomy

```python
# This is ALL an agent needs to be:
class MyAgent(BaseAgent):
    @property
    def identifier(self) -> str:
        return "my_agent"
```

```yaml
# agent_config/my_agent.yaml — The real "brain instructions"
domain: "describe the domain"
response_schema:
  - field_1
  - field_2
capabilities:
  - "What this agent can do"
extra_instructions: |
  Domain-specific rules go here.
```

### Current Agent Roster

| Agent | Domain | Kernel Features Used |
|---|---|---|
| **Copywriter** | Headlines, CTAs, body copy | Reasoning + Voice Profile |
| **SEO** | Meta tags, keywords, optimization | Reasoning + RAG |
| **Social Media** | Platform posts, hashtags, schedules | Reasoning + Voice + Image Gen |
| **Email Campaign** | Newsletters, subject lines, drips | Reasoning + Sentiment |
| **WhatsApp** | Template messages, quick replies | Reasoning + Language Detection |
| **SMS** | Short promotional messages | Reasoning + Token Budget |
| **Ad Creative** | Facebook/Google/LinkedIn ad copy | Reasoning + Voice Profile |
| **Brand Auditor** | Voice consistency, style guide adherence | Reasoning + RAG + Quality Gate |
| **Content Repurposer** | Multi-channel content adaptation | Reasoning + Voice + All Drivers |
| **Click Shield** | Ad fraud detection and reporting | Anomaly Detection + Self-Learning |

---

## 3. Future Agent Roadmap

Because the kernel handles all intelligence, shipping a new agent takes **< 1 hour**. Here is the planned expansion:

### Phase 1: Marketing Intelligence (Next)
| Agent | Domain | Unique Kernel Usage |
|---|---|---|
| **Audience Persona Builder** | ICP creation, buyer persona generation | RAG + Sentiment + Learning |
| **Campaign Strategist** | Multi-channel campaign blueprints | Reasoning (Complex tier) |
| **A/B Test Advisor** | Suggests headline/CTA variations, analyzes results | Quality Gate + A/B Testing |
| **Competitor Analyst** | Scrapes and summarizes competitor positioning | RAG + Web Crawler |

### Phase 2: Analytics & Insights
| Agent | Domain | Unique Kernel Usage |
|---|---|---|
| **Performance Reporter** | KPI dashboards, trend summaries | Reasoning + Data Aggregation |
| **Budget Optimizer** | Cross-channel budget allocation recommendations | Token Budget + Cost Awareness |
| **Anomaly Alerter** | Detects sudden traffic drops, spend spikes | Anomaly Detection + Webhooks |

### Phase 3: Creative & Media
| Agent | Domain | Unique Kernel Usage |
|---|---|---|
| **Visual Designer** | Banner, social media graphic generation | Image Generation Drivers |
| **Video Scriptwriter** | Scripts for reels, ads, explainers | Reasoning (Complex) + Voice |
| **Landing Page Builder** | Hero copy, CTA, layout suggestions | Reasoning + RAG + SEO |

### Phase 4: Autonomous Operations
| Agent | Domain | Unique Kernel Usage |
|---|---|---|
| **Auto-Publisher** | Scheduled content publishing across platforms | Execution + Webhooks |
| **Lead Scorer** | AI-driven lead quality scoring | Anomaly Detection + Learning |
| **Chatbot Trainer** | Generates FAQ responses from knowledge base | RAG + Voice + Learning |

---

## 4. The Self-Learning Flywheel

Every interaction makes the engine smarter. This is the core differentiator.

```
User Prompt → Kernel Processes → Agent Responds
                                      │
                                      ▼
                              User Feedback (👍/👎)
                                      │
                                      ▼
                              AgentFeedback Table
                                      │
                                      ▼
                           ┌──────────┴──────────┐
                           │                     │
                     OPRO Optimizer        Edit Diff Analyzer
                           │                     │
                     New System Prompt    Learned Preferences
                           │                     │
                           └──────────┬──────────┘
                                      ▼
                              A/B Test (10% traffic)
                                      │
                                      ▼
                              Performance Improved?
                                   │         │
                                  Yes        No
                                   │         │
                             Promote      Discard
                                   │
                                   ▼
                            Better Engine ─── (Cycle Repeats)
```

---

## 5. Monetization via Software Factory

The beauty of the factory model: **every new agent is a new revenue stream** with zero additional infrastructure cost.

| Tier | What They Get | Price |
|---|---|---|
| **Starter** | 3 agents (Copy, Social, Email) | ₹1,999/mo |
| **Professional** | All marketing agents + Click Shield | ₹7,999/mo |
| **Enterprise** | All agents + Custom agents + API access | ₹19,999/mo |
| **Platform** | White-label the entire engine | Custom |

### Key Insight
> New agents cost **₹0 in infrastructure** because they reuse the kernel's drivers, cache, learning pipeline, and safety systems. Every new agent is **pure margin**.

---

## 6. Implementation Status

| Layer | Components | Status |
|---|---|---|
| **Perception** | Sentiment, Language, PII, Moderation | ✅ Complete |
| **Memory** | RAG, Cache (Prompt + Semantic), Voice Profiles, History | ✅ Complete |
| **Reasoning** | SmartRouter, QualityGate, TokenBudget, Thinking | ✅ Complete |
| **Execution** | 6 Drivers, Fallback, CircuitBreaker, Retry, Streaming | ✅ Complete |
| **Learning** | OPRO, EditDiff, A/B Testing, Click Learning | ✅ Complete |
| **Agents** | 10 registered (Copy, SEO, Social, Email, WhatsApp, SMS, Ad, Brand Auditor, Content Repurposer, Click Shield) | ✅ Complete |
| **Infrastructure** | Docker, PostgreSQL, Redis, Qdrant, Celery | ✅ Complete |

---

*The SutraAI Engine is not a tool. It is a factory. You build once, ship forever.*
