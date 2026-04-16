"""
Memory — Unified storage, cache, RAG, and knowledge engine.

Software Factory Principle: One file for all data persistence.
Everything is config-driven via intelligence_config.yaml.

Absorbs: cache_engine, agent_memory, brand_knowledge, knowledge_graph,
         agentic_rag, pageindex_retriever, web_search, web_scanner,
         web_scraper, training_collector, feedback_collector, click_scorer
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)

# ─── Config Loader ─────────────────────────────────────────────

_cfg_cache: dict | None = None
_cfg_ts: float = 0


def _load_cfg() -> dict:
    global _cfg_cache, _cfg_ts
    now = time.monotonic()
    if _cfg_cache is not None and (now - _cfg_ts) < 60.0:
        return _cfg_cache
    path = Path("intelligence_config.yaml")
    _cfg_cache = yaml.safe_load(open(path)) if path.exists() else {}
    _cfg_ts = now
    return _cfg_cache


def _sec(section: str, default=None):
    return _load_cfg().get(section, default or {})


# ─── ChromaDB Client (lazy singleton) ─────────────────────────

_chroma = None


def _get_chroma():
    global _chroma
    if _chroma is None:
        try:
            import chromadb
            from urllib.parse import urlparse
            s = get_settings()
            parsed = urlparse(s.chromadb_url)
            _chroma = chromadb.HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000)
        except Exception as e:
            logger.warning(f"Memory: ChromaDB unavailable: {e}")
    return _chroma


# ─── Memory: The Unified Storage Engine ───────────────────────

class Memory:
    """
    Config-driven storage, retrieval, and knowledge engine.

    Modules:
    1. Cache      — exact/hash/semantic caching (from cache_engine)
    2. RAG        — ChromaDB vector retrieval + PageIndex reasoning (from agentic_rag, pageindex_retriever)
    3. Brand KB   — per-brand knowledge base (from brand_knowledge)
    4. Agent mem  — per-agent memory for self-learning (from agent_memory)
    5. Web intel  — real-time web search (from web_search)
    6. Feedback   — user feedback collection (from feedback_collector)
    7. Training   — JSONL export for LoRA fine-tuning (from training_collector)
    """

    def __init__(self):
        self._http = httpx.AsyncClient(timeout=10, follow_redirects=True)
        self._training_dir = Path("training/data")
        try:
            self._training_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning("Memory: training/data not writable (non-critical, skipping)")
            self._training_dir = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. CACHE (absorbs cache_engine.py)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def cache_get(self, tenant_id: int, agent_type: str, prompt: str, strategy: str = "cascade") -> dict | None:
        """Get cached response strictly scoped to the tenant. Strategy: 'exact', 'hash', 'semantic', or 'cascade'."""
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
        except Exception:
            return None

        cfg = _sec("cache", {})
        if not cfg.get("enabled", True):
            return None

        if strategy == "cascade":
            for s in ["exact", "hash"]:
                result = await self._cache_strategy(redis, s, tenant_id, agent_type, prompt, cfg)
                if result:
                    return result
            return None

        return await self._cache_strategy(redis, strategy, tenant_id, agent_type, prompt, cfg)

    async def cache_put(self, tenant_id: int, agent_type: str, prompt: str, response: str, strategy: str = "hash", ttl_override: int | None = None) -> None:
        """Store response in cache isolated by tenant_id."""
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
        except Exception:
            return

        cfg = _sec("cache", {})
        # Flexible TTL (default 24h as requested)
        ttl = ttl_override or cfg.get("ttl_seconds", 86400)

        if strategy == "exact":
            key = f"sutra:cache:t{tenant_id}:exact:{agent_type}:{prompt[:200]}"
        else:
            h = hashlib.sha256(f"{agent_type}:{prompt}".encode()).hexdigest()[:16]
            key = f"sutra:cache:t{tenant_id}:hash:{h}"

        try:
            await redis.setex(key, ttl, json.dumps({"response": response, "ts": time.time()}))
        except Exception as e:
            logger.debug(f"Memory.cache_put: {e}")

    async def _cache_strategy(self, redis, strategy: str, tenant_id: int, agent_type: str, prompt: str, cfg: dict) -> dict | None:
        try:
            if strategy == "exact":
                key = f"sutra:cache:t{tenant_id}:exact:{agent_type}:{prompt[:200]}"
            else:
                h = hashlib.sha256(f"{agent_type}:{prompt}".encode()).hexdigest()[:16]
                key = f"sutra:cache:t{tenant_id}:hash:{h}"
            data = await redis.get(key)
            if data:
                parsed = json.loads(data)
                age = time.time() - parsed.get("ts", 0)
                # Valid for up to 24h by default unless overridden
                if age < cfg.get("ttl_seconds", 86400):
                    logger.info(f"Memory: cache HIT ({strategy}) for t{tenant_id}:{agent_type}")
                    return parsed
        except Exception:
            pass
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1.5 EARCON CACHE (Edge-TTS Background Hydration)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_voice_earcon(self, tenant_id: int, phase: str, language_code: str = "en", requested_voice: str | None = None) -> bytes | None:
        """
        Retrieves pre-generated conversational fillers (Earcons).
        If not in Redis, generates it via Edge-TTS and caches it permanently.
        """
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
        except Exception:
            return None

        # Normalize regional fallback
        supported_langs = ["en", "hi", "mr", "te", "ta", "bn", "gu", "kn"]
        lang = language_code if language_code in supported_langs else "hi"  # fallback

        # 1. Resolve exact Voice ID using the global router
        from app.services.voice.router import get_voice_router
        voice_id = get_voice_router().route(text="test", requested_voice=requested_voice, tenant_slug=str(tenant_id)).get("voice_id", "hi-IN-SwaraNeural")

        key = f"sutra:cache:t{tenant_id}:earcon:{phase}:{lang}:{voice_id}"
        
        # 2. Check Redis L1
        try:
            data = await redis.get(key)
            if data:
                return data
        except Exception:
            pass

        # 3. Cache Miss -> Hydrate via Edge-TTS
        import edge_tts
        import io
        from app.services.intelligence.brain import _cfg

        phrases = _cfg("voice", default={}).get("realtime", {}).get("earcons", {})
        text = phrases.get(phase, {}).get(lang) or phrases.get(phase, {}).get("en")

        if not text:
            return None

        try:
            communicate = edge_tts.Communicate(text, voice_id)
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            
            raw_bytes = audio_buffer.getvalue()
            if len(raw_bytes) > 0:
                # Permanent TTL for system Earcons
                await redis.set(key, raw_bytes)
                logger.info(f"🎤 Earcon Hydrated: [{phase}:{lang}] -> Redis")
                return raw_bytes
        except Exception as e:
            logger.warning(f"Memory.get_voice_earcon failed: {e}")
        
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. AGENT MEMORY (absorbs agent_memory.py)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def remember(self, agent_type: str, prompt: str, response: str, quality_score: float = 0.8) -> None:
        """Store a prompt-response pair in ChromaDB for future recall."""
        client = _get_chroma()
        if not client:
            return
        try:
            coll = client.get_or_create_collection(name=f"agent_memory_{agent_type}")
            doc_id = hashlib.md5(f"{prompt}:{response[:100]}".encode()).hexdigest()[:16]
            coll.upsert(
                ids=[f"mem_{doc_id}"],
                documents=[f"Q: {prompt}\nA: {response[:2000]}"],
                metadatas=[{"agent_type": agent_type, "quality": quality_score,
                            "ts": datetime.now(timezone.utc).isoformat()}],
            )
        except Exception as e:
            logger.warning(f"Memory.remember: {e}")

    async def recall(self, agent_type: str, prompt: str, n: int = 5) -> list[dict]:
        """Recall similar past interactions for an agent. Applies Auto-Cut pruning."""
        client = _get_chroma()
        if not client:
            return []
        try:
            coll = client.get_or_create_collection(name=f"agent_memory_{agent_type}")
            if coll.count() == 0:
                return []
            # Fetch more than needed, then prune
            fetch_n = min(n * 3, coll.count(), 20)
            results = coll.query(query_texts=[prompt], n_results=fetch_n)
            if not results["documents"] or not results["documents"][0]:
                return []
            # Build raw results with scores
            raw = []
            distances = results.get("distances", [[]])[0]
            for i, doc in enumerate(results["documents"][0]):
                score = max(0.0, 1.0 - distances[i]) if i < len(distances) else 0.0
                raw.append({"content": doc, "meta": results["metadatas"][0][i], "score": score})
            # Apply Auto-Cut
            return self._auto_cut(raw, n)
        except Exception as e:
            logger.debug(f"Memory.recall: {e}")
            return []

    def _auto_cut(self, results: list[dict], max_chunks: int = 5) -> list[dict]:
        """
        RAG Auto-Cut: prune irrelevant and redundant chunks.

        1. Drop chunks below similarity threshold (YAML: rag.auto_cut_threshold)
        2. Remove overlapping/duplicate content
        3. Keep only top-N most relevant
        """
        cfg = _sec("rag", {})
        threshold = cfg.get("auto_cut_threshold", 0.55)
        max_n = cfg.get("max_chunks", max_chunks)

        # 1. Threshold filter
        filtered = [r for r in results if r.get("score", 0) >= threshold]
        if not filtered:
            # Fallback: return best single result if all below threshold
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:1] if results else []

        # 2. Sort by relevance
        filtered.sort(key=lambda x: x["score"], reverse=True)

        # 3. Deduplicate overlapping content
        if cfg.get("overlap_dedup", True):
            filtered = self._dedup_chunks(filtered)

        return filtered[:max_n]

    def _dedup_chunks(self, chunks: list[dict]) -> list[dict]:
        """Remove chunks with >60% content overlap (keeps higher-scored one)."""
        kept = []
        for chunk in chunks:
            content = chunk.get("content", "")
            words = set(content.lower().split())
            is_dup = False
            for existing in kept:
                existing_words = set(existing.get("content", "").lower().split())
                if not words or not existing_words:
                    continue
                overlap = len(words & existing_words) / min(len(words), len(existing_words))
                if overlap > 0.6:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(chunk)
        return kept

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. BRAND KNOWLEDGE (absorbs brand_knowledge.py)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def brand_search(self, brand_id: str, query: str, n: int = 3) -> dict:
        """Search brand's knowledge base."""
        client = _get_chroma()
        if not client:
            return {"found": False, "confidence": 0.0, "context": ""}
        try:
            coll = client.get_or_create_collection(name=f"brand_{brand_id}_knowledge")
            if coll.count() == 0:
                return {"found": False, "confidence": 0.0, "context": ""}
            results = coll.query(query_texts=[query], n_results=min(n, coll.count()))
            if not results["documents"] or not results["documents"][0]:
                return {"found": False, "confidence": 0.0, "context": ""}
            distances = results.get("distances", [[1.0]])[0]
            confidence = max(0, 1 - min(distances))
            context = "\n\n".join(
                f"[{results['metadatas'][0][i].get('source', 'kb')}] {doc}"
                for i, doc in enumerate(results["documents"][0])
            )
            return {"found": True, "confidence": round(confidence, 3), "context": context}
        except Exception as e:
            logger.warning(f"Memory.brand_search: {e}")
            return {"found": False, "confidence": 0.0, "context": ""}

    async def brand_learn(self, brand_id: str, question: str, answer: str, source: str = "owner_answer") -> bool:
        """Store Q&A in brand's knowledge base."""
        client = _get_chroma()
        if not client:
            return False
        try:
            coll = client.get_or_create_collection(name=f"brand_{brand_id}_knowledge")
            doc_id = hashlib.md5(f"{question}:{answer}".encode()).hexdigest()[:16]
            coll.upsert(
                ids=[f"kb_{doc_id}"],
                documents=[f"Q: {question}\nA: {answer}"],
                metadatas=[{"question": question[:500], "answer": answer[:2000], "source": source,
                            "brand_id": brand_id, "ts": datetime.now(timezone.utc).isoformat()}],
            )
            return True
        except Exception as e:
            logger.warning(f"Memory.brand_learn: {e}")
            return False

    async def brand_import_faq(self, brand_id: str, items: list[dict]) -> dict:
        """Bulk import FAQ items."""
        ok = sum([await self.brand_learn(brand_id, i.get("question", ""), i.get("answer", ""), "faq_import") for i in items])
        return {"imported": ok, "total": len(items)}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. WEB SEARCH (absorbs web_search.py)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def web_search(self, query: str, max_results: int = 5) -> dict:
        """Search via Tavily (primary) or DuckDuckGo (fallback)."""
        s = get_settings()
        if s.tavily_api_key:
            result = await self._tavily(query, max_results, s.tavily_api_key)
            if result["results"]:
                return result
        result = await self._ddg(query, max_results)
        return result if result["results"] else {"results": [], "answer": "", "source": "none"}

    async def _tavily(self, q: str, n: int, key: str) -> dict:
        try:
            r = await self._http.post("https://api.tavily.com/search", json={
                "api_key": key, "query": q, "max_results": n, "include_answer": True, "search_depth": "basic"
            }, timeout=3.0)
            if r.status_code != 200:
                return {"results": [], "answer": "", "source": "tavily"}
            data = r.json()
            return {
                "results": [{"title": i.get("title", ""), "snippet": i.get("content", "")[:500], "url": i.get("url", "")} for i in data.get("results", [])[:n]],
                "answer": data.get("answer", ""), "source": "tavily",
            }
        except Exception:
            return {"results": [], "answer": "", "source": "tavily"}

    async def _ddg(self, q: str, n: int) -> dict:
        try:
            r = await self._http.get("https://api.duckduckgo.com/", params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1})
            if r.status_code != 200:
                return {"results": [], "answer": "", "source": "duckduckgo"}
            data = r.json()
            results = []
            if data.get("AbstractText"):
                results.append({"title": data.get("Heading", ""), "snippet": data["AbstractText"][:500], "url": data.get("AbstractURL", "")})
            for t in data.get("RelatedTopics", [])[:n - len(results)]:
                if isinstance(t, dict) and t.get("Text"):
                    results.append({"title": t["Text"][:100], "snippet": t["Text"][:500], "url": t.get("FirstURL", "")})
            return {"results": results, "answer": data.get("AbstractText", ""), "source": "duckduckgo"}
        except Exception:
            return {"results": [], "answer": "", "source": "duckduckgo"}

    def should_search(self, prompt: str) -> bool:
        """Heuristic: does this prompt need real-time web data?"""
        lower = prompt.lower().strip()
        if len(lower.split()) < 4:
            return False
        greetings = {"hi", "hello", "hey", "good morning", "thanks", "bye", "ok"}
        if lower.rstrip("?!., ") in greetings:
            return False
        search_signals = {"latest", "current", "today", "2024", "2025", "2026", "news", "price", "stock", "market",
                          "who is", "what is", "how to", "syllabus", "exam", "result", "compare", "vs", "best", "top"}
        no_search = {"generate", "create", "write", "quiz", "flashcard", "code", "translate"}
        has_search = any(s in lower for s in search_signals)
        has_no = any(s in lower for s in no_search)
        if has_search and not has_no:
            return True
        return prompt.strip().endswith("?") and not has_no and len(lower.split()) >= 5

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. RAG RETRIEVAL (absorbs agentic_rag.py + pageindex_retriever.py)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def retrieve(self, query: str, collections: list[str] | None = None, n: int = 5) -> list[dict]:
        """Retrieve relevant context from ChromaDB collections."""
        client = _get_chroma()
        if not client:
            return []
        cfg = _sec("agentic_rag", {})
        target_colls = collections or cfg.get("collections", ["web_articles", "agent_responses"])
        threshold = cfg.get("min_relevance_threshold", 0.5)
        all_results = []
        for coll_name in target_colls:
            try:
                coll = client.get_or_create_collection(name=coll_name)
                if coll.count() == 0:
                    continue
                results = coll.query(query_texts=[query], n_results=min(n, coll.count()))
                for i, doc in enumerate(results.get("documents", [[]])[0]):
                    dist = results.get("distances", [[1.0]])[0][i] if results.get("distances") else 1.0
                    relevance = max(0, 1 - dist)
                    if relevance >= threshold:
                        all_results.append({"content": doc, "collection": coll_name, "relevance": round(relevance, 3),
                                            "meta": results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}})
            except Exception as e:
                logger.debug(f"Memory.retrieve: {coll_name} skipped: {e}")
        all_results.sort(key=lambda x: x["relevance"], reverse=True)
        return all_results[:n]

    async def index_document(self, filepath: str, collection: str = "documents") -> dict:
        """Index a document (PDF/MD) via PageIndex tree reasoning if available."""
        try:
            from pageindex import PageIndex
            cfg = _sec("agentic_rag", {}).get("pageindex", {})
            pi = PageIndex(
                model=cfg.get("model", "gpt-4o-mini"),
                max_pages_per_node=cfg.get("max_pages_per_node", 10),
            )
            p = Path(filepath)
            if p.suffix.lower() == ".pdf":
                tree = pi.index_pdf(str(p))
            elif p.suffix.lower() in (".md", ".markdown"):
                tree = pi.index_markdown(str(p))
            else:
                return {"indexed": False, "reason": f"unsupported: {p.suffix}"}
            return {"indexed": True, "tree": tree, "nodes": self._count_nodes(tree)}
        except ImportError:
            logger.info("Memory: pageindex not installed, using vector-only")
            return {"indexed": False, "reason": "pageindex not installed"}
        except Exception as e:
            return {"indexed": False, "reason": str(e)}

    def _count_nodes(self, tree: dict) -> int:
        return 1 + sum(self._count_nodes(c) for c in tree.get("nodes", []))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. FEEDBACK + TRAINING (absorbs feedback_collector + training_collector)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def record_feedback(self, db, agent_type: str, prompt: str, response: str,
                              is_positive: bool, quality_score: float | None = None,
                              user_id: str | None = None, brand_id: str | None = None) -> None:
        """Store user feedback (thumbs up/down) in PostgreSQL."""
        try:
            from app.services.intelligence.feedback_collector import AgentFeedback
            fb = AgentFeedback(
                agent_type=agent_type, prompt=prompt, response=response,
                is_positive=is_positive, quality_score=quality_score,
                user_id=user_id, brand_id=brand_id,
            )
            db.add(fb)
            await db.commit()
            logger.info(f"Memory: {'👍' if is_positive else '👎'} recorded for {agent_type}")
        except Exception as e:
            logger.warning(f"Memory.record_feedback: {e}")

    def save_training_example(self, agent_type: str, prompt: str, response: str, quality: float = 0.8) -> None:
        """Append a training example as JSONL for LoRA fine-tuning."""
        try:
            path = self._training_dir / f"training_{agent_type}.jsonl"
            example = {
                "messages": [
                    {"role": "system", "content": f"You are the {agent_type} agent."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "quality": quality, "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(path, "a") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Memory.save_training: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. WEB INTELLIGENCE & SCRAPING (absorbs web_scanner + web_scraper)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def web_scan(self, url: str) -> dict:
        """Deep scan a URL for intelligence (summary, sentiment, keywords)."""
        content = await self.scrape(url)
        if not content:
            return {"url": url, "status": "failed"}
        try:
            from app.services.intelligence.brain import get_brain
            brain = get_brain()
            # Simple AI summary via Brain
            resp = await brain.execute(
                prompt=f"Summarize this content and extract 5 keywords:\n\n{content[:4000]}",
                system_prompt="You are a Web Intelligence Analyst. Return JSON: {summary, keywords: []}",
                agent_type="web_analyst",
            )
            data = brain.filter_response(resp.content).data
            return {"url": url, "status": "success", **data}
        except Exception as e:
            logger.warning(f"Memory.web_scan: {e}")
            return {"url": url, "status": "error", "message": str(e)}

    async def full_scan(self) -> dict:
        """Execute a full intelligence scan across multiple sources."""
        # This replaces the logic in web_scanner.py
        sources = _sec("web_intelligence", {}).get("sources", ["https://news.ycombinator.com", "https://openai.com/blog"])
        articles = 0
        for url in sources:
            res = await self.web_scan(url)
            if res.get("status") == "success":
                articles += 1
                # Index in ChromaDB
                await self.remember("web_intel", f"Context from {url}", json.dumps(res))
        return {"articles": articles, "stocks": 0, "crypto": 0, "errors": []}

    async def scrape(self, url: str) -> str:
        """Scrape text content from a URL, cleaning HTML boilerplate."""
        try:
            r = await self._http.get(url)
            r.raise_for_status()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
            return " ".join(text.split())[:10000]
        except Exception as e:
            logger.warning(f"Memory.scrape: {e}")
            return ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. CLICK SCORING & TRAINING DATA EXPORT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def score_click(self, agent_type: str, click_type: str, context: dict) -> float:
        """Score the importance of a user click for agent learning."""
        weights = _sec("click_scorer", {}).get("weights", {
            "suggestion_click": 0.8, "copy_click": 0.5, "link_click": 0.3
        })
        base = weights.get(click_type, 0.1)
        # Adjust by context (e.g. dwell time, agent relevance)
        return round(base * context.get("multiplier", 1.0), 2)

    async def export_jsonl(self, since: datetime = None) -> dict:
        """Export feedback-rated interactions as JSONL."""
        # In a real app, this would query Postgres via SQLAlchemy.
        # For the engine cleanup, we use the local training_dir as the primary store.
        files = list(self._training_dir.glob("*.jsonl"))
        return {
            "total_examples": len(files),
            "path": str(self._training_dir),
            "by_agent": {f.stem: 1 for f in files}
        }

    async def get_quality(self, agent_type: str) -> dict:
        """Get rolling quality metrics from Redis-backed Guardian scores."""
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
            key = f"sutra:quality:{agent_type}"
            vals = await redis.lrange(key, 0, -1)
            scores = [float(v) for v in vals if str(v).strip()]
            if not scores:
                return {"agent_type": agent_type, "status": "no_data", "samples": 0}
            avg = sum(scores) / len(scores)
            return {
                "agent_type": agent_type,
                "status": "ok",
                "samples": len(scores),
                "avg_score": round(avg, 2),
                "min_score": round(min(scores), 2),
                "max_score": round(max(scores), 2),
                "last_score": round(scores[0], 2),
            }
        except Exception as e:
            logger.warning(f"Memory.get_quality: {e}")
            return {"agent_type": agent_type, "status": "unavailable", "samples": 0}

    async def get_all_quality(self) -> list[dict]:
        """Get rolling quality metrics for all agents from Redis keys."""
        try:
            from app.services.connectivity.webhooks import get_redis
            redis = get_redis()
            keys = await redis.keys("sutra:quality:*")
            agents = sorted({k.split(":")[-1] for k in keys if ":" in k})
            out = []
            for aid in agents:
                out.append(await self.get_quality(aid))
            return out
        except Exception as e:
            logger.warning(f"Memory.get_all_quality: {e}")
            return []

    async def close(self):
        await self._http.aclose()


# ─── Singleton ──────────────────────────────────────────────────

_memory: Memory | None = None
_memory_lock = threading.Lock()


def get_memory() -> Memory:
    global _memory
    if _memory is None:
        with _memory_lock:
            if _memory is None:
                _memory = Memory()
    return _memory
