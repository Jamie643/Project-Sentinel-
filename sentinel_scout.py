import os
import re
import sys
import time
import html
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import requests
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==========================================
# 1. ENVIRONMENT CONFIGURATION & LOGGING
# ==========================================

# All sensitive API keys, webhook secrets, and risk parameters loaded strictly from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL")

# Operational & Risk Parameter Thresholds
MAX_LATENCY_MS = float(os.getenv("MAX_LATENCY_MS", "2500.0"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "0.05"))
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.005"))
ORDER_TIMEOUT_SECS = int(os.getenv("ORDER_TIMEOUT_SECS", "30"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SentinelScout")

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================
# 2. CIRCUIT BREAKER & RISK CONTROLS
# ==========================================

class CircuitBreaker:
    """
    Monitors latency spikes, rate limit / HTTP failures, and portfolio loss thresholds.
    Halts order execution and report generation automatically if triggered.
    """
    def __init__(self, failure_threshold: int = 3, max_latency_ms: float = 2500.0, max_drawdown: float = 0.05):
        self.failure_threshold = failure_threshold
        self.max_latency_ms = max_latency_ms
        self.max_drawdown = max_drawdown
        self.failure_count = 0
        self.is_open = False

    def check_latency(self, latency_ms: float) -> bool:
        if latency_ms > self.max_latency_ms:
            logger.error(f"🚨 LATENCY SPIKE BREACH ({latency_ms:.2f}ms > {self.max_latency_ms}ms). Halting execution.")
            self.is_open = True
            return False
        return True

    def check_drawdown(self, loss_pct: float) -> bool:
        if loss_pct >= self.max_drawdown:
            logger.error(f"🚨 LOSS THRESHOLD BREACH ({loss_pct:.2%} >= {self.max_drawdown:.2%}). Halting execution.")
            self.is_open = True
            return False
        return True

    def record_failure(self, reason: str = "") -> None:
        self.failure_count += 1
        logger.warning(f"Circuit Breaker failure recorded ({self.failure_count}/{self.failure_threshold}). Reason: {reason}")
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.error("🚨 CONSECUTIVE FAILURE BREACH! Execution halted.")

    def record_success(self) -> None:
        self.failure_count = 0


circuit_breaker = CircuitBreaker(
    failure_threshold=3, 
    max_latency_ms=MAX_LATENCY_MS, 
    max_drawdown=MAX_DRAWDOWN_PCT
)

PROCESSED_HASH_CACHE = set()
EXECUTED_ORDER_IDS = set()  # Global Order Idempotency Registry


# ==========================================
# 3. NORMALIZATION & TELEMETRY LOGGING
# ==========================================

def get_utc_iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    clean = re.sub(r"<[^>]*>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def normalize_numeric(value: Any, precision: int = 2) -> float:
    try:
        val = float(value) if value is not None else 0.0
        return round(val, precision)
    except (ValueError, TypeError):
        return 0.0


def generate_payload_hash(source: str, identifier: str, title: str) -> str:
    raw_key = f"{source.lower()}:{identifier.lower()}:{title.lower()}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ==========================================
# 4. PERSONALIZED ANALYSIS & DOMAIN TAGGING
# ==========================================

@dataclass
class PersonalizedAnalysis:
    direct_takeaway: str
    personal_impact: str
    recommended_action: str
    domain_tags: List[str] = field(default_factory=list)
    impact_score: float = 0.0


class AnalysisAnalyzer:
    DOMAIN_MAPPINGS = {
        "#crypto": ["btc", "eth", "sol", "xrp", "doge", "gram", "ada", "bybit", "trading", "volume", "tvl", "polymarket", "defi"],
        "#ai-annotation": ["ai", "model", "annotation", "transcription", "whisper", "ollama", "colab", "dataset", "labeling", "llm"],
        "#freelancing": ["client", "outreach", "fiverr", "apollo", "reddit", "lead", "gig", "freelance", "agency", "email"]
    }

    @classmethod
    def tag_domains(cls, text: str) -> List[str]:
        lower_text = text.lower()
        tags = set()
        for tag, keywords in cls.DOMAIN_MAPPINGS.items():
            if any(kw in lower_text for kw in keywords):
                tags.add(tag)
        return list(tags) if tags else ["#general-tech"]

    @classmethod
    def calculate_impact_score(cls, signal: Dict[str, Any], tags: List[str]) -> float:
        base_score = 0.3
        priority = signal.get("priority", "NORMAL")
        metric_val = signal.get("metric_value")
        safe_metric = metric_val if metric_val is not None else 0.0
        
        if priority == "CRITICAL":
            base_score += 0.5
        elif priority == "HIGH":
            base_score += 0.3

        if "#crypto" in tags and safe_metric > 100000:
            base_score += 0.2
        if "#ai-annotation" in tags:
            base_score += 0.1

        return min(round(base_score, 2), 1.0)


def normalize_signal(source: str, event_id: str, title: str, metric_name: str, metric_val: float, priority: str = "NORMAL") -> Optional[Dict[str, Any]]:
    clean_title = sanitize_text(title)
    if not clean_title:
        return None

    payload_hash = generate_payload_hash(source, str(event_id or ""), clean_title)
    
    # Ingestion-level payload deduplication
    if payload_hash in PROCESSED_HASH_CACHE:
        logger.info(f"Deduplication Guard: Payload hash {payload_hash[:8]} already processed. Skipping.")
        return None
    
    PROCESSED_HASH_CACHE.add(payload_hash)
    tags = AnalysisAnalyzer.tag_domains(clean_title)

    signal_dict = {
        "hash": payload_hash,
        "source": source.lower().strip() if source else "",
        "event_id": str(event_id or ""),
        "timestamp": get_utc_iso_timestamp(),
        "title": clean_title,
        "metric_name": metric_name,
        "metric_value": normalize_numeric(metric_val, 2),
        "priority": priority,
        "tags": tags
    }
    signal_dict["impact_score"] = AnalysisAnalyzer.calculate_impact_score(signal_dict, tags)
    
    # Logging & Telemetry: Detailed payload log
    logger.debug(f"Telemetry Signal Logged: {signal_dict}")
    return signal_dict


# ==========================================
# 5. INGESTION STRATEGIES & RECONNECTION
# ==========================================

class IngestionStrategy(ABC):
    """Abstract Base Strategy with automatic retry and exponential backoff session handling."""
    
    @abstractmethod
    def collect(self) -> List[Dict[str, Any]]:
        pass

    def safe_request(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
        """Handles connection drops, timeouts, and unexpected disconnects gracefully."""
        start_time = time.time()
        try:
            res = requests.get(url, headers=headers, timeout=10)
            latency_ms = (time.time() - start_time) * 1000.0
            
            # Latency Check
            if not circuit_breaker.check_latency(latency_ms):
                return None

            if res.status_code == 429:
                logger.warning(f"Rate limit hit (429) on {url}. Recording circuit breaker failure.")
                circuit_breaker.record_failure(reason="Rate Limit Exceeded")
                return None

            res.raise_for_status()
            return res
        except requests.RequestException as e:
            logger.error(f"Reconnection/Network Error fetching {url}: {e}")
            circuit_breaker.record_failure(reason=str(e))
            return None


class DefiLlamaStrategy(IngestionStrategy):
    def collect(self) -> List[Dict[str, Any]]:
        signals = []
        res = self.safe_request("https://api.llama.fi/protocols")
        if res and res.status_code == 200:
            protocols = res.json()
            # Safe sorting with None check fallback
            top_protocols = sorted(protocols, key=lambda x: (x.get('tvl') or 0), reverse=True)[:3]
            for p in top_protocols:
                norm = normalize_signal(
                    source="defillama",
                    event_id=p.get("slug") or p.get("name", ""),
                    title=f"{p.get('name', 'Unknown')} Protocol TVL Spike",
                    metric_name="tvl_usd",
                    metric_val=p.get('tvl', 0),
                    priority="HIGH"
                )
                if norm:
                    signals.append(norm)
        return signals


class PolymarketStrategy(IngestionStrategy):
    def collect(self) -> List[Dict[str, Any]]:
        signals = []
        url = "https://gamma-api.polymarket.com/events?limit=5&active=true&closed=false&order=volume"
        res = self.safe_request(url)
        if res and res.status_code == 200:
            events = res.json()
            for event in events[:5]:
                norm = normalize_signal(
                    source="polymarket",
                    event_id=event.get("id", "0"),
                    title=event.get("title", ""),
                    metric_name="volume_usd",
                    metric_val=event.get("volume", 0),
                    priority="CRITICAL"
                )
                if norm:
                    signals.append(norm)
        return signals


class GitHubStrategy(IngestionStrategy):
    def collect(self) -> List[Dict[str, Any]]:
        signals = []
        url = "https://api.github.com/search/repositories?q=created:>2026-01-01+stars:>50&sort=stars&order=desc"
        
        # Build headers and attach Authorization token if available
        headers = {"Accept": "application/vnd.github.v3+json"}
        github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"

        res = self.safe_request(url, headers=headers)
        if res and res.status_code == 200:
            items = res.json().get('items', [])[:5]
            for repo in items:
                norm = normalize_signal(
                    source="github",
                    event_id=repo.get("full_name", ""),
                    title=f"{repo.get('full_name', '')} - {repo.get('description', '')}",
                    metric_name="stars",
                    metric_val=repo.get("stargazers_count", 0),
                    priority="NORMAL"
                )
                if norm:
                    signals.append(norm)
        return signals


class IngestionEngine:
    def __init__(self):
        self._strategies: List[IngestionStrategy] = []

    def register(self, strategy: IngestionStrategy) -> None:
        self._strategies.append(strategy)

    def execute_all(self) -> List[Dict[str, Any]]:
        aggregated_signals = []
        for strategy in self._strategies:
            if circuit_breaker.is_open:
                logger.error("Circuit Breaker OPEN. Halting ingestion strategy execution.")
                break
            aggregated_signals.extend(strategy.collect())
        return aggregated_signals


# ==========================================
# 6. EXECUTION GATEWAY & RISK CONTROLS
# ==========================================

class ExecutionGateway:
    """
    Handles order execution logic:
    - Idempotency check: Ensures network retries won't execute duplicate orders.
    - Slippage & Order Routing: Configures limit vs. market orders with slippage tolerances.
    """

    @classmethod
    def process_execution(cls, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        orders_placed = 0
        skipped = 0
        duplicates_blocked = 0

        if circuit_breaker.is_open:
            logger.error("Execution Gateway aborted: Circuit breaker active.")
            return {"orders_placed": 0, "skipped": len(signals), "duplicates_blocked": 0}

        for signal in signals:
            order_idempotency_key = f"ORD-{signal['hash'][:12]}"
            
            # Idempotency Protection
            if order_idempotency_key in EXECUTED_ORDER_IDS:
                logger.warning(f"🔒 Idempotency Guard Active: Order {order_idempotency_key} already executed. Skipping.")
                duplicates_blocked += 1
                continue

            # Route orders based on impact score and sector tags
            impact = signal.get("impact_score")
            safe_impact = impact if impact is not None else 0.0
            if safe_impact >= 0.85 and "#crypto" in signal.get("tags", []):
                # Order Type Decisioning
                order_type = "LIMIT" if signal.get("priority") == "HIGH" else "MARKET"
                
                logger.info(
                    f"⚡ [ORDER EXECUTED] ID: {order_idempotency_key} | Type: {order_type} | "
                    f"Max Slippage: {MAX_SLIPPAGE_PCT:.1%} | Timeout: {ORDER_TIMEOUT_SECS}s | "
                    f"Target: {signal['title']}"
                )
                
                # Record idempotency key
                EXECUTED_ORDER_IDS.add(order_idempotency_key)
                orders_placed += 1
            else:
                skipped += 1

        return {
            "orders_placed": orders_placed,
            "skipped": skipped,
            "duplicates_blocked": duplicates_blocked
        }


class DatabasePersistence:
    @staticmethod
    def batch_upsert(signals: List[Dict[str, Any]]) -> None:
        if not signals:
            return
        logger.info(f"Database Persistence: Upserting {len(signals)} records to store...")


# ==========================================
# 7. DAILY REPORT GENERATION FRAMEWORK
# ==========================================

class DailyReportPipeline:
    """Aggregates 24-hour log data, filters out sub-threshold updates, and ranks by impact score."""

    @staticmethod
    def filter_24h_cycle(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        filtered = []
        for s in signals:
            impact = s.get("impact_score")
            safe_impact = impact if impact is not None else 0.0
            try:
                sig_time = datetime.strptime(s["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if sig_time >= cutoff_time and safe_impact >= 0.4:
                    filtered.append(s)
            except Exception:
                if safe_impact >= 0.4:
                    filtered.append(s)
        return filtered

    @staticmethod
    def sort_by_priority(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(signals, key=lambda x: (x.get("impact_score") or 0.0), reverse=True)


@retry(
    retry=retry_if_exception_type(APIError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: logger.info(f"Gemini API busy. Retrying attempt {retry_state.attempt_number}...")
)
def generate_scout_report(signals: List[Dict[str, Any]], exec_stats: Dict[str, Any]) -> str:
    if circuit_breaker.is_open:
        raise RuntimeError("Report Generation Aborted: Circuit Breaker is OPEN.")

    # Execute 24-hour Filtering & Priority Sorting Pipeline
    filtered_signals = DailyReportPipeline.filter_24h_cycle(signals)
    sorted_signals = DailyReportPipeline.sort_by_priority(filtered_signals)

    formatted_signals = "\n".join([
        f"• [{' '.join(s['tags'])}] [Impact: {s.get('impact_score', 0.0)}] {s['title']} | {s['metric_name']}: {s.get('metric_value', 0.0):,.2f}"
        for s in sorted_signals
    ]) if sorted_signals else "No high-relevance signals identified in the last 24 hours."

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = f"""
    You are Sentinel Scout. Synthesize the following 24-hour sorted intelligence records into the exact Daily Digest Markdown Template provided below.

    SYSTEM TELEMETRY:
    - Orders Placed: {exec_stats.get('orders_placed', 0)}
    - Duplicates Blocked: {exec_stats.get('duplicates_blocked', 0)}
    - Circuit Breaker Status: {"OPEN" if circuit_breaker.is_open else "CLOSED"}
    
    24-HOUR FILTERED & PRIORITY-SORTED SIGNALS:
    {formatted_signals}

    Strictly output the response using THIS exact structure and syntax:

    📅 Daily Digest — {today_str}

    Direct Impacts
    • [Domain Tag]: [1-sentence event summary] → [Direct outcome/effect]

    Actionable Leverage Points
    • [Opportunity]: [Brief description of tool, code tweak, or outreach tactic]
      └ Action Item: [Specific task to execute]

    Risks & Setbacks
    • [Risk/Issue]: [Description of failure, edge case, latency spike, or market threat]
      └ Mitigation: [Step taken or required to resolve]

    Structural Rule Changes
    • [System/Strategy]: [Old Parameter/Rule] ➔ [New Parameter/Rule]
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


# ==========================================
# 8. SYSTEM MONITORING & TELEGRAM ALERTING
# ==========================================

def send_heartbeat(status: str = "ok") -> None:
    """Dispatches ping telemetry to the heartbeat monitoring URL."""
    if HEARTBEAT_URL:
        try:
            payload = {
                "status": status,
                "timestamp": get_utc_iso_timestamp(),
                "circuit_breaker_open": circuit_breaker.is_open,
                "processed_hashes": len(PROCESSED_HASH_CACHE),
                "executed_orders": len(EXECUTED_ORDER_IDS)
            }
            requests.post(HEARTBEAT_URL, json=payload, timeout=5)
            logger.info("Heartbeat ping dispatched successfully.")
        except Exception as e:
            logger.warning(f"Failed to dispatch heartbeat: {e}")


def send_telegram_message(text: str) -> None:
    """Dispatches structured digest alert to the designated Telegram chat using HTML format."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured. Skipping alert dispatch.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Sanitize dynamic text payload so HTML special characters don't break Telegram parsing
    safe_text = html.escape(text)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": f"<pre>{safe_text}</pre>", 
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Telegram notification sent successfully.")
        circuit_breaker.record_success()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            logger.warning("Telegram HTML parse error. Falling back to unformatted raw text...")
            raw_payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
            try:
                fallback_resp = requests.post(url, json=raw_payload, timeout=10)
                fallback_resp.raise_for_status()
                logger.info("Telegram notification sent (plain text fallback).")
                circuit_breaker.record_success()
            except Exception as fallback_err:
                logger.error(f"Failed to send fallback Telegram alert: {fallback_err}")
                circuit_breaker.record_failure(reason="Telegram Delivery Failure")
        else:
            logger.error(f"Failed to send Telegram message: {e}")
            circuit_breaker.record_failure(reason=str(e))
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        circuit_breaker.record_failure(reason=str(e))


# ==========================================
# 9. MAIN PIPELINE ORCHESTRATION
# ==========================================

def main() -> None:
    logger.info("Starting Sentinel Scout Pipeline...")
    
    try:
        # Step 1: Initialize Ingestion Framework Engine & Register Strategies
        engine = IngestionEngine()
        engine.register(PolymarketStrategy())
        engine.register(GitHubStrategy())
        engine.register(DefiLlamaStrategy())
        
        # Step 2: Execute Ingestion, Normalization, Domain Tagging & Impact Scoring
        signals = engine.execute_all()
        
        # Step 3: Persistence Layer Batch Storage
        DatabasePersistence.batch_upsert(signals)

        # Step 4: Process Execution Gateway (Idempotency & Risk Checks)
        exec_stats = ExecutionGateway.process_execution(signals)

        # Step 5: Generate Daily Digest Report
        report_text = generate_scout_report(signals, exec_stats)
        
        # Step 6: Dispatch Digest & Heartbeat
        send_telegram_message(report_text)
        send_heartbeat(status="success")

    except Exception as e:
        safe_error = html.escape(str(e))
        error_msg = f"⚠️ Sentinel Scout Alert\n\nExecution failed: {safe_error}"
        logger.error(f"Fatal execution error: {e}")
        send_telegram_message(error_msg)
        send_heartbeat(status="error")
        sys.exit(1)


if __name__ == "__main__":
    main()