"""
Token usage & cloud-LLM cost estimator for the `request_logs` table.

The local stack runs Qwen on vLLM, so it costs nothing per token. This script
answers a "what if" question: if the same traffic had been served by a hosted
model (GPT-4o or Gemini 2.5 Pro), what would the bill have been?

WHY WE TOKENIZE INSTEAD OF READING `tokens_used`:
    The `tokens_used` column is never populated — `VLLMClient` logs the vLLM
    `usage` block but `TaskProcessor.log_completion()` is called without it, so
    every row is NULL. We therefore re-tokenize the stored `input_text` and
    `output_text` with tiktoken's `o200k_base` (the GPT-4o encoding) to get an
    input/output token split that the pricing models need.

    Gemini uses a different (SentencePiece) tokenizer, so its token counts are
    an APPROXIMATION based on the GPT-4o encoding. For mostly-Arabic text the
    real Gemini count can differ by ~10-20%; treat the Gemini figure as an
    order-of-magnitude estimate, not an invoice.

Usage:
    python scripts/token_cost_report.py
    python scripts/token_cost_report.py --since 2026-05-01 --until 2026-06-01
    python scripts/token_cost_report.py --status completed
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

# Add the repo root to PYTHONPATH so `app` imports resolve when run directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psycopg2  # noqa: E402

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("o200k_base")  # GPT-4o / GPT-4.1 encoding
    _TOKENIZER = "tiktoken:o200k_base"

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))

except ImportError:  # pragma: no cover - fallback path
    _TOKENIZER = "heuristic:chars/4 (tiktoken not installed)"

    def count_tokens(text: str) -> int:
        # ~4 characters per token is the common rule-of-thumb fallback.
        return (len(text or "") + 3) // 4


# ── Pricing (USD per 1,000,000 tokens) ───────────────────────────────────────
# Verified June 2026. Gemini 2.5 Pro has tiered pricing above a 200k-token
# prompt; our prompts are tiny so the standard (<=200k) tier always applies.
# Sources:
#   https://openai.com/api/pricing/
#   https://ai.google.dev/gemini-api/docs/pricing
@dataclass(frozen=True)
class Pricing:
    name: str
    input_per_m: float
    output_per_m: float
    note: str = ""


MODELS = [
    Pricing("GPT-4o", input_per_m=2.50, output_per_m=10.00),
    Pricing(
        "Gemini 2.5 Pro",
        input_per_m=1.25,
        output_per_m=10.00,
        note="<=200k context tier; token count approximated via GPT-4o encoding",
    ),
]


@dataclass
class Bucket:
    rows: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, in_tok: int, out_tok: int) -> None:
        self.rows += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok


def pg_dsn_from_settings() -> dict:
    """Build psycopg2 connection kwargs from the app's DATABASE_URL.

    The configured URL points at `host.docker.internal` (for containers) and
    uses the asyncpg driver; from the host we rewrite it to 127.0.0.1 + libpq.
    """
    try:
        from app.config import get_settings

        url = get_settings().DATABASE_URL
    except Exception:
        url = os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://llm_user:llm_pass@host.docker.internal:5432/llm_logs",
        )

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    if host == "host.docker.internal":
        host = "127.0.0.1"
    return {
        "host": host,
        "port": parsed.port or 5432,
        "dbname": (parsed.path or "/llm_logs").lstrip("/"),
        "user": unquote(parsed.username or "llm_user"),
        "password": unquote(parsed.password or "llm_pass"),
    }


def fetch_rows(args) -> list[tuple[str, str, str]]:
    where = []
    params: list = []
    if args.status:
        where.append("status = %s")
        params.append(args.status)
    if args.since:
        where.append("created_at >= %s")
        params.append(args.since)
    if args.until:
        where.append("created_at < %s")
        params.append(args.until)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    sql = (
        "SELECT task_type, input_text, output_text "
        f"FROM request_logs{clause} ORDER BY created_at"
    )

    conn = psycopg2.connect(**pg_dsn_from_settings())
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        conn.close()


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since", help="created_at >= this ISO date/timestamp")
    p.add_argument("--until", help="created_at < this ISO date/timestamp")
    p.add_argument(
        "--status",
        help="filter by status (e.g. completed). Default: all rows with output.",
    )
    args = p.parse_args()

    rows = fetch_rows(args)

    overall = Bucket()
    per_task: dict[str, Bucket] = defaultdict(Bucket)
    skipped_no_output = 0

    for task_type, input_text, output_text in rows:
        # A row with no output never produced billable completion tokens.
        if not output_text:
            skipped_no_output += 1
        in_tok = count_tokens(input_text)
        out_tok = count_tokens(output_text)
        overall.add(in_tok, out_tok)
        per_task[task_type].add(in_tok, out_tok)

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  TOKEN USAGE & HOSTED-LLM COST ESTIMATE  --  request_logs")
    print("=" * 72)
    print(f"  Tokenizer        : {_TOKENIZER}")
    filt = []
    if args.status:
        filt.append(f"status={args.status}")
    if args.since:
        filt.append(f"since={args.since}")
    if args.until:
        filt.append(f"until={args.until}")
    print(f"  Filter           : {' '.join(filt) if filt else 'none (all rows)'}")
    print(f"  Rows analysed    : {fmt_int(overall.rows)}")
    print(f"  Rows w/o output  : {fmt_int(skipped_no_output)} (0 output tokens)")
    print()

    if overall.rows == 0:
        print("  No rows matched the filter — nothing to estimate.")
        return

    total_tok = overall.input_tokens + overall.output_tokens
    avg_in = overall.input_tokens / overall.rows
    avg_out = overall.output_tokens / overall.rows

    # ── Token summary ─────────────────────────────────────────────────────
    print("-" * 72)
    print("  TOKEN TOTALS")
    print("-" * 72)
    print(f"  Input tokens     : {fmt_int(overall.input_tokens):>18}")
    print(f"  Output tokens    : {fmt_int(overall.output_tokens):>18}")
    print(f"  Total tokens     : {fmt_int(total_tok):>18}")
    print(f"  Avg input / req  : {avg_in:>18,.1f}")
    print(f"  Avg output / req : {avg_out:>18,.1f}")
    print()

    # ── Per-task breakdown ────────────────────────────────────────────────
    print("-" * 72)
    print("  BY TASK TYPE")
    print("-" * 72)
    print(f"  {'task':<12}{'rows':>10}{'input tok':>16}{'output tok':>16}")
    for task, b in sorted(per_task.items(), key=lambda kv: -kv[1].rows):
        print(
            f"  {task:<12}{fmt_int(b.rows):>10}"
            f"{fmt_int(b.input_tokens):>16}{fmt_int(b.output_tokens):>16}"
        )
    print()

    # ── Cost comparison ───────────────────────────────────────────────────
    print("-" * 72)
    print("  ESTIMATED COST IF SERVED BY A HOSTED MODEL")
    print("-" * 72)
    print(
        f"  {'model':<16}{'input $':>12}{'output $':>12}"
        f"{'total $':>12}{'batch -50%':>14}"
    )
    for m in MODELS:
        in_cost = overall.input_tokens / 1_000_000 * m.input_per_m
        out_cost = overall.output_tokens / 1_000_000 * m.output_per_m
        total = in_cost + out_cost
        print(
            f"  {m.name:<16}{fmt_usd(in_cost):>12}{fmt_usd(out_cost):>12}"
            f"{fmt_usd(total):>12}{fmt_usd(total / 2):>14}"
        )
    print()
    print("  Rates (USD / 1M tokens):")
    for m in MODELS:
        note = f"  [{m.note}]" if m.note else ""
        print(
            f"    {m.name:<16} input {fmt_usd(m.input_per_m)} / "
            f"output {fmt_usd(m.output_per_m)}{note}"
        )
    print()
    print("  Notes:")
    print("    * Local Qwen-on-vLLM serving cost is $0/token (self-hosted).")
    print("    * 'batch -50%' = OpenAI/Google async Batch API discount.")
    print("    * Gemini counts approximated with the GPT-4o tokenizer.")
    print("=" * 72)


if __name__ == "__main__":
    main()
