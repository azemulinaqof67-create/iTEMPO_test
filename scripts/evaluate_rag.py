"""
Оценка качества AI-ассистента: релевантность ответов и скорость.

Метрики:
  - keyword_hit_rate   : доля ответов, содержащих хотя бы одно из expected_contains_any

Примечание: chat_history автоматически отключается во время eval,
чтобы не зависеть от доступности PostgreSQL.
  - avg_latency_s      : среднее время ответа в секундах
  - p50_latency_s      : медиана времени ответа
  - p95_latency_s      : 95-й перцентиль времени ответа
  - has_context_rate   : доля ответов, где RAG вернул контекст
  - empty_answer_rate  : доля пустых / «не знаю» ответов
  - avg_answer_len     : средняя длина ответа (символы)

Использование:
  uv run python scripts/evaluate_rag.py [--test-file PATH] [--out PATH] [--session-prefix STR]
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.assistant import AssistantService  # noqa: E402
from src.core.config import Config  # noqa: E402

# ---------------------------------------------------------------------------
# Логирование — только WARNING для сторонних библиотек
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
# Включаем INFO только для этого скрипта
logger = logging.getLogger("evaluate_rag")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
NEGATIVE_PHRASES = [
    "не знаю",
    "не могу ответить",
    "не нашел",
    "не нашёл",
    "информация отсутствует",
    "нет данных",
    "не удалось найти",
    "не могу помочь",
    "недостаточно информации",
    "не обладаю",
]


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    id: str
    question: str
    category: str = ""
    expected_keywords: List[str] = field(default_factory=list)
    expected_contains_any: List[str] = field(default_factory=list)
    context_hint: str = ""


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    answer: str
    latency_s: float
    has_context: bool
    keyword_hit: bool           # хотя бы 1 из expected_contains_any найден
    is_empty_answer: bool       # ответ пустой или «не знаю»
    matched_keywords: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EvalReport:
    timestamp: str
    model: str
    total_cases: int
    keyword_hit_rate: float
    avg_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    has_context_rate: float
    empty_answer_rate: float
    avg_answer_len: float
    by_category: Dict[str, Any] = field(default_factory=dict)
    cases: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def load_test_cases(path: Path) -> List[TestCase]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for item in raw:
        cases.append(
            TestCase(
                id=item.get("id", "unknown"),
                question=item["question"],
                category=item.get("category", ""),
                expected_keywords=item.get("expected_keywords", []),
                expected_contains_any=item.get("expected_contains_any", []),
                context_hint=item.get("context_hint", ""),
            )
        )
    return cases


def check_keyword_hit(answer: str, patterns: List[str]) -> tuple[bool, List[str]]:
    """Возвращает (попадание, список совпавших паттернов)."""
    answer_lower = answer.lower()
    matched = [p for p in patterns if p.lower() in answer_lower]
    return bool(matched), matched


def check_empty_answer(answer: str) -> bool:
    if not answer or len(answer.strip()) < 20:
        return True
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in NEGATIVE_PHRASES)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def format_bar(value: float, width: int = 30) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Основная логика оценки
# ---------------------------------------------------------------------------
async def run_case(
    assistant: AssistantService,
    case: TestCase,
    session_prefix: str,
) -> CaseResult:
    session_id = f"{session_prefix}_{case.id}"
    error = None
    answer = ""
    latency_s = 0.0
    has_context = False

    try:
        start = time.perf_counter()
        response = await asyncio.wait_for(
            assistant.process_text_query(
                query=case.question,
                session_id=session_id,
                platform="eval",
            ),
            timeout=60.0,
        )
        latency_s = time.perf_counter() - start

        answer = response.get("answer", "")
        context = response.get("context", []) or response.get("raw_results", [])
        has_context = bool(context)

    except asyncio.TimeoutError:
        latency_s = 60.0
        error = "TIMEOUT (>60s)"
        logger.warning(f"[{case.id}] Таймаут (>60s)")
    except Exception as exc:
        latency_s = time.perf_counter() - start if "start" in dir() else 0.0
        error = str(exc)
        logger.warning(f"[{case.id}] Ошибка: {exc}")

    keyword_hit, matched = check_keyword_hit(answer, case.expected_contains_any)
    is_empty = check_empty_answer(answer)

    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer=answer,
        latency_s=latency_s,
        has_context=has_context,
        keyword_hit=keyword_hit,
        is_empty_answer=is_empty,
        matched_keywords=matched,
        error=error,
    )


async def evaluate(
    test_file: Path,
    output_file: Optional[Path],
    session_prefix: str,
) -> EvalReport:
    logger.info(f"Загрузка тест-кейсов из {test_file}")
    cases = load_test_cases(test_file)
    if not cases:
        logger.error("Тест-кейсы не найдены.")
        sys.exit(1)

    logger.info("Инициализация AssistantService...")
    config = Config.from_env()
    config.chat_history_enabled = False
    assistant = AssistantService(config)
    await assistant.initialize()

    results: List[CaseResult] = []
    model_name = config.text_model

    total = len(cases)
    for i, case in enumerate(cases):
        # Защита от Rate Limits: пауза между запросами (бесплатный Gemini = 15 RPM)
        if i > 0:
            logger.info("Пауза 8 сек для остывания Rate Limits API...")
            await asyncio.sleep(8.0)

        logger.info(f"[{i+1}/{total}] {case.id}: {case.question[:60]}...")
        result = await run_case(assistant, case, session_prefix)

        status_icon = "✅" if result.keyword_hit and not result.is_empty_answer else (
            "⚠️" if result.is_empty_answer else "❌"
        )
        logger.info(
            f"  {status_icon} латентность={result.latency_s:.2f}s  "
            f"keyword_hit={result.keyword_hit}  "
            f"has_context={result.has_context}"
        )
        results.append(result)

    # ---------------------------------------------------------------------------
    # Агрегация метрик
    # ---------------------------------------------------------------------------
    latencies = [r.latency_s for r in results if r.error is None]
    keyword_hits = [r for r in results if r.keyword_hit]
    has_context_count = sum(1 for r in results if r.has_context)
    empty_count = sum(1 for r in results if r.is_empty_answer)
    answer_lengths = [len(r.answer) for r in results if r.answer]

    # По категориям
    by_category: Dict[str, Dict] = {}
    for r in results:
        cat = r.category or "other"
        if cat not in by_category:
            by_category[cat] = {"total": 0, "hits": 0, "latencies": []}
        by_category[cat]["total"] += 1
        if r.keyword_hit:
            by_category[cat]["hits"] += 1
        if r.error is None:
            by_category[cat]["latencies"].append(r.latency_s)

    cat_summary = {}
    for cat, data in by_category.items():
        cat_summary[cat] = {
            "total": data["total"],
            "keyword_hit_rate": data["hits"] / data["total"] if data["total"] else 0,
            "avg_latency_s": sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0,
        }

    report = EvalReport(
        timestamp=datetime.now().isoformat(),
        model=model_name,
        total_cases=total,
        keyword_hit_rate=len(keyword_hits) / total if total else 0,
        avg_latency_s=sum(latencies) / len(latencies) if latencies else 0,
        p50_latency_s=percentile(latencies, 50),
        p95_latency_s=percentile(latencies, 95),
        has_context_rate=has_context_count / total if total else 0,
        empty_answer_rate=empty_count / total if total else 0,
        avg_answer_len=sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
        by_category=cat_summary,
        cases=[asdict(r) for r in results],
    )

    return report


# ---------------------------------------------------------------------------
# Вывод отчёта
# ---------------------------------------------------------------------------
def print_report(report: EvalReport) -> None:
    import sys
    # UTF-8 stdout для Windows-консоли
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sep = "─" * 60

    print(f"\n{'═' * 60}")
    print("  📊  RAG EVALUATION REPORT")
    print(f"{'═' * 60}")
    print(f"  Время:   {report.timestamp}")
    print(f"  Модель:  {report.model}")
    print(f"  Тестов:  {report.total_cases}")
    print(sep)

    print("\n  📈 ОБЩИЕ МЕТРИКИ\n")
    metrics = [
        ("Keyword Hit Rate", report.keyword_hit_rate, True),
        ("Has Context Rate", report.has_context_rate, True),
        ("Empty Answer Rate", report.empty_answer_rate, False),
    ]
    for name, val, higher_is_better in metrics:
        bar = format_bar(val)
        emoji = "🟢" if (val >= 0.7 and higher_is_better) or (val <= 0.15 and not higher_is_better) else (
                "🟡" if (val >= 0.4 and higher_is_better) or (val <= 0.3 and not higher_is_better) else "🔴"
        )
        print(f"  {emoji} {name:<22} {val:.1%}  {bar}")

    print("\n  ⏱  СКОРОСТЬ\n")
    print(f"  {'Avg Latency':<22} {report.avg_latency_s:.2f}s")
    print(f"  {'P50 Latency':<22} {report.p50_latency_s:.2f}s")
    print(f"  {'P95 Latency':<22} {report.p95_latency_s:.2f}s")
    speed_emoji = "🟢" if report.avg_latency_s < 5 else ("🟡" if report.avg_latency_s < 15 else "🔴")
    speed_label = "Быстро" if report.avg_latency_s < 5 else ("Умеренно" if report.avg_latency_s < 15 else "Медленно")
    print(f"  {speed_emoji} Оценка скорости: {speed_label}")
    print(f"  {'Avg Answer Len':<22} {report.avg_answer_len:.0f} символов")

    print(f"\n{sep}")
    print("  📁 ПО КАТЕГОРИЯМ\n")
    for cat, data in sorted(report.by_category.items()):
        hit_bar = format_bar(data["keyword_hit_rate"], 15)
        print(
            f"  {cat:<18} hit={data['keyword_hit_rate']:.0%} {hit_bar}  "
            f"avg={data['avg_latency_s']:.1f}s  "
            f"n={data['total']}"
        )

    print(f"\n{sep}")
    print("  🔍 ДЕТАЛИ ПО КЕЙСАМ\n")
    for case in report.cases:
        ok = "✅" if case["keyword_hit"] and not case["is_empty_answer"] else (
             "⚠️" if case["is_empty_answer"] else "❌"
        )
        ctx = "📚" if case["has_context"] else "   "
        err = f" ERROR: {case['error']}" if case.get("error") else ""
        matched = ", ".join(case.get("matched_keywords", [])) or "—"
        print(
            f"  {ok} {ctx} [{case['id']}] ({case['category']})  "
            f"{case['latency_s']:.2f}s  matched=[{matched}]{err}"
        )
        answer_preview = case["answer"][:150].replace("\n", " ") if case["answer"] else "(пусто)"
        print(f"       ↳ {answer_preview}...")

    print(f"\n{'═' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка качества RAG-ассистента")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=PROJECT_ROOT / "tests" / "rag_eval.json",
        help="Путь к JSON с тест-кейсами",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Путь для сохранения JSON-отчёта (необязательно)",
    )
    parser.add_argument(
        "--session-prefix",
        type=str,
        default="eval",
        help="Префикс session_id (для изоляции истории)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if not args.test_file.exists():
        logger.error(f"Файл тест-кейсов не найден: {args.test_file}")
        print(f"\n❌ Файл тест-кейсов не найден: {args.test_file}")
        print(f'   Создайте файл tests/rag_eval.json в формате:')
        print(
            '   [{"id": "tc_01", "question": "...", '
            '"expected_contains_any": ["ключ1", "ключ2"]}]'
        )
        sys.exit(1)

    report = await evaluate(args.test_file, args.out, args.session_prefix)
    print_report(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(f"  💾 Отчёт сохранён: {args.out}\n")
    else:
        # Автосохранение рядом со скриптом
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_out = PROJECT_ROOT / "logs" / f"eval_{ts}.json"
        auto_out.parent.mkdir(parents=True, exist_ok=True)
        with auto_out.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(f"  💾 Отчёт автосохранён: {auto_out}\n")


if __name__ == "__main__":
    asyncio.run(main())
