"""
Оценка качества RAG на наборе тестовых вопросов.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.assistant.assistant import AssistantService  # noqa: E402
from src.core.config import Config  # noqa: E402


def load_test_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


async def main():
    config = Config.from_env()
    assistant = AssistantService(config)

    test_path = Path("tests") / "rag_eval.json"
    if not test_path.exists():
        print("Файл tests/rag_eval.json не найден.")
        print('Создайте файл со списком тестов: [{"question": "...", "sources": ["..."]}]')
        return

    test_cases = load_test_cases(test_path)
    if not test_cases:
        print("Тестовые кейсы пустые.")
        return

    metrics = {"recall@5": 0, "recall@10": 0, "mrr": 0, "avg_latency": 0}

    for case in test_cases:
        question = case.get("question", "")
        expected_sources = set(case.get("sources", []))

        start = time.time()
        response = await assistant.process_text_query(question)
        latency = time.time() - start

        documents = response.get("documents", [])
        retrieved_sources = [d.get("source", "") for d in documents]

        metrics["avg_latency"] += latency

        def hit_at_k(k: int, expected: set, retrieved: list) -> bool:
            return any(src in expected for src in retrieved[:k])

        if hit_at_k(5, expected_sources, retrieved_sources):
            metrics["recall@5"] += 1
        if hit_at_k(10, expected_sources, retrieved_sources):
            metrics["recall@10"] += 1

        # MRR
        rank = 0
        for idx, src in enumerate(retrieved_sources, start=1):
            if src in expected_sources:
                rank = idx
                break
        if rank > 0:
            metrics["mrr"] += 1.0 / rank

    total = len(test_cases)
    metrics["recall@5"] /= total
    metrics["recall@10"] /= total
    metrics["mrr"] /= total
    metrics["avg_latency"] /= total

    print("RAG Evaluation результаты:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
