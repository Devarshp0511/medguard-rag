"""
Runs the hand-curated test set through the full RAG pipeline and scores
the results using Ragas across four dimensions:

  faithfulness         -- does the answer stick to the retrieved evidence,
                          or invent claims not in it? (catches hallucination)
  answer_relevancy      -- does the answer actually address the question?
  context_precision     -- of the chunks retrieved, how many are relevant?
  context_recall        -- did we retrieve everything needed to answer?

Faithfulness + answer_relevancy diagnose the GENERATION half.
Context precision + recall diagnose the RETRIEVAL half.

Output: prints per-metric averages, per-category breakdowns, and saves a
detailed per-query CSV to docs/eval_report.csv for pasting into the README
or referencing in interviews.

Run as: python -m app.eval.run_eval
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import anthropic
from datasets import Dataset
from qdrant_client import QdrantClient
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.embeddings import EMBEDDING_MODEL_NAME
from app.eval.test_set import TEST_SET, TestCase
from app.generation.answer import answer_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_pipeline_for_case(
    case: TestCase,
    session: Session,
    qdrant_client: QdrantClient,
    anthropic_client: anthropic.Anthropic,
) -> dict:
    """
    Runs one test case end-to-end and returns Ragas-shaped inputs:
      - question
      - answer (from our system)
      - contexts (the retrieved chunks, as plain strings)
      - ground_truth (the reference answer from the test set)
    """
    result = answer_query(case.question, session, qdrant_client, anthropic_client)

    # Ragas checks every claim in the answer against the "contexts" we
    # supply. Our answers can draw on TWO sources: vector-retrieved FDA
    # chunks AND the structured DDInter severity record. If we only pass
    # the chunks, a correct claim like "Major severity" (sourced from the
    # structured DB) has no matching context to be verified against, and
    # faithfulness scoring unfairly penalizes it as unsupported. Including
    # a plain-text rendering of the structured fact as an additional
    # context fixes this.
    contexts = [c["text"] for c in result.retrieval.retrieved_chunks]
    if result.retrieval.structured_interaction:
        s = result.retrieval.structured_interaction
        contexts.append(
            f"Structured interaction record (DDInter): {s['drug_a']} and "
            f"{s['drug_b']} have a documented {s['severity']} severity interaction."
        )

    return {
        "question": case.question,
        "answer": result.answer_text,
        "contexts": contexts,
        "ground_truth": case.ground_truth,
        "category": case.category,
        "matched_drugs": result.retrieval.matched_drugs,
    }


def _save_per_query_report(records: list[dict], scores: dict, output_path: Path) -> None:
    """Writes a per-query CSV with each test case's inputs, outputs, and scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "question", "matched_drugs", "answer_snippet",
            "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        ])
        for i, rec in enumerate(records):
            writer.writerow([
                rec["category"],
                rec["question"],
                ", ".join(rec["matched_drugs"]),
                rec["answer"][:200].replace("\n", " "),
                scores["faithfulness"][i] if scores.get("faithfulness") else "",
                scores["answer_relevancy"][i] if scores.get("answer_relevancy") else "",
                scores["context_precision"][i] if scores.get("context_precision") else "",
                scores["context_recall"][i] if scores.get("context_recall") else "",
            ])


def _print_summary(records: list[dict], evaluation_result) -> None:
    """Prints overall + per-category averages of each Ragas metric."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    df = evaluation_result.to_pandas()

    print("\nOverall metric averages:")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            avg = df[metric].mean()
            print(f"  {metric:<22} {avg:.3f}")

    # Per-category breakdown, since e.g. "refusal" cases should score
    # very differently from "structured_pair" cases and averaging them
    # together hides real behavior.
    print("\nPer-category averages:")
    categories = sorted({r["category"] for r in records})
    df["category"] = [r["category"] for r in records]
    for cat in categories:
        sub = df[df["category"] == cat]
        row = [f"  {cat:<18} (n={len(sub)}):"]
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric in sub.columns:
                row.append(f"{metric[:4]}={sub[metric].mean():.3f}")
        print(" ".join(row))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the MedGuard RAG evaluation suite")
    parser.add_argument(
        "--category",
        choices=["structured_pair", "single_drug", "refusal", "out_of_scope"],
        default=None,
        help="Only run test cases from this category (useful for cheap, "
             "targeted re-testing after a fix, instead of spending API "
             "credits re-running the full 20-case suite).",
    )
    args = parser.parse_args()

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    test_cases = TEST_SET
    if args.category:
        test_cases = [c for c in TEST_SET if c.category == args.category]
        logger.info("Filtered to category %r: %d cases", args.category, len(test_cases))

    logger.info("Loading pipeline components...")
    engine = create_engine(settings.database_url)
    qdrant_client = QdrantClient(
        host=settings.qdrant_host, port=settings.qdrant_port, https=False
    )
    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    logger.info("Running %d test cases through the pipeline...", len(test_cases))
    records = []
    with Session(engine) as session:
        for i, case in enumerate(test_cases, start=1):
            logger.info("  [%d/%d] %s", i, len(test_cases), case.question[:60])
            rec = _run_pipeline_for_case(case, session, qdrant_client, anthropic_client)
            records.append(rec)

    # Ragas uses an LLM to judge the answers (normal -- how the framework
    # works). We point it at Claude via LangChain so grading uses the same
    # model family as generation. We ALSO must explicitly supply an
    # embeddings model: answer_relevancy computes embedding similarity
    # under the hood, and without an explicit embeddings argument Ragas
    # silently defaults to OpenAI embeddings -- which fails since this
    # project deliberately uses local BGE embeddings and has no OpenAI key.
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(model=settings.claude_model, api_key=settings.anthropic_api_key)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    )

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"] if r["contexts"] else ["(no context retrieved)"],
            "ground_truth": r["ground_truth"],
        }
        for r in records
    ])

    logger.info("Scoring with Ragas (this calls the LLM as judge -- takes a minute)...")
    evaluation_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    _print_summary(records, evaluation_result)

    report_name = f"eval_report_{args.category}.csv" if args.category else "eval_report.csv"
    report_path = Path("docs") / report_name
    scores = evaluation_result.to_pandas().to_dict(orient="list")
    _save_per_query_report(records, scores, report_path)
    logger.info("Per-query report saved to %s", report_path)


if __name__ == "__main__":
    main()
