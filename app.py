import argparse
from typing import Optional, Sequence

from src.search import RAGSearch


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your local documents.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Business Economics and Financial Analysis",
        help="Question to answer (defaults to the example question).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.2,
        help="Minimum retrieval similarity score.",
    )
    parser.add_argument(
        "--persist-directory",
        default="vectorstore",
        help="Directory containing the persisted Chroma vector store.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="Ollama chat model to use.",
    )
    parser.add_argument(
        "--return-context",
        action="store_true",
        help="Print the full retrieved context after the answer.",
    )
    args = parser.parse_args(argv)

    rag_search = RAGSearch(
        persist_directory=args.persist_directory,
        model_name=args.model,
    )
    result = rag_search.search(
        args.query,
        min_score=args.min_score,
        return_context=args.return_context,
    )

    print(f"Question: {args.query}\n")
    print(f"Answer:\n{result['answer']}")
    print(f"\nConfidence: {result['confidence']:.3f}")

    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(
                f"- {source['source']} (page {source['page']}, "
                f"score {source['score']:.3f})"
            )

    if args.return_context and result.get("context"):
        print(f"\nRetrieved context:\n{result['context']}")


if __name__ == "__main__":
    main()
