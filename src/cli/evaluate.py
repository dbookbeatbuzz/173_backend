"""Command line interface for evaluating a client model."""

from __future__ import annotations

import argparse
import json
import sys

from src.config import get_settings
from src.services.evaluation import evaluate_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a client model on DomainNet splits")
    parser.add_argument("client_id", type=int, help="Client identifier")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output result as JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    result = evaluate_client(
        client_id=args.client_id,
        split=args.split,
        limit=args.limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        models_root=settings.models_root,
        data_root=settings.data_root,
        preprocessor_json=settings.preprocessor_json,
    )

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("评测结果：")
        print(f"  Client: {result['client_id']}")
        print(f"  Split: {result['split']}")
        print(f"  Samples: {result['samples']}")
        print(f"  Correct: {result['correct']}")
        print(f"  Accuracy: {result['accuracy']:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
