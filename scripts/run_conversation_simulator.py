"""CLI to run scripted conversational scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shl_recommender.evaluation.conversation_simulator import ConversationSimulator


DEFAULT_SCENARIOS = {
    "clarify_then_recommend": [
        "Hi",
        "I am hiring a Java developer who collaborates with business teams",
    ],
    "refine": [
        "Need cognitive and personality tests for a graduate analyst role",
        "Focus more on personality and fewer long tests",
    ],
    "refusal": [
        "What is the weather in London?",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run conversation simulator scenarios")
    parser.add_argument(
        "--scenario",
        choices=list(DEFAULT_SCENARIOS.keys()),
        default="clarify_then_recommend",
    )
    parser.add_argument("--turn", action="append", help="Custom user turn (repeatable)")
    args = parser.parse_args()

    turns = args.turn or DEFAULT_SCENARIOS[args.scenario]
    simulator = ConversationSimulator()

    for idx, user_text in enumerate(turns, start=1):
        turn = simulator.send(user_text)
        print(f"\n=== Turn {idx} ===")
        print(f"User: {user_text}")
        print(f"Reply: {turn.response.reply}")
        print(f"Recommendations: {len(turn.response.recommendations)}")
        print(f"End: {turn.response.end_of_conversation}")
        if turn.response.recommendations:
            print(
                json.dumps(
                    [item.model_dump(mode="json") for item in turn.response.recommendations[:3]],
                    indent=2,
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
