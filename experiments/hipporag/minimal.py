"""最小测试代码，验证检索和问答功能是否正常工作。"""

from __future__ import annotations

import argparse
import json

from src.baselines.hipporag import HippoRAG
from src.common.utils import LLM_Model
from src.common.evaluate import Evaluator


PASSAGES = [
    "0:Ada Vale works as a scientist at Harbor Institute.",
    "1:Harbor Institute is located in the city of Lydon.",
    "2:Lydon is a city in the country of Norland.",
    "3:Ada Vale was born in Westhaven.",
    "4:Westhaven is a city in the country of Orania.",
    "5:The novel Quiet Orbit was written by Mira Sol.",
    "6:Mira Sol was born in Eastport.",
    "7:Eastport is a city in the country of Belvaria.",
    "8:The musician Leon Reed works at the Northbank Conservatory.",
    "9:Northbank Conservatory is located in the city of Stoneford.",
    "10:The historian Nora Finch wrote the book Glass Harbor.",
    "11:Glass Harbor was published by Cedar Press.",
]
QUESTIONS = [
    dict(id="bridge-1", question="In which city is the institute where Ada Vale works located?", answer="Lydon"),
    dict(id="bridge-2", question="In which country is the birthplace of Ada Vale located?", answer="Orania"),
    dict(id="bridge-3", question="In which country was the author of Quiet Orbit born?", answer="Belvaria"),
]


def main():
    """验证小型调用；共享缓存与本次生成、评价结果分别保存。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="minimal-cache-split-20261001")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    model = HippoRAG(llm_model=None if args.retrieval_only else LLM_Model("qwen3.8-flash"),
                    experiment_id=args.experiment_id, max_workers=4)
    model.index(PASSAGES)
    results = model.retrieve(QUESTIONS) if args.retrieval_only else model.qa(QUESTIONS)
    assert len(results) == 3
    for result in results:
        assert len(result["sorted_passage"]) == 5
        assert all(p in PASSAGES for p in result["sorted_passage"])
    summary = dict(status="passed", fixture="synthetic integration only",
                   passages=len(PASSAGES), questions=len(QUESTIONS), top_k=5)
    if not args.retrieval_only:
        evaluator = Evaluator(LLM_Model("qwen3.8-flash"), str(model.run_dir / "predictions.json"))
        llm_acc, contain_acc = evaluator.evaluate(max_workers=3)
        summary.update(llm_accuracy=llm_acc, contain_accuracy=contain_acc)
    filename = "validation_retrieval.json" if args.retrieval_only else "validation.json"
    (model.run_dir / filename).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    print(f"Evidence: {model.run_dir}")


if __name__ == "__main__":
    main()
