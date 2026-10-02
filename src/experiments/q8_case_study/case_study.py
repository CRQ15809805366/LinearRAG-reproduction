"""复现 Q8 单案例，并观察原版 HippoRAG 的局部图和支持文本排名。"""

from __future__ import annotations

import json
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.linear.config import LinearRAGConfig
from src.methods.linear.LinearRAG import LinearRAG
from src.paths import EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.common.utils import LLM_Model
from src.datasets.input_package import load_package, input_record, runtime_questions


DATASET = "2wikimultihop"
QUESTION_ID = "3a3c2efe0bdc11eba7f7acde48001122"
EXPERIMENT_ID = "q8-20passages-case-20261001"

# 没有改算法, 只是周围加了一层简单记录，把每轮选中了什么句子、激活了什么实体保存下来。
import random
from src.paths import DATASETS_DIR, DERIVED_CORPORA_DIR
from src.datasets.build_corpus import file_hash
from src.datasets.input_package import save_package


SUPPORT_FRAGMENTS = (
    "holy roman empress by marriage to frederick barbarossa",
    "he was elected king of germany at frankfurt on 4 march 1152",
)


def build_case(seed, distractor_count):
    """按原规则定位支持段落和抽取干扰段落；准备不依赖历史模型缓存。"""
    source = DATASETS_DIR / DATASET
    questions = json.loads((source / "questions.json").read_text(encoding="utf-8"))
    question_index, question = next((i, q) for i, q in enumerate(questions) if q.get("id") == QUESTION_ID)
    chunks = json.loads((source / "chunks.json").read_text(encoding="utf-8"))
    support_indices = []
    for fragment in SUPPORT_FRAGMENTS:
        matches = [i for i, chunk in enumerate(chunks) if fragment in f"{i}:{chunk}".lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected one support passage for: {fragment}")
        support_indices.append(matches[0])

    distractors = [i for i in range(len(chunks)) if i not in support_indices]
    source_indices = sorted(support_indices + random.Random(seed).sample(distractors, distractor_count))
    observation = dict(
        entities=["Beatrice I", "Frederick Barbarossa", "Germany"],
        support_passage_indices=[source_indices.index(i) for i in support_indices],
        source_passage_indices=source_indices,
    )
    manifest = dict(
        source_dataset=DATASET, prepared_for="q8", seed=seed,
        question_ids=[str(question["id"])], source_question_indices=[question_index],
        source_questions_path=str(source / "questions.json"),
        source_questions_sha256=file_hash(source / "questions.json"),
        source_chunks_path=str(source / "chunks.json"),
        source_chunks_sha256=file_hash(source / "chunks.json"),
        source_passage_count=len(chunks),
        source_passage_indices=source_indices, support_source_indices=support_indices,
        support_fragments=list(SUPPORT_FRAGMENTS), distractor_count=distractor_count,
        case_observation=observation,
        construction="two original support passages plus fixed random distractors; original numbering",
        builder_sha256=file_hash(Path(__file__)),
    )
    return question, [chunks[i] for i in source_indices], source_indices, manifest


class CaseStudyLinearRAG(LinearRAG):
    """记录原 BFS 选中的句子和激活实体，不改变传播计算。"""

    def __init__(self, global_config: LinearRAGConfig):
        """初始化案例轨迹和原 LinearRAG。"""
        self.trace: list[dict] = []
        super().__init__(global_config)


    def calculate_entity_scores(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        """执行原 BFS 计算，并保留每轮选中的句子与激活实体。"""
        active_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for index, text, hash_id, score in zip(
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        ):
            active_entities[hash_id] = (index, score, 1)
            entity_weights[self.node_name_to_vertex_idx[hash_id]] = score
            self.trace.append({
                "iteration": 0,
                "entity": text,
                "score": float(score),
                "kind": "seed",
            })

        used_sentences = set()
        current_entities = active_entities.copy()
        iteration = 1
        while current_entities and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (_entity_id, entity_score, _tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [
                    sid
                    for sid in self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]
                    if sid not in used_sentences
                ]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [
                    self.sentence_embedding_store.hash_id_to_idx[sid]
                    for sid in sentence_hash_ids
                ]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_column = question_embedding.reshape(-1, 1)
                similarities = np.dot(sentence_embeddings, question_column).flatten()
                top_indices = np.argsort(similarities)[::-1][: self.config.top_k_sentence]
                for top_index in top_indices:
                    sentence_hash_id = sentence_hash_ids[top_index]
                    sentence_score = float(similarities[top_index])
                    used_sentences.add(sentence_hash_id)
                    activated = []
                    for next_hash_id in self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id]:
                        next_score = float(entity_score * sentence_score)
                        if next_score < self.config.iteration_threshold:
                            continue
                        node_index = self.node_name_to_vertex_idx[next_hash_id]
                        entity_weights[node_index] += next_score
                        new_entities[next_hash_id] = (node_index, next_score, iteration + 1)
                        activated.append({
                            "entity": self.entity_embedding_store.hash_id_to_text[next_hash_id],
                            "score": next_score,
                        })
                    self.trace.append({
                        "iteration": iteration,
                        "from_entity": self.entity_embedding_store.hash_id_to_text[entity_hash_id],
                        "sentence": self.sentence_embedding_store.hash_id_to_text[sentence_hash_id],
                        "sentence_score": sentence_score,
                        "activated_entities": activated,
                    })
            active_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, active_entities


def load_inputs(corpus_dir):
    """读取指定案例及其观察信息，核对原编号和支持段落位置。"""
    questions, passages, package = load_package(corpus_dir)
    if package["source_dataset"] != DATASET or len(questions) != 1 or questions[0].get("id") != QUESTION_ID:
        raise ValueError("Q8 requires the specified single 2Wiki case")
    observation = package["case_observation"]
    indices = observation["source_passage_indices"]
    supports = observation["support_passage_indices"]
    if indices != package["passage_indices"] or len(passages) < 5 or len(supports) != 2:
        raise ValueError("Q8 passage numbering or support observations differ from input")
    for position, source_index, fragment in zip(supports, package["support_source_indices"], package["support_fragments"]):
        if position < 0 or position >= len(passages) or indices[position] != source_index or fragment not in passages[position].lower():
            raise ValueError("Q8 support passage differs from observation")
    return questions[0], passages, observation, package


def main() -> int:
    """读取固定案例输入并运行案例实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, help="实验输入包目录。")
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--distractors", type=int, default=18)
    parser.add_argument("--corpus-id")
    args = parser.parse_args()
    if not args.corpus_dir:
        if args.distractors < 3:
            parser.error("At least three distractors are required for Top-5 retrieval")
        question, chunks, indices, manifest = build_case(args.seed, args.distractors)
        corpus_id = args.corpus_id or f"q8-case-n{len(chunks)}-s{args.seed}"
        args.corpus_dir = DERIVED_CORPORA_DIR / corpus_id
        if args.corpus_dir.exists():
            _, _, existing = load_package(args.corpus_dir)
            for key in ("seed", "source_passage_indices", "source_questions_sha256", "source_chunks_sha256"):
                if existing.get(key) != manifest[key]:
                    raise ValueError(f"Existing case package differs: {args.corpus_dir} ({key})")
        else:
            args.corpus_dir = save_package(corpus_id, chunks, [question], manifest, indices)
    question, passages, observation, package = load_inputs(args.corpus_dir)

    output_dir = EXPERIMENT_RESULTS_DIR / "q8_case_study" / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = dict(
        status="prepared", dataset=DATASET, question=question,
        input_package=input_record(args.corpus_dir, package),
        passage_count=len(passages),
        corpus_sha256=hashlib.sha256(
            json.dumps(passages, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        generation_model=args.llm_model, extraction_model="qwen3.8-flash",
        embedding_model="all-mpnet-base-v2", baseline="HippoRAG 2024 v1.0.0",
        case_observation=observation,
        observation_role="Read-only diagnostics; not used in ranking or generation",
        indexing_llm_calls_if_no_cache=2 * len(passages),
        query_ner_llm_calls_if_no_cache=1, generation_calls=2,
    )
    prepared_path = output_dir / "prepared.json"
    if prepared_path.exists():
        if json.loads(prepared_path.read_text(encoding="utf-8")) != prepared:
            raise ValueError("Existing case input differs; use a new experiment ID")
    else:
        prepared_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8")


    # 先运行 LinearRAG，立即保存，避免后续 HippoRAG 失败丢失结果。
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(str(MODELS_DIR / "all-mpnet-base-v2"), device="cuda")
    llm_model = LLM_Model(args.llm_model)
    config = LinearRAGConfig(
        dataset_name=DATASET,
        embedding_model=embedding_model,
        llm_model=llm_model,
        spacy_model="en_core_web_trf",
        working_dir=output_dir / "linearrag_cache",
        max_workers=1,
        max_iterations=3,
        iteration_threshold=0.4,
        passage_ratio=0.05,
        top_k_sentence=1,
        use_vectorized_retrieval=False,
    )
    rag = CaseStudyLinearRAG(config)
    rag.index(passages)
    # 两种方法必须使用相同 passage 文本，核对当前运行缓存。
    if set(rag.passage_embedding_store.texts) != set(passages):
        raise ValueError("LinearRAG cached corpus differs from HippoRAG input")
    prediction = rag.qa(runtime_questions([question]))[0]
    prediction["id"] = question["id"]

    linear_result = dict(
        prediction=prediction, trace=rag.trace,
        parameters=dict(max_iterations=3, iteration_threshold=0.4,
                        passage_ratio=0.05, top_k_sentence=1,
                        retrieval_top_k=5, retrieval_path="BFS"),
    )
    (output_dir / "linearrag.json").write_text(
        json.dumps(linear_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from src.methods.hippo.adapter import HippoRAG
    hippo = HippoRAG(
        llm_model=llm_model, extraction_model="qwen3.8-flash",
        retrieval_top_k=5, max_workers=4,
        working_dir=output_dir, experiment_id="hipporag",
        case_observation=observation,
    )
    hippo.index(passages)
    hippo_prediction = hippo.qa(runtime_questions([question]))[0]

    result = dict(
        experiment_id=args.experiment_id, dataset=DATASET, question=question,
        generation_model=args.llm_model, embedding_model="all-mpnet-base-v2",
        baseline="HippoRAG 2024 v1.0.0; not HippoRAG2",
        linearrag=linear_result, hipporag=hippo_prediction,
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(dict(
        output=str(output_dir / "result.json"),
        linearrag_answer=prediction["pred_answer"],
        hipporag_answer=hippo_prediction["pred_answer"],
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
