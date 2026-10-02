"""离线构造 Q1/Q2 共用的 HotpotQA 语料，以及独立的 Q4 Medical 语料。

本地 evidence 是候选上下文，不是金标准支持事实。保留抽中问题的全部
候选文档，再加入固定随机文档；原句不改写，按文档边界组成 passages。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from src.paths import DATASETS_DIR, DERIVED_CORPORA_DIR, PROJECT_ROOT


def file_hash(path: Path) -> str:
    """计算文件哈希，用于追踪输入版本。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    """写出人可读的 UTF-8 JSON 文件。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def document_text(title: str, sentences: list[str]) -> str:
    """保留原标题与原句，在标题后增加换行作为文档边界。"""
    return title + "\n" + "".join(sentences)


def collect_documents(questions: list[dict]) -> tuple[dict, list[list[str]]]:
    """按完整文本去重候选文档，并保留所有原问题和文档位置。"""
    documents = {}
    question_documents = []
    for question_index, question in enumerate(questions):
        keys = []
        for context_index, (title, sentences) in enumerate(question["evidence"]):
            text = document_text(title, sentences)
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key not in documents:
                documents[key] = dict(text=text, title=title, source_locations=[])

            documents[key]["source_locations"].append(
                dict(question_index=question_index, question_id=question["id"],
                     context_index=context_index)
            )
            keys.append(key)

        question_documents.append(keys)

    return documents, question_documents


def select_questions(count: int, size: int, seed: int) -> list[int]:
    """仅按固定种子抽题，不使用方法成绩或题目难度筛选。"""
    return sorted(random.Random(f"{seed}:hotpotqa").sample(range(count), size))


def build_package(questions, documents, question_documents, size, distractors, seed):
    """生成共同混合语料，以及问题与候选上下文的映射。"""
    selected = select_questions(len(questions), size, seed)
    required = {key for index in selected for key in question_documents[index]}
    remaining = sorted(set(documents) - required)
    extra = random.Random(f"{seed}:distractors").sample(remaining, distractors)

    # 在去重并加入随机干扰后打乱，避免候选文档总是排在池的前部。
    keys = sorted(required) + extra
    random.Random(f"{seed}:order").shuffle(keys)
    positions = {key: index for index, key in enumerate(keys)}
    chunks = [documents[key]["text"] for key in keys]
    mapping = {
        str(questions[index]["id"]): [positions[key] for key in question_documents[index]]
        for index in selected
    }
    provenance = [
        dict(chunk_index=index, text_sha256=key, title=documents[key]["title"],
             role="selected_question_context" if key in required else "random_distractor",
             source_locations=documents[key]["source_locations"])
        for index, key in enumerate(keys)
    ]
    stats = dict(
        question_count=size, candidate_document_count=len(required),
        added_distractor_count=distractors, passage_count=len(chunks),
        character_count=sum(map(len, chunks)),
        whitespace_word_count=sum(len(text.split()) for text in chunks),
    )
    return [questions[index] for index in selected], chunks, mapping, provenance, stats


MEDICAL_TYPES = (
    "Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation",
)


def build_medical_package(questions, chunks, per_type, candidate_count, distractors, seed, additions):
    """分层抽题，合并词面候选与离线核对补充的原段落；候选不冒充金标准。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    selected = []
    for task in MEDICAL_TYPES:
        group = [q for q in questions if q["question_type"] == task]
        rng = random.Random(f"{seed}:medical:{task}")
        selected.extend(group[i] for i in sorted(rng.sample(range(len(group)), per_type)))
    if set(additions) - {q["id"] for q in selected}:
        raise ValueError("Medical additions contain questions outside the fixed sample")

    # 只用离线词面匹配定位待核对段落，不用答案造文档或运行待比较方法。
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(chunks)
    candidates = {}
    required = set()
    for question in selected:
        relations = question["evidence_relations"]
        relations = " ".join(relations) if isinstance(relations, list) else relations
        text = question["question"] + " " + relations
        scores = (matrix @ vectorizer.transform([text]).T).toarray().ravel()
        ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:candidate_count]
        supplement = additions.get(question["id"], {})
        extra = supplement.get("indices", [])
        if any(i < 0 or i >= len(chunks) for i in extra):
            raise ValueError("Medical addition index is outside the source corpus")
        required.update(ranked)
        required.update(extra)
        candidates[question["id"]] = dict(
            candidate_source_indices=ranked, added_source_indices=extra,
            review_note=supplement.get("reason", "Not individually verified"),
        )

    remaining = sorted(set(range(len(chunks))) - required)
    extra = random.Random(f"{seed}:medical:distractors").sample(remaining, distractors)
    indices = sorted(required) + extra
    random.Random(f"{seed}:medical:order").shuffle(indices)
    corpus = [chunks[i] for i in indices]
    positions = {source_index: i for i, source_index in enumerate(indices)}
    mapping = {
        question_id: [positions[i] for i in dict.fromkeys(
            row["candidate_source_indices"] + row["added_source_indices"]
        )]
        for question_id, row in candidates.items()
    }
    provenance = [
        dict(chunk_index=i, source_chunk_index=source_index,
             text_sha256=hashlib.sha256(chunks[source_index].encode("utf-8")).hexdigest(),
             role="question_candidate" if source_index in required else "random_extra")
        for i, source_index in enumerate(indices)
    ]
    stats = dict(
        question_count=len(selected), questions_per_type=per_type,
        candidate_document_count=len(required), added_distractor_count=distractors,
        passage_count=len(corpus), character_count=sum(map(len, corpus)),
        whitespace_word_count=sum(len(text.split()) for text in corpus),
    )
    return selected, corpus, mapping, provenance, stats, candidates


def main() -> None:
    """输出规模盘点或固定输入包，全程不加载模型、不调用 API。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hotpotqa", "medical"], default="hotpotqa")
    parser.add_argument("--questions-per-type", type=int, default=3)
    parser.add_argument("--candidates-per-question", type=int, default=2)
    parser.add_argument("--medical-additions", type=Path, help="Offline-reviewed extra source indices and notes by question ID.")
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--distractors", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--corpus-id", default=None)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.distractors is None:
        args.distractors = 5 if args.dataset == "medical" else 50
    if args.corpus_id is None:
        args.corpus_id = (f"medical-type{args.questions_per_type}-noise{args.distractors}-s{args.seed}"
                          if args.dataset == "medical" else "hotpotqa-context20-noise50-s20261001")
    if min(args.questions, args.questions_per_type, args.candidates_per_question) < 1 or args.distractors < 0:
        parser.error("questions must be positive and distractors nonnegative")
    if Path(args.corpus_id).name != args.corpus_id:
        parser.error("corpus-id must be a directory name")

    source = DATASETS_DIR / args.dataset / "questions.json"
    if args.dataset == "medical":
        questions = json.loads(source.read_text(encoding="utf-8"))
        chunk_source = source.parent / "chunks.json"
        original_chunks = json.loads(chunk_source.read_text(encoding="utf-8"))
        additions = {}
        if args.medical_additions:
            additions_path = PROJECT_ROOT / args.medical_additions
            additions = json.loads(additions_path.read_text(encoding="utf-8"))
        selected, chunks, mapping, provenance, stats, candidates = build_medical_package(
            questions, original_chunks, args.questions_per_type,
            args.candidates_per_question, args.distractors, args.seed, additions,
        )
        if args.audit_only:
            print(json.dumps(stats))
            return

        output = DERIVED_CORPORA_DIR / args.corpus_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "questions.json", selected)
        write_json(output / "chunks.json", chunks)
        write_json(output / "manifest.json", dict(
            corpus_id=args.corpus_id, source_dataset="medical", seed=args.seed,
            construction="balanced random questions; TF-IDF candidates plus reviewed additions and random extra passages; shuffled",
            evidence_kind="reference_statements_not_source_passage_ids",
            source_questions_path=str(source), source_questions_sha256=file_hash(source),
            source_chunks_sha256=file_hash(chunk_source), builder_sha256=file_hash(Path(__file__)),
            questions_sha256=file_hash(output / "questions.json"),
            chunks_sha256=file_hash(output / "chunks.json"),
            question_ids=[q["id"] for q in selected],
            candidates_per_question=args.candidates_per_question,
            candidate_chunk_indices=mapping, passage_provenance=provenance,
            offline_review=candidates, statistics=stats,
            limitations=[
                "Candidate selection uses question and reference relations; this is a constructed retrieval pool.",
                "Candidate mappings and partial review are not verified complete gold support labels.",
                "Creative reference statements may contain hypothetical patient details absent from source text.",
                "Random extra passages are not guaranteed irrelevant; original passages are unchanged.",
            ],
        ))
        print(json.dumps(stats))
        print(output)
        return

    source = DATASETS_DIR / "hotpotqa" / "questions.json"
    questions = json.loads(source.read_text(encoding="utf-8"))
    documents, question_documents = collect_documents(questions)
    if args.audit_only:
        for size in (20, 30, 50):
            *_, stats = build_package(
                questions, documents, question_documents, size, args.distractors, args.seed
            )
            print(json.dumps(stats))
        return

    selected, chunks, mapping, provenance, stats = build_package(
        questions, documents, question_documents,
        args.questions, args.distractors, args.seed,
    )
    output = DERIVED_CORPORA_DIR / args.corpus_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "questions.json", selected)
    write_json(output / "chunks.json", chunks)
    manifest = dict(
        corpus_id=args.corpus_id, source_dataset="hotpotqa", seed=args.seed,
        construction="all sampled candidate contexts plus random documents; shuffled",
        evidence_kind="candidate_contexts_not_gold_supporting_facts",
        source_questions_sha256=file_hash(source),
        source_questions_path=str(source),
        builder_sha256=file_hash(Path(__file__)),
        questions_sha256=file_hash(output / "questions.json"),
        chunks_sha256=file_hash(output / "chunks.json"),
        question_ids=[str(question["id"]) for question in selected],
        candidate_chunk_indices=mapping, passage_provenance=provenance, statistics=stats,
        limitations=[
            "No gold supporting_facts available; context coverage is not gold evidence recall.",
            "Document boundaries and casing differ from the original merged chunks.",
            "Random extra documents can contain useful facts; no irrelevance guarantee.",
        ],
    )
    write_json(output / "manifest.json", manifest)
    print(json.dumps(stats))
    print(output)


if __name__ == "__main__":
    main()
