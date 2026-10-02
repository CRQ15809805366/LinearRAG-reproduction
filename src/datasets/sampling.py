"""提供多实验共用的固定种子抽样、分层抽样和完整原语料输入包保存。"""

import json
import random
from collections import defaultdict
from pathlib import Path

from src.datasets.build_corpus import MEDICAL_TYPES, file_hash
from src.datasets.input_package import load_package, save_package
from src.paths import DATASETS_DIR, DERIVED_CORPORA_DIR


def sample_dataset(dataset_name, count, seed, stratified=False):
    """迁移原抽样规则，保留完整问题字段、问题顺序和原始片段顺序。"""
    source = DATASETS_DIR / dataset_name
    all_questions = json.loads((source / "questions.json").read_text(encoding="utf-8"))
    chunks = json.loads((source / "chunks.json").read_text(encoding="utf-8"))
    if stratified:
        grouped = defaultdict(list)
        for index, question in enumerate(all_questions):
            grouped[question["question_type"]].append(index)
        indices = []
        for question_type in MEDICAL_TYPES:
            available = grouped[question_type]
            rng = random.Random(f"{seed}:{dataset_name}:{question_type}")
            positions = sorted(rng.sample(range(len(available)), count))
            indices.extend(available[position] for position in positions)
    else:
        rng = random.Random(f"{seed}:{dataset_name}")
        indices = sorted(rng.sample(range(len(all_questions)), count))
    return [all_questions[index] for index in indices], chunks, indices, len(all_questions)


def build_sample_inputs(
    datasets,
    count,
    seed,
    corpus_prefix,
    *,
    stratified=False,
    corpus_id=None,
):
    """统一抽样并保存输入包，复用来源一致的已有输入包。"""
    paths = []

    for dataset in datasets:
        questions, chunks, indices, available = sample_dataset(dataset, count, seed, stratified)
        # 缺失 ID 时统一使用原问题位置，保证跨实验配对一致。
        ids = [str(q.get("id", index)) for q, index in zip(questions, indices)]
        source = DATASETS_DIR / dataset
        package_id = corpus_id or f"{corpus_prefix}-{dataset}-n{len(questions)}-s{seed}"
        if corpus_id and len(datasets) > 1:
            package_id += f"-{dataset}"
        manifest = dict(
            source_dataset=dataset, prepared_for=corpus_prefix, seed=seed,
            input_mode="sampled", corpus_scope="full_source_chunks",
            construction="balanced per-type sample" if stratified else "fixed-seed sorted random sample",
            sample_count=len(questions), available_questions=available,
            questions_per_type=count if stratified else None,
            max_questions=count if not stratified else None,
            question_ids=ids, source_question_indices=indices,
            source_questions_path=str(source / "questions.json"),
            source_questions_sha256=file_hash(source / "questions.json"),
            source_chunks_path=str(source / "chunks.json"),
            source_chunks_sha256=file_hash(source / "chunks.json"),
            chunk_numbering="zero-based original source chunk order; unchanged text",
            builder_sha256=file_hash(Path(__file__)),
            statistics=dict(question_count=len(questions), passage_count=len(chunks),
                            character_count=sum(map(len, chunks)),
                            whitespace_word_count=sum(len(chunk.split()) for chunk in chunks)),
        )
        output = DERIVED_CORPORA_DIR / package_id
        if output.exists():
            _, _, existing = load_package(output)
            for key in ("source_dataset", "seed", "question_ids", "source_question_indices",
                        "source_questions_sha256", "source_chunks_sha256", "corpus_scope"):
                if existing.get(key) != manifest[key]:
                    raise ValueError(f"Existing sample package differs: {output} ({key})")
        else:
            output = save_package(package_id, chunks, questions, manifest)
        paths.append(output)
    return paths
