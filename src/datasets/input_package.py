"""保存和读取固定输入包，核对文本、编号和来源记录，不接触模型缓存。"""

import json
from pathlib import Path

from src.datasets.build_corpus import file_hash, write_json
from src.paths import DERIVED_CORPORA_DIR, PROJECT_ROOT


def resolve_path(path):
    """相对路径始终从项目根目录解析。"""
    return (PROJECT_ROOT / Path(path)).resolve()


def save_package(corpus_id, chunks, questions, manifest, passage_indices=None):
    """保存新输入包及其哈希；已存在的目录一律不覆盖。"""
    if Path(corpus_id).name != corpus_id:
        raise ValueError("corpus-id must be a directory name")
    output = DERIVED_CORPORA_DIR / corpus_id
    output.mkdir(parents=True, exist_ok=False)
    indices = list(range(len(chunks))) if passage_indices is None else passage_indices
    passages = [f"{index}:{chunk}" for index, chunk in zip(indices, chunks)]
    write_json(output / "chunks.json", chunks)
    write_json(output / "passages.json", passages)
    manifest = dict(manifest, corpus_id=corpus_id,
                    chunks_sha256=file_hash(output / "chunks.json"),
                    passages_sha256=file_hash(output / "passages.json"),
                    passage_indices=indices, passage_count=len(chunks))
    if questions is not None:
        write_json(output / "questions.json", questions)
        manifest.update(questions_sha256=file_hash(output / "questions.json"),
                        question_count=len(questions))
    write_json(output / "manifest.json", manifest)
    return output


def load_package(corpus_dir, require_questions=True):
    """读取预先准备的输入并核验哈希；已有三文件输入包仍可直接读取。"""
    if corpus_dir is None:
        raise ValueError("Provide --corpus-dir; first run the corresponding experiment preparation entrypoint")
    directory = resolve_path(corpus_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    if file_hash(directory / "chunks.json") != manifest["chunks_sha256"]:
        raise ValueError("Changed corpus file: chunks.json")
    if not chunks or not all(isinstance(chunk, str) for chunk in chunks):
        raise ValueError("Corpus chunks must be a nonempty list of strings")
    indices = manifest.get("passage_indices", list(range(len(chunks))))
    if len(indices) != len(chunks) or len(set(indices)) != len(indices):
        raise ValueError("Passage numbering differs from corpus")
    if "passage_count" in manifest and manifest["passage_count"] != len(chunks):
        raise ValueError("Passage count differs from manifest")
    passages = [f"{index}:{chunk}" for index, chunk in zip(indices, chunks)]
    if "passages_sha256" in manifest:
        path = directory / "passages.json"
        if file_hash(path) != manifest["passages_sha256"]:
            raise ValueError("Changed corpus file: passages.json")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != passages:
            raise ValueError("Prepared passages differ from chunks and numbering")
        passages = saved

    questions = []
    if require_questions:
        path = directory / "questions.json"
        if file_hash(path) != manifest["questions_sha256"]:
            raise ValueError("Changed corpus file: questions.json")
        questions = json.loads(path.read_text(encoding="utf-8"))
        ids = [str(value) for value in manifest["question_ids"]]
        if not questions or len(ids) != len(questions) or len(set(ids)) != len(ids):
            raise ValueError("Questions must be nonempty and have unique recorded IDs")
        if any("id" in q and str(q["id"]) != identifier for q, identifier in zip(questions, ids)):
            raise ValueError("Question IDs differ from manifest")
        if "question_count" in manifest and manifest["question_count"] != len(questions):
            raise ValueError("Question count differs from manifest")
        # 运行侧统一使用输入包记录的 ID，包括缺失 ID 时的原问题位置。
        questions = [dict(question, id=identifier) for question, identifier in zip(questions, ids)]
    return questions, passages, manifest


def runtime_questions(questions):
    """仅保留原方法读取的字段，不将参考证据或来源映射传入方法。"""
    return [{key: q[key] for key in ("id", "question", "answer") if key in q}
            for q in questions]


def input_record(corpus_dir, manifest):
    """记录本次读取的输入包位置和文件哈希，供运行记录与恢复核对使用。"""
    directory = resolve_path(corpus_dir)
    return dict(input_dir=str(directory), manifest_sha256=file_hash(directory / "manifest.json"),
                chunks_sha256=manifest["chunks_sha256"],
                questions_sha256=manifest.get("questions_sha256"),
                passages_sha256=manifest.get("passages_sha256"))
