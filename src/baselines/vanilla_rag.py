"""Reusable dense-retrieval Vanilla RAG baseline."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm

from src.common.embedding_store import EmbeddingStore
from src.paths import CACHE_DIR


QA_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze "
    "text passages and corresponding questions meticulously. Your response "
    'start after "Thought: ", where you will methodically break down the '
    'reasoning process, illustrating how you arrive at conclusions. Conclude '
    'with "Answer: " to present a concise, definitive response, devoid of '
    "additional elaborations."
)


class VanillaRAG:
    """Top-k dense retrieval followed by answer generation."""

    def __init__(
        self,
        dataset_name,
        embedding_model,
        llm_model,
        max_workers=16,
        retrieval_top_k=5,
        batch_size=128,
        working_dir: Union[str, Path] = CACHE_DIR,
    ):
        self.dataset_name = dataset_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.max_workers = max_workers
        self.retrieval_top_k = retrieval_top_k
        self.batch_size = batch_size
        self.working_dir = Path(working_dir)
        self.passage_store = EmbeddingStore(
            embedding_model,
            db_filename=str(
                self.working_dir / dataset_name / "passage_embedding.parquet"
            ),
            batch_size=batch_size,
            namespace="passage",
        )
        self.passages = []
        self.passage_embeddings = np.array([])

    def index(self, passages):
        """Index the supplied corpus and freeze the searchable passage order."""
        self.passage_store.insert_text(passages)
        passage_hash_ids = [
            self.passage_store.text_to_hash_id[passage] for passage in passages
        ]
        self.passages = list(passages)
        self.passage_embeddings = np.asarray(
            self.passage_store.get_embeddings(passage_hash_ids), dtype=np.float32
        )

    def retrieve(self, questions):
        """Return the top-k passages for every question by cosine similarity."""
        if not self.passages:
            raise RuntimeError("Call index(passages) before retrieve(questions).")

        results = []
        for question_info in tqdm(questions, desc="Vanilla retrieval"):
            question_embedding = self.embedding_model.encode(
                question_info["question"],
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            scores = self.passage_embeddings @ question_embedding
            top_indices = np.argsort(scores)[::-1][: self.retrieval_top_k]
            results.append(
                {
                    "id": question_info.get("id"),
                    "question": question_info["question"],
                    "sorted_passage": [self.passages[index] for index in top_indices],
                    "sorted_passage_scores": [
                        float(scores[index]) for index in top_indices
                    ],
                    "gold_answer": question_info["answer"],
                }
            )
        return results

    def qa(self, questions):
        """Retrieve context and generate one final answer per question."""
        results = self.retrieve(questions)
        messages = []
        for result in results:
            context = "".join(
                f"{passage}\n" for passage in result["sorted_passage"]
            )
            user_prompt = f"{context}Question: {result['question']}\n Thought: "
            messages.append(
                [
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            responses = list(
                tqdm(
                    executor.map(self.llm_model.infer, messages),
                    total=len(messages),
                    desc="Vanilla generation",
                )
            )

        for result, response in zip(results, responses):
            if "Answer:" in response:
                response = response.split("Answer:", 1)[1].strip()
            result["pred_answer"] = response
        return results
