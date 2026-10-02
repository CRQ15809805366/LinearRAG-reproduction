"""LinearRAG 的命令行运行入口。

本模块负责解析运行参数、加载数据集与嵌入模型、构建 LinearRAG，
并串联索引、问答、结果保存和评测的完整流程。
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from src.methods.linear.config import LinearRAGConfig
from src.methods.linear.LinearRAG import LinearRAG

import warnings

from src.common.evaluate import Evaluator
from src.common.utils import LLM_Model
from src.common.utils import setup_logging
from src.paths import DATASETS_DIR, MODELS_DIR, EXPERIMENT_RESULTS_DIR

from datetime import datetime

# 忽略警告
warnings.filterwarnings('ignore')

# == 运行准备：参数与资源加载 ==
def parse_arguments():
    """解析 LinearRAG 运行所需的命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2", help="The path of embedding model to use")
    parser.add_argument("--dataset_name", type=str, default="novel", help="The dataset to use")
    parser.add_argument("--llm_model", type=str, default="qwen3.8-flash", help="The LLM model to use")
    parser.add_argument("--max_workers", type=int, default=16, help="The max number of workers to use")
    parser.add_argument("--max_iterations", type=int, default=3, help="The max number of iterations to use")
    parser.add_argument("--iteration_threshold", type=float, default=0.4, help="The threshold for iteration")
    parser.add_argument("--passage_ratio", type=float, default=2, help="The ratio for passage")
    parser.add_argument("--top_k_sentence", type=int, default=3, help="The top k sentence to use")
    parser.add_argument("--use_vectorized_retrieval", action="store_true", help="Use vectorized matrix-based retrieval instead of BFS iteration")
    parser.add_argument("--max_questions", type=int, default=None, help="Only run the first N questions")
    args = parser.parse_args()
    if args.max_questions is not None and args.max_questions <= 0:
        parser.error("--max_questions must be greater than 0")
    return args


def load_dataset(dataset_name): 
    """加载指定数据集的问题和文本块，并为文本块加上顺序编号。"""
    questions_path = DATASETS_DIR / dataset_name / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = DATASETS_DIR / dataset_name / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    return questions, passages


def load_embedding_model(embedding_model):
    """从指定路径加载用于检索的 SentenceTransformer 模型。"""
    embedding_model = SentenceTransformer(str(embedding_model), device="cuda")
    return embedding_model


# == 主流程：索引 → 问答 → 保存 → 评测 ==
def main():
    """执行从资源加载到索引、问答、保存和评测的主流程。"""

    # 记录当前时间
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")

    # 解析命令行参数
    args = parse_arguments()

    # 加载嵌入模型和数据集，设置日志记录
    embedding_model = load_embedding_model(args.embedding_model)
    questions, passages = load_dataset(args.dataset_name)
    if args.max_questions is not None:
        questions = questions[:args.max_questions]
    output_dir = EXPERIMENT_RESULTS_DIR / args.dataset_name / time_str
    setup_logging(str(output_dir / "log.txt"))

    # 导入配置
    llm_model = LLM_Model(args.llm_model)
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model, # type: ignore
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval
    )
    rag_model = LinearRAG(global_config=config) # 完成初始化

    # 索引片段
    rag_model.index(passages) 
    # 对问题进行问答
    questions = rag_model.qa(questions)

    # 将结果保存到文件中
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

    # 评估结果
    evaluator = Evaluator(llm_model=llm_model, predictions_path=str(predictions_path))
    evaluator.evaluate(max_workers=args.max_workers)


# 程序入口
if __name__ == "__main__":
    main()
