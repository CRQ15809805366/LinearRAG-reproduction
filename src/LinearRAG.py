"""LinearRAG 的索引、检索与问答核心实现。

本模块从片段中提取实体和句子关联，构建实体与片段图；
查询时使用种子实体传播、片段语义分数和个性化 PageRank 完成排序，
并在无可用实体时退化为稠密片段检索，最后组织上下文交给 LLM 生成答案。
"""

from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import torch

logger = logging.getLogger(__name__)


class LinearRAG:

    # =========================================================================
    # 初始化过程
    # =========================================================================
    
    def __init__(self, global_config):
        """加载配置、嵌入存储、LLM 和 NER 模型，并初始化空图。"""

        # 导入全局配置, 配置日志
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")

        # 根据配置选择检索方法，并记录使用的检索方法。
        retrieval_method = "Vectorized Matrix-based" if self.config.use_vectorized_retrieval else "BFS Iteration"
        logger.info(f"Using retrieval method: {retrieval_method}")
        if self.config.use_vectorized_retrieval:
                    logger.info(f"Using device: {self.device} for vectorized retrieval")

        # 设置用于 GPU 加速的设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 配置数据集, 加载嵌入存储、LLM 模型和 NER 模型，并构建图结构。
        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)

    def load_embedding_store(self):
        """初始化片段、实体和句子的嵌入存储。"""
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")


    # =========================================================================
    # 索引阶段(公式1-2)
    # =========================================================================
    
    def index(self, passages):
        """索引片段，产生 NER 映射、嵌入存储和用于检索的 GraphML 图。"""
    
        # 初始化图结构和统计信息
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
    
        # 将文档插入嵌入存储
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
    
        # 对新文档进行批量 NER 处理，并将结果与现有数据合并
        existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
    
        # 保存 NER 结果到 JSON 文件
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
    
        # 对NER结果进行整理
        entity_nodes, sentence_nodes, passage_hash_id_to_entities, self.entity_to_sentence, self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
    
        # 将实体和句子节点插入到向量化存储中
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
   
        # 构建实体和句子的双向映射
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
    
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
   
        # 计算实体与片段, 相邻片段之间的边权, 并建图
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
    
        # 将图保存为 GraphML 文件
        output_graphml_path = os.path.join(self.config.working_dir, self.dataset_name, "LinearRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)   
        self.graph.write_graphml(output_graphml_path)

    # -------------------------------------------------------------------------
    # 索引辅助：NER 结果管理与节点整理
    # -------------------------------------------------------------------------
    
    def load_existing_data(self, passage_hash_ids):
        """加载已有 NER 结果，并返回现有映射与尚未处理的片段 ID。"""
        self.ner_results_path = os.path.join(self.config.working_dir, self.dataset_name, "ner_results.json")

        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        """将新片段的 NER 结果合并到现有映射。"""
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)

        # 返回更新后的字典
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        """将片段到实体和句子到实体的映射保存为 JSON。"""
        with open(self.ner_results_path, "w") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities, "sentence_to_entities": existing_sentence_to_entities}, f)

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        """将已有 NER 结果整理为实体、句子及它们之间的映射。"""

        # 五个集合：实体节点、句子节点、passage_hash_id 到实体的映射、实体到句子的映射、句子到实体的映射
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence = defaultdict(set)
        sentence_to_entity = defaultdict(set)

        # 遍历现有的 NER 结果，填充各个集合和字典
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)

        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)

        # 返回五个集合和字典
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity


    # ---------建图：边权统计与图写入--------
    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        """按实体在片段中的相对出现次数计算实体与片段的边权。"""

        passage_to_entity_count = {} # 片段内该实体出现次数
        passage_to_all_score = defaultdict(int) # 片段内所有已识别实体的总出现次数

        # 统计每个实体在所属片段中的出现次数，
        # 同时累计该片段内所有已识别实体的总出现次数。
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity) # type: ignore
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count

        # 用“当前实体出现次数 / passage 内全部实体出现总次数” 作为边权。
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def add_adjacent_passage_edges(self):
        """按片段编号连接相邻片段，并设置固定边权。"""

        # 获取 passage_id 到文本的映射，并使用正则表达式提取索引
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip())) # type: ignore
        ]
        indexed_items.sort(key=lambda x: x[0])

        # 添加边权，权重为 1.0
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        """将当前收集的节点和边写入 igraph 图。"""
        self.add_nodes()
        self.add_edges()
    def add_nodes(self):
        """将尚未存在的实体节点和片段节点加入图。"""

        # 获取图中已有节点的名称和索引
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 

        # 获取实体和片段的哈希 ID 到文本的映射，并合并为一个字典
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}

        # 获取所有片段的哈希 ID
        passage_hash_ids = set(passage_hash_id_to_text.keys())

        # 将新的节点添加到图中
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)

        # 创建节点名称到节点索引的映射，以及标记片段节点
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id] 
            for passage_id in passage_hash_ids 
            if passage_id in self.node_name_to_vertex_idx
        ]
    def add_edges(self):
        """将已统计的节点关联及权重加入图。"""

        # 遍历 node_to_node_stats 字典，构建边列表和权重列表
        edges = []
        weights = []
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)

        # 将边和权重添加到图中
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights



    # =========================================================================
    # 问答阶段(公式3-7)
    # =========================================================================
    
    def qa(self, questions):
        """对每个问题进行检索和问答，并返回最终的问答结果。"""
        # 检索相关片段
        retrieval_results = self.retrieve(questions)
    
        # 构建系统提示和用户提示，准备输入给 LLM 模型进行问答。
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
        all_messages = []
    
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = """"""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
    
        # 使用线程池并行处理所有问题的问答任务
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))
    
        for qa_result, question_info in zip(all_qa_results, retrieval_results):
            try:
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                pred_ans = qa_result
    
            question_info["pred_answer"] = pred_ans
    
        # 返回问答结果
        return retrieval_results
   
    def retrieve(self, questions):
        """为每个问题选择图检索或稠密检索，并返回排序后的片段。"""
        # 加载现有的 NER 结果数据，如果存在的话，并返回现有的映射和新的片段哈希 ID。
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}
    
        # 在需要时为向量化检索预计算稀疏矩阵。
        if self.config.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})")
            logger.info(f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})")
            logger.info(f"Device: {self.device}")
    
        # 对每个问题进行检索
        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            # 对问题进行向量化
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(question, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size)
    
            # 获取种子实体及其索引、哈希 ID 和分数
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = self.get_seed_entities(question)
    
            # 如果识别到种子实体，则使用图搜索方法进行检索；否则，使用稠密检索方法进行检索
            if len(seed_entities) != 0:
                # 从种子实体出发进行检索(有实体, linear rag)
                sorted_passage_hash_ids, sorted_passage_scores = self.graph_search_with_seed_entities(question, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
    
                # 根据配置的 top_k 参数选择最终的片段和分数
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[passage_hash_id] for passage_hash_id in final_passage_hash_ids]
            else:
                # 使用稠密检索方法进行检索(无实体, 退化到普通rag)
                sorted_passage_indices, sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
    
                # 根据配置的 top_k 参数选择最终的片段和分数
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
    
            # 将检索结果存储在字典中，包括问题、排序后的片段、排序后的片段分数和金标准答案
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"]
            }
            retrieval_results.append(result)
    
        return retrieval_results

    # -------------------------------------------------------------------------
    # 检索辅助：种子实体选择
    # -------------------------------------------------------------------------

    def get_seed_entities(self, question):
        """识别问题实体，并在图实体中找出最相似的种子实体及分数。"""
        question_entities = list(self.spacy_ner.question_ner(question))

        # 如果没有识别到实体，则返回空列表
        if len(question_entities) == 0:
            return [], [], [], []

        # 对识别到的实体进行向量化并计算与图实体的相似度
        question_entity_embeddings = self.config.embedding_model.encode(question_entities, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size)
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)

        # 获取与每个查询实体最相似的图实体
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)

        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores


    # -----------图检索：实体传播 → 片段评分 → PPR 排序-----------
    def graph_search_with_seed_entities(self, question, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        """从种子实体出发计算节点权重，再通过 PPR 排序片段。"""

        # 根据配置选择检索方法，并调用相应的计算实体分数的方法(公式五)
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)

        # 计算片段分数，并将实体分数和片段分数组合成节点权重(公式七)
        passage_weights = self.calculate_passage_scores(question, question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights

        # 使用 PPR 方法对图进行排序，得到排序后的片段索引和分数(公式六)
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(node_weights)

        return ppr_sorted_passage_indices, ppr_sorted_passage_scores

    def calculate_entity_scores(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        """使用 BFS 式迭代传播计算实体分数，对应论文公式五。"""

        # 初始化已激活的实体字典和实体权重向量
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))

        # 将种子实体添加到已激活的实体字典中，并初始化其权重
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    

        # 初始化已使用的句子哈希 ID 集合和当前迭代的实体集合
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1

        # 在达到最大迭代次数或没有新的实体时，进行迭代传播
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {} # 存储当前迭代中新激活的实体

            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                # 如果实体分数低于阈值，则跳过该实体
                if entity_score < self.config.iteration_threshold:
                    continue

                # 获取当前实体关联的句子哈希 ID，并过滤掉已经使用过的句子
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue

                # 计算当前实体与句子的相似度，并选择 Top-k 句子进行传播
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]

                # 对于每个选中的句子，获取其关联的实体，并根据当前实体分数和句子分数计算新的实体分数
                for top_sentence_index in top_sentence_indices:
                    # 获取当前句子的哈希 ID 和分数，并将其标记为已使用
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]

                    # 对于当前句子中的每个实体，计算新的实体分数，并根据阈值进行剪枝
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        # 计算实体分数
                        next_entity_score = entity_score * top_sentence_score

                        # 丢弃低于阈值的实体
                        if next_entity_score < self.config.iteration_threshold:
                            continue

                        # 如果实体已经在已激活的实体中，则更新其分数和迭代次数；否则，将其添加到新激活的实体中
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration + 1)

            # 更新已激活的实体集合和当前迭代的实体集合，准备进行下一轮迭代
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1

        # 返回实体权重和已激活的实体信息
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        """结合稠密相似度与已激活实体计算片段分数，对应论文公式七。"""

        # 初始化片段权重向量，并使用稠密检索方法获取与问题最相关的片段索引和分数
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores) # 归一化分数

        # 属性增强, 当前关闭
        apply_attribute_boost = (
            self.config.enable_hybrid_attribute_fallback
            and self._is_attribute_query(question)
        )

        # 将问题转换为小写
        question_lower = question.lower()

        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            # 初始化实体加分总和，并获取当前片段的哈希 ID
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]

            # 片段与问题相似度
            dpr_passage_score = dpr_passage_scores[i]

            # 将片段文本转换为小写
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower() # type: ignore

            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                # 计算实体在片段中的出现次数
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower() # type: ignore
                entity_occurrences = passage_text_lower.count(entity_lower)

                # 根据实体出现次数和 tier 计算实体加分
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1 # 实体层级
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus

            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)

            # 属性增强，未启用
            if apply_attribute_boost:
                overlap = self._attribute_keyword_overlap(question_lower, passage_text_lower)
                if overlap > 0:
                    passage_score += self.config.attribute_keyword_boost * math.log(1 + overlap)

            # 最终计算片段初始分
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight # 片段权重

        # 返回片段初始分
        return passage_weights

    def run_ppr(self, node_weights):
        """使用节点初始权重运行个性化 PageRank，并按分数排序片段。"""
        # 清理非法的初始权重：
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)

        pagerank_scores = self.graph.personalized_pagerank(
            # 为图中的全部节点计算 PageRank。
            vertices=range(len(self.node_name_to_vertex_idx)),

            # PPR参数
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        # 取出所有 passage 节点的分数
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]

        # 将排序后的 passage 局部下标还原成 passage 哈希 ID
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices_in_doc_scores
        ]

        # 返回排好序的 passage ID，以及与其一一对应的 PPR 分数。
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()


    # ---------------稠密检索：片段评分的基础，同时用于无实体时的回退--------------
    def dense_passage_retrieval(self, question_embedding):
        """使用问题与片段嵌入的相似度执行稠密片段检索。"""
        # 将问题向量重塑为二维数组，以便与片段向量计算相似度
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
    
        # 对片段相似度进行排序，获取排序后的索引和分数
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
    
        return sorted_passage_indices, sorted_passage_scores   
    







    # =========================================================================
    # 可选分支：片段评分的属性增强
    # =========================================================================
    def _is_attribute_query(self, question):
        """判断问题是否命中可选属性增强的关键词。"""
        tokens = set(re.findall(r"\w+", question.lower()))
        return any(keyword in tokens for keyword in self.config.attribute_query_keywords)

    def _attribute_keyword_overlap(self, question_lower, passage_text_lower):
        overlap = 0
        for keyword in self.config.attribute_query_keywords:
            if keyword in question_lower and keyword in passage_text_lower:
                overlap += 1
        return overlap

    # =========================================================================
    # 可选分支：实体传播的向量化实现
    # =========================================================================
    def _precompute_sparse_matrices(self):
        """预计算并缓存实体与句子的稀疏邻接矩阵。

        矩阵供 PyTorch 向量化检索使用，在 retrieve() 开始时构建一次，
        避免为每个查询重复构建。
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)

        # 使用 COO 格式构建实体到句子的矩阵，即提及矩阵。
        entity_to_sentence_indices = []
        entity_to_sentence_values = []

        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)

        # 构建句子到实体的矩阵。
        sentence_to_entity_indices = []
        sentence_to_entity_values = []

        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)

        # 转换为 PyTorch 稀疏张量，先使用 COO 格式，再按需转为高效的 CSR 格式。
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )

        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )

    def calculate_entity_scores_vectorized(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        """使用 PyTorch 稀疏张量向量化计算实体传播分数。

        该实现对应论文公式五，矩阵和实体分数向量均采用稀疏表示。
        动态剪枝与 BFS 行为保持一致：
        - 句子去重，即跟踪已经使用的句子；
        - 每个实体独立选择 Top-k 句子；
        - 按阈值正确剪枝。
        """
        # 初始化实体权重。
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)

        # 一次性计算所有句子与问题的相似度。
        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()

        # 转换为 torch 张量并移动到目标设备。
        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)

        # 跟踪已经使用的句子以便去重，与 BFS 版本一致。
        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)

        # 将种子实体分数初始化为稀疏张量。
        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()

        # 同时维护用于累积总分的稠密张量。
        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.device), 
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device))

        # 初始化已激活实体。
        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score

        current_entity_scores_sparse = entity_scores_sparse

        # 在 GPU 上使用稀疏矩阵执行迭代式矩阵传播。
        for iteration in range(1, self.config.max_iterations):
            # 将稀疏张量转为稠密张量，以执行阈值操作。
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()

            # 对当前分数应用阈值。
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold, 
                current_entity_scores_dense, 
                torch.zeros_like(current_entity_scores_dense)
            )

            # 获取非零索引，用于构造稀疏表示。
            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)

            if len(nonzero_indices) == 0:
                break

            # 提取非零值并创建稀疏张量。
            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.device
            ).coalesce()

            # 第 1 步：稀疏实体分数乘以稀疏的实体到句子矩阵。
            # 将稀疏向量转换为二维形式，以便执行矩阵乘法。
            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device
            ).coalesce()

            # E 乘以 E2S 得到句子激活分数，稀疏矩阵相乘后得到稠密结果。
            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(),
                current_scores_2d
            )

            # 在 squeeze 前转为稠密张量，避免 CUDA 稀疏张量问题。
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()

            # 执行句子去重：屏蔽已经使用的句子。
            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )

            # 第 2 步：为每个实体选择 Top-k 句子。
            # 此处与 BFS 行为一致：每个实体独立选择自己的 Top-k 句子。
            selected_sentence_indices_list = []

            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:
                # 遍历每个活跃实体。
                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]

                    # 从稀疏矩阵中取得与当前实体相连的句子。
                    # entity_to_sentence_sparse 的形状为实体数乘以句子数。
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]  # 取得列索引。

                    if len(entity_sentence_indices) == 0:
                        continue

                    # 过滤已经使用的句子。
                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]

                    if len(available_sentence_indices) == 0:
                        continue

                    # 取得用于排序的句子相似度。
                    sentence_sims = sentence_similarities[available_sentence_indices]

                    # 只根据句子相似度选择 Top-k 句子，与 BFS 第 240 行一致。
                    # 选择时不乘以 entity_score。
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)

                # 合并所有选中句子，并通过 unique 去重。
                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)

                    # 将选中句子标记为已使用。
                    used_sentence_mask[unique_selected_sentences] = True

                    # 计算用于传播的加权句子分数。
                    # 加权分数 = 句子激活值 × 句子相似度。
                    weighted_sentence_scores = sentence_activation * sentence_similarities

                    # 将未选中句子的分数清零。
                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    # 没有选中句子时，创建零向量。
                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            else:
                # 没有活跃实体，或 top_k_sentence 为 0。
                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)

            # 第 3 步：加权句子乘以句子到实体矩阵，传播到下一批实体。
            # 转为稀疏表示以提高计算效率。
            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)

            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device
                ).coalesce()

                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(),
                    weighted_scores_2d
                )

                # 在 squeeze 前转为稠密张量，避免 CUDA 稀疏张量问题。
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)

            # 更新实体分数，并以稠密形式累积。
            entity_scores_dense += next_entity_scores_dense

            # 更新 actived_entities 字典，与 BFS 一样记录最后一次触发。
            # 此处与 BFS 行为一致：无条件更新高于阈值的实体。
            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                # 无条件更新以记录最后一次触发，与 BFS 第 252 行一致。
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)

            # 为下一轮迭代准备稀疏张量。
            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values, 
                    (num_entities,), device=self.device
                ).coalesce()
            else:
                break

        # 转回 NumPy 以执行最终处理。
        entity_scores_final = entity_scores_dense.cpu().numpy()

        # 将实体分数映射为图节点权重，只处理非零分数。
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)

        return entity_weights, actived_entities
