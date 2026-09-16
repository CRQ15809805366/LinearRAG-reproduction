"""LinearRAG 的 spaCy 命名实体识别。

本模块批量处理片段，建立片段、句子与实体之间的映射，
并为查询提取可用作图检索起点的实体。
"""

import spacy
from collections import defaultdict
import pdb


class SpacyNER:
    def __init__(self, spacy_model):
        self.spacy_model = spacy.load(spacy_model)


    # == 索引阶段：批量处理片段及单文档抽取 ==
    def batch_ner(self, hash_id_to_passage, max_workers):
        """批量处理片段，返回片段到实体和句子到实体的映射。"""

        # 将传入的片段转换为列表
        passage_list = list(hash_id_to_passage.values())

        # 计算批处理大小，并使用 spaCy 的管道方法进行批量处理
        batch_size = len(passage_list) // max_workers
        docs_list = self.spacy_model.pipe(passage_list, batch_size=batch_size)

        # 初始化字典来存储每个片段的哈希 ID 到实体的映射，以及每个句子到实体的映射
        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)

        # 遍历处理后的文档列表，提取实体和句子
        for idx, doc in enumerate(docs_list):
            # 获取当前片段的哈希 ID, 以及单个片段的哈希 ID / 句子 到实体的映射
            passage_hash_id = list(hash_id_to_passage.keys())[idx]
            single_passage_hash_id_to_entities, single_sentence_to_entities = self.extract_entities_sentences(doc, passage_hash_id)

            # 将单个片段的哈希 ID / 句子 到实体的映射合并到总的映射中
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
            for sent, ents in single_sentence_to_entities.items():
                for e in ents:
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)

        # 返回每个片段的哈希 ID 到实体的映射，以及每个句子到实体的映射
        return passage_hash_id_to_entities, sentence_to_entities

    def extract_entities_sentences(self, doc, passage_hash_id):
        """从单个 spaCy 文档中提取有效实体及其所在句子。"""

        # 初始化集合和字典来存储唯一实体、句子到实体的映射，以及片段哈希 ID 到实体的映射
        unique_entities = set()
        sentence_to_entities = defaultdict(list)
        passage_hash_id_to_entities = {}

        # 遍历文档中的实体，提取非序数和非基数的实体，并将其与句子关联
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = ent.text

            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)

            unique_entities.add(ent_text)

        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)

        # 返回单个片段的哈希 ID 到实体的映射，以及每个句子到实体的映射
        return passage_hash_id_to_entities, sentence_to_entities


    # == 检索阶段：提取问题实体 ==
    def question_ner(self, question: str):
        """从问题中提取非序数、非基数的小写实体集合。"""
        doc = self.spacy_model(question)

        question_entities = set()

        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            question_entities.add(ent.text.lower())

        return question_entities
