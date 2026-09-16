"""LinearRAG 问答结果的评测与保存。

本模块同时计算两类指标：使用 LLM 判断预测答案是否正确，
以及检查标准化后的预测答案是否包含金标准答案。
"""

import json
import os
from src.utils import normalize_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# 获取日志记录器
logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, llm_model, predictions_path):
        """绑定评测所用的 LLM 和预测结果文件，并加载预测数据。"""
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        """从指定 JSON 文件加载预测结果。"""
        prediction_results = json.load(open(self.predictions_path))
        return prediction_results

    def calculate_llm_accuracy(self,pre_answer,gold_ans):
        """调用 LLM 判断单个预测答案与金标准答案是否一致。"""
        system_prompt = """You are an expert evaluator. 
        """
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
        """
        response = self.llm_model.infer([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])

        if response.strip().lower() == "correct": # 必须为correct，其他情况都视为错误
            return 1.0
        else:
            return 0.0

    def calculate_contain(self,pre_answers,gold_ans):
        """检查标准化后的预测答案是否包含金标准答案。"""
        # 处理空值
        if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
            return 0            

        if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
            return 0

        # 答案标准化
        s1 = normalize_answer(pre_answers)
        s2 = normalize_answer(gold_ans)

        if s2 in s1:
            return 1
        else:
            return 0

    def evaluate_sig_sample(self,idx,prediction):
        """对单个样本同时执行 LLM 评测和包含评测。"""
        pre_answer = prediction["pred_answer"]
        gold_ans = prediction["gold_answer"]

        # 如需跳过大语言模型评测，可将下一行替换为：llm_acc = 0.0
        llm_acc = self.calculate_llm_accuracy(pre_answer, gold_ans)
        contain_acc = self.calculate_contain(pre_answer, gold_ans)

        return idx, llm_acc, contain_acc

    def evaluate(self,max_workers):
        """并行评测全部样本，写回样本结果并保存总体指标。"""
        # 初始化评测结果列表
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)

        # 使用线程池并行评测
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 同时评测多个样本；谁先完成就先处理，但仍然保留每个结果原来的样本序号。
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx 
                for idx, pred in enumerate(self.prediction_results)
            }

            # 记录已完成的任务数和总分数
            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0

            pbar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample") # 显示进度条

            # 迭代完成的任务，并更新评测结果
            for future in as_completed(futures):
                # 获取样本索引和评测结果
                idx, llm_acc, contain_acc  = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc

                # 更新评测结果
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc

                # 累加已经完成的分数
                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1

                # 更新进度条显示
                current_llm_acc = total_llm_score / completed
                current_contain_acc = total_contain_score / completed

                # 在进度条后显示当前指标
                pbar.set_postfix({
                    'LLM_Acc': f'{current_llm_acc:.3f}',
                    'Contain_Acc': f'{current_contain_acc:.3f}'
                })

                # 让进度增加一个单位
                pbar.update(1)

            pbar.close() # 关闭进度条

        # 计算总体评测结果
        llm_accuracy = sum(llm_scores) / len(llm_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)

        # 向日志输出评测结果
        logger.info(f"Evaluation Results:")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.4f} ({sum(llm_scores)}/{len(llm_scores)})")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")

        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)

        with open(os.path.join(os.path.dirname(self.predictions_path), "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump({"llm_accuracy": llm_accuracy, "contain_accuracy": contain_accuracy}, f, ensure_ascii=False, indent=4)

        # 返回评测结果
        return llm_accuracy, contain_accuracy
