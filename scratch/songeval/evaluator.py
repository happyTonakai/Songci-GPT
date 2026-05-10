"""
Evaluator - 模型性能量化模块

功能：
1. Perplexity (PPL) 计算
2. 格律符合度打分 (Form Score)
   - Structure Match
   - Tonal Accuracy
   - Rhyme Consistency
"""

import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pypinyin import Style, pinyin


class Evaluator:
    """
    宋词生成模型评估器

    提供两种评估方式：
    1. PPL 计算 - 评估模型概率建模能力
    2. 格律打分 - 评估生成文本符合度
    """

    # 复用 RegistryBuilder 的韵部映射
    RHYME_MAPPING = {
        "an": "AN",
        "ian": "AN",
        "uan": "AN",
        "üan": "AN",
        "van": "AN",
        "ang": "ANG",
        "iang": "ANG",
        "uang": "ANG",
        "en": "EN",
        "in": "EN",
        "un": "EN",
        "ün": "EN",
        "vn": "EN",
        "eng": "ENG",
        "ing": "ENG",
        "ueng": "ENG",
        "ong": "ENG",
        "iong": "ENG",
        "ao": "AO",
        "iao": "AO",
        "ou": "OU",
        "iu": "OU",
        "ai": "AI",
        "uai": "AI",
        "ei": "EI",
        "ui": "EI",
        "uei": "EI",
        "a": "A",
        "ia": "A",
        "ua": "A",
        "o": "O",
        "uo": "O",
        "io": "O",
        "e": "E",
        "ie": "E",
        "ue": "E",
        "üe": "E",
        "ve": "E",
        "i": "I",
        "er": "I",
        "-i": "I",
        "u": "U",
        "ü": "V",
        "v": "V",
    }

    def __init__(
        self, registry_path: Optional[str] = None, registry: Optional[Dict] = None
    ):
        """
        Args:
            registry_path: 格律库 JSON 文件路径
            registry: 或直接传入格律库字典
        """
        if registry is not None:
            self.registry = registry
        elif registry_path is not None:
            with open(registry_path, "r", encoding="utf-8") as f:
                self.registry = json.load(f)
        else:
            raise ValueError("必须提供 registry_path 或 registry")

    @torch.no_grad()
    def compute_ppl(self, model, data_loader, device="cuda") -> float:
        """
        计算困惑度 (Perplexity)

        Args:
            model: 训练好的模型
            data_loader: 验证集数据加载器
            device: 计算设备

        Returns:
            PPL 值
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            # 前向传播获取 logits 和 loss
            try:
                # 尝试调用模型的 forward 方法
                logits, loss = model(x, y)
            except Exception:
                # 如果模型不支持直接返回 loss，手动计算
                logits = model(x)
                # 计算交叉熵 loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
                )

            # 累加 loss (注意：loss 可能是平均值，需要还原)
            if loss.dim() == 0:  # 标量 loss
                # 假设 loss 是 mean over batch and seq
                total_loss += loss.item() * x.size(0) * x.size(1)
                total_tokens += x.size(0) * x.size(1)
            else:
                total_loss += loss.sum().item()
                total_tokens += (y != -100).sum().item()  # 排除 padding

        # 计算平均 loss
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        ppl = math.exp(avg_loss)

        return ppl

    def get_char_tone(self, char: str) -> str:
        """获取单个字的平仄"""
        if not char or len(char) != 1:
            return "?"

        try:
            py = pinyin(char, style=Style.TONE3, heteronym=False)
            if not py or not py[0]:
                return "?"

            py_str = py[0][0]
            tone = None
            for c in py_str:
                if c.isdigit():
                    tone = int(c)
                    break

            if tone is None:
                return "?"

            if tone in [1, 2]:
                return "P"
            elif tone in [3, 4]:
                return "Z"
            else:
                return "?"
        except Exception:
            return "?"

    def get_rhyme_id(self, char: str) -> str:
        """获取字的韵部 ID"""
        if not char or len(char) != 1:
            return "UNKNOWN"

        try:
            py = pinyin(char, style=Style.FINALS, heteronym=False)
            if not py or not py[0]:
                return "UNKNOWN"

            final = py[0][0].lower()

            if "ü" in final or "v" in final:
                for rhyme, group in self.RHYME_MAPPING.items():
                    if rhyme in final:
                        return group

            return self.RHYME_MAPPING.get(final, final.upper() if final else "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    def parse_generated_text(self, text: str, title: str = "") -> List[str]:
        """
        解析生成的文本，提取句子

        Args:
            text: 生成的宋词文本
            title: 词牌名（用于智能拆分）

        Returns:
            句子列表（不含标点）
        """
        # 移除特殊标记
        text = text.replace("<bos>", "").replace("<eos>", "").replace("<sep>", "")

        # 按标点分割
        import re

        # 匹配中文标点和常见分隔符
        sentences = re.split(r"[，。、！？；：,.!?;:]", text)

        # 清理每句
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            # 移除所有非中文字符（保留少量常用字符）
            sent = re.sub(r"[^\u4e00-\u9fa5]", "", sent)
            if sent:
                # 智能拆分：处理数据源中合并的段落
                # 这些词牌的14字段落实际上是两个7字句合并
                split_14_titles = [
                    "浣溪沙",
                    "鹧鸪天",
                    "玉楼春",
                    "木兰花",
                    "踏莎行",
                    "减字木兰花",
                    "瑞鹧鸪",
                    "蝶恋花",
                ]
                if len(sent) == 14 and title in split_14_titles:
                    cleaned.append(sent[:7])
                    cleaned.append(sent[7:])
                else:
                    cleaned.append(sent)

        return cleaned

    def evaluate_structure(self, title: str, text: str) -> Tuple[bool, float, List[str]]:
        """
        评估结构匹配度

        Returns:
            (是否完全匹配, 句长匹配比例 0-1, 实际句子列表)
        """
        if title not in self.registry:
            return False, 0.0, []

        standard = self.registry[title]
        expected_structure = standard["structure"]

        sentences = self.parse_generated_text(text, title)
        actual_structure = [len(sent) for sent in sentences]

        # 如果句数不匹配，尝试智能合并短句
        if len(actual_structure) != len(expected_structure):
            merged_sentences = self._try_merge_sentences(sentences, expected_structure)
            if merged_sentences:
                sentences = merged_sentences
                actual_structure = [len(sent) for sent in sentences]

        # 比较句数和每句字数
        matches = len(actual_structure) == len(expected_structure)
        if matches:
            for actual, expected in zip(actual_structure, expected_structure):
                if actual != expected:
                    matches = False
                    break

        # 计算连续结构分：句长匹配比例
        n = min(len(actual_structure), len(expected_structure))
        if n == 0:
            ratio = 0.0
        else:
            matched = sum(1 for a, e in zip(actual_structure, expected_structure) if a == e)
            # 句数不匹配时也扣分
            ratio = matched / len(expected_structure)

        return matches, ratio, sentences

    def _try_merge_sentences(
        self, sentences: List[str], expected_structure: List[int]
    ) -> Optional[List[str]]:
        """
        尝试合并短句以匹配预期的结构

        策略：如果连续短句的总长度等于预期的某句长度，则合并
        """
        if not sentences or not expected_structure:
            return None

        result = []
        i = 0
        j = 0

        while i < len(sentences) and j < len(expected_structure):
            current_sentences = [sentences[i]]
            current_len = len(sentences[i])
            expected_len = expected_structure[j]

            # 如果当前句子长度正好匹配
            if current_len == expected_len:
                result.append(sentences[i])
                i += 1
                j += 1
            # 如果当前句子太短，尝试合并后续句子
            elif current_len < expected_len:
                # 尝试合并，直到长度达到或超过预期
                k = i + 1
                while k < len(sentences) and current_len < expected_len:
                    current_sentences.append(sentences[k])
                    current_len += len(sentences[k])
                    k += 1

                if current_len == expected_len:
                    # 合并成功
                    result.append("".join(current_sentences))
                    i = k
                    j += 1
                else:
                    # 无法匹配，放弃合并
                    return None
            else:
                # 当前句子太长，无法匹配
                return None

        # 检查是否所有句子都被处理
        if i == len(sentences) and j == len(expected_structure):
            return result

        return None

    def evaluate_tonal(self, title: str, text: str) -> Tuple[float, List[str]]:
        """
        评估平仄准确度

        Returns:
            (准确度 0-1, 实际平仄模式)
        """
        if title not in self.registry:
            return 0.0, []

        standard = self.registry[title]
        tonal_template = standard["tonal_template"]

        sentences = self.parse_generated_text(text)

        total_fixed = 0  # 需要平仄固定的位置总数
        correct = 0  # 符合标准的位置数
        actual_patterns = []

        for sent_idx, (sent, template) in enumerate(zip(sentences, tonal_template)):
            actual_pattern = []

            for char_idx, char in enumerate(sent):
                tone = self.get_char_tone(char)
                actual_pattern.append(tone)

                if char_idx < len(template):
                    expected = template[char_idx]
                    if expected != "*":  # 非 * 位置需要检查
                        total_fixed += 1
                        if tone == expected:
                            correct += 1

            actual_patterns.append("".join(actual_pattern))

        accuracy = correct / total_fixed if total_fixed > 0 else 1.0

        return accuracy, actual_patterns

    def evaluate_tonal_with_sentences(
        self, sentences: List[str], tonal_template: List[str]
    ) -> Tuple[float, List[str]]:
        """
        基于给定句子评估平仄准确度

        Args:
            sentences: 预处理后的句子列表
            tonal_template: 标准平仄模板

        Returns:
            (准确度 0-1, 实际平仄模式)
        """
        total_fixed = 0
        correct = 0
        actual_patterns = []

        for sent_idx, (sent, template) in enumerate(zip(sentences, tonal_template)):
            actual_pattern = []

            for char_idx, char in enumerate(sent):
                tone = self.get_char_tone(char)
                actual_pattern.append(tone)

                if char_idx < len(template):
                    expected = template[char_idx]
                    if expected != "*":
                        total_fixed += 1
                        if tone == expected:
                            correct += 1

            actual_patterns.append("".join(actual_pattern))

        accuracy = correct / total_fixed if total_fixed > 0 else 1.0
        return accuracy, actual_patterns

    def evaluate_rhyme_with_sentences(
        self, sentences: List[str], rhyme_groups: List[List[int]]
    ) -> Tuple[float, Dict]:
        if not rhyme_groups:
            return 1.0, {"message": "该词牌无押韵要求"}

        all_indices = [idx for group in rhyme_groups for idx in group]
        if len(sentences) < max(all_indices) + 1:
            return 0.0, {"error": "句子数量不足"}

        group_results = []
        total_consistency = 0.0

        for group in rhyme_groups:
            rhyme_ids = []
            for idx in group:
                if idx < len(sentences):
                    last_char = sentences[idx][-1] if sentences[idx] else ""
                    if last_char:
                        rhyme_id = self.get_rhyme_id(last_char)
                        if rhyme_id != "UNKNOWN":
                            rhyme_ids.append(rhyme_id)

            if not rhyme_ids:
                group_results.append(
                    {"indices": group, "consistency": 0.0, "rhyme_ids": []}
                )
                continue

            rhyme_counter = Counter(rhyme_ids)
            max_count = rhyme_counter.most_common(1)[0][1]
            consistency = max_count / len(rhyme_ids)

            group_results.append(
                {
                    "indices": group,
                    "consistency": consistency,
                    "rhyme_ids": rhyme_ids,
                    "distribution": dict(rhyme_counter),
                }
            )
            total_consistency += consistency

        avg_consistency = total_consistency / len(rhyme_groups) if rhyme_groups else 0.0

        details = {
            "rhyme_groups": rhyme_groups,
            "group_results": group_results,
            "avg_consistency": avg_consistency,
        }

        return avg_consistency, details

    def evaluate_rhyme(self, title: str, text: str) -> Tuple[float, Dict]:
        if title not in self.registry:
            return 0.0, {}

        standard = self.registry[title]
        rhyme_groups = standard.get("rhyme_indices", [])

        if not rhyme_groups:
            return 1.0, {"message": "该词牌无押韵要求"}

        sentences = self.parse_generated_text(text)
        return self.evaluate_rhyme_with_sentences(sentences, rhyme_groups)

    def evaluate(self, title: str, text: str) -> Dict:
        """
        综合评估生成文本

        Args:
            title: 词牌名
            text: 生成的宋词文本

        Returns:
            评估报告字典
        """
        if title not in self.registry:
            return {"error": f'词牌 "{title}" 不在格律库中', "form_score": 0.0}

        standard = self.registry[title]
        expected_structure = standard["structure"]

        # 1. 结构匹配（包含智能合并）
        raw_sentences = self.parse_generated_text(text, title)
        structure_match, structure_ratio, merged_sentences = self.evaluate_structure(title, text)

        # 使用合并后的句子进行评估
        sentences_to_evaluate = merged_sentences if structure_match else raw_sentences
        actual_structure = [len(s) for s in sentences_to_evaluate]

        # 2. 平仄准确度（基于标准结构的句子）
        tonal_accuracy, actual_patterns = self.evaluate_tonal_with_sentences(
            sentences_to_evaluate, standard["tonal_template"]
        )

        # 3. 押韵一致性
        rhyme_consistency, rhyme_details = self.evaluate_rhyme_with_sentences(
            sentences_to_evaluate, standard["rhyme_indices"]
        )

        # 综合打分 (连续结构分)
        form_score = (
            0.4 * structure_ratio
            + 0.4 * tonal_accuracy
            + 0.2 * rhyme_consistency
        )

        report = {
            "title": title,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "structure": {
                "expected": self.registry[title]["structure"],
                "actual": actual_structure,
                "match": structure_match,
                "ratio": round(structure_ratio, 4),
            },
            "tonal": {
                "expected": self.registry[title]["tonal_template"],
                "actual": actual_patterns,
                "accuracy": round(tonal_accuracy, 4),
            },
            "rhyme": {
                "expected_indices": self.registry[title]["rhyme_indices"],
                "consistency": round(rhyme_consistency, 4),
                "details": rhyme_details,
            },
            "form_score": round(form_score, 4),
        }

        return report

    def evaluate_batch(self, samples: List[Dict]) -> Dict:
        """
        批量评估

        Args:
            samples: 列表，每个元素是 {'title': 词牌名, 'text': 生成文本}

        Returns:
            批量评估统计
        """
        results = []
        structure_matches = 0
        tonal_accuracies = []
        rhyme_consistencies = []
        form_scores = []

        for sample in samples:
            title = sample.get("title", "")
            text = sample.get("text", "")

            report = self.evaluate(title, text)
            results.append(report)

            if "error" not in report:
                if report["structure"]["match"]:
                    structure_matches += 1
                tonal_accuracies.append(report["tonal"]["accuracy"])
                rhyme_consistencies.append(report["rhyme"]["consistency"])
                form_scores.append(report["form_score"])

        # 统计
        total = len(samples)
        stats = {
            "total_samples": total,
            "structure_match_rate": round(structure_matches / total, 4)
            if total > 0
            else 0,
            "avg_tonal_accuracy": round(
                sum(tonal_accuracies) / len(tonal_accuracies), 4
            )
            if tonal_accuracies
            else 0,
            "avg_rhyme_consistency": round(
                sum(rhyme_consistencies) / len(rhyme_consistencies), 4
            )
            if rhyme_consistencies
            else 0,
            "avg_form_score": round(sum(form_scores) / len(form_scores), 4)
            if form_scores
            else 0,
            "details": results,
        }

        return stats


if __name__ == "__main__":
    # 测试代码
    evaluator = Evaluator("standard.json")

    # 测试评估
    test_text = "明月几时有把酒问青天不知天上宫阙今夕是何年"
    report = evaluator.evaluate("水调歌头", test_text)
    print(json.dumps(report, ensure_ascii=False, indent=2))
