"""
RegistryBuilder - 格律先验库构建模块

功能：
1. Top 50 词牌筛选
2. 样本清洗与众数过滤
3. 平仄统计 (0.8 阈值)
4. 韵律定位与韵部映射
"""

import glob
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from pypinyin import Style, pinyin


class RegistryBuilder:
    """
    宋词格律库构建器

    从训练语料中自动提取格律标准，包括：
    - 结构标准 (句数、每句字数)
    - 平仄模板
    - 押韵位置
    """

    # 韵部映射表 (简版)
    RHYME_MAPPING = {
        # Group_AN
        "an": "AN",
        "ian": "AN",
        "uan": "AN",
        "üan": "AN",
        "van": "AN",
        # Group_ANG
        "ang": "ANG",
        "iang": "ANG",
        "uang": "ANG",
        # Group_EN
        "en": "EN",
        "in": "EN",
        "un": "EN",
        "ün": "EN",
        "vn": "EN",
        # Group_ENG
        "eng": "ENG",
        "ing": "ENG",
        "ueng": "ENG",
        "ong": "ENG",
        "iong": "ENG",
        # Group_AO
        "ao": "AO",
        "iao": "AO",
        # Group_OU
        "ou": "OU",
        "iu": "OU",
        # Group_AI
        "ai": "AI",
        "uai": "AI",
        # Group_EI
        "ei": "EI",
        "ui": "EI",
        "uei": "EI",
        # Group_A
        "a": "A",
        "ia": "A",
        "ua": "A",
        # Group_O
        "o": "O",
        "uo": "O",
        "io": "O",
        # Group_E
        "e": "E",
        "ie": "E",
        "ue": "E",
        "üe": "E",
        "ve": "E",
        # Group_I
        "i": "I",
        "er": "I",
        "-i": "I",
        # Group_U
        "u": "U",
        # Group_V (ü 单独处理)
        "ü": "V",
        "v": "V",
    }

    def __init__(
        self,
        data_path: str,
        top_n: int = None,
        min_samples: int = 50,
        confidence: float = 0.8,
    ):
        """
        Args:
            data_path: 宋词 JSON 文件路径 (支持通配符)
            top_n: 保留的词牌数量 (与 min_samples 二选一，默认 None)
            min_samples: 最小样本量阈值，只保留样本数 >= 此值的词牌 (默认 50)
            confidence: 平仄判定阈值 (默认 0.8)
        """
        self.data_path = data_path
        self.top_n = top_n
        self.min_samples = min_samples
        self.confidence = confidence
        self.registry = {}

    def load_data(self) -> List[Dict]:
        """加载所有宋词数据"""
        all_data = []
        files = glob.glob(self.data_path)
        print(f"找到 {len(files)} 个数据文件")

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                print(f"警告: 无法读取文件 {file_path}: {e}")

        print(f"总共加载 {len(all_data)} 首宋词")
        return all_data

    def filter_titles(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        筛选词牌（按 top_n 或 min_samples）

        Returns:
            Dict[词牌名, 该词牌的所有样本]
        """
        # 统计词牌频率
        title_counter = Counter(item["rhythmic"] for item in data if "rhythmic" in item)
        total_titles = len(title_counter)

        # 根据参数决定筛选策略
        if self.top_n is not None:
            # 使用 Top N 策略
            selected_titles = [
                title for title, _ in title_counter.most_common(self.top_n)
            ]
            print(f"\n筛选策略: Top {self.top_n}")
            print(f"总共 {total_titles} 个词牌，选中 {len(selected_titles)} 个")
            for i, (title, count) in enumerate(
                title_counter.most_common(self.top_n), 1
            ):
                print(f"  {i:2d}. {title}: {count}首")
        else:
            # 使用 min_samples 阈值策略
            selected_titles = [
                title
                for title, count in title_counter.items()
                if count >= self.min_samples
            ]
            selected_titles.sort(key=lambda t: title_counter[t], reverse=True)

            total_covered = sum(title_counter[t] for t in selected_titles)
            print(f"\n筛选策略: 样本量 >= {self.min_samples} 首")
            print(f"总共 {total_titles} 个词牌，选中 {len(selected_titles)} 个")
            print(
                f"覆盖样本: {total_covered}首 ({total_covered / len(data) * 100:.1f}%)"
            )

            print("\n选中的词牌统计:")
            for i, title in enumerate(selected_titles[:20], 1):
                print(f"  {i:2d}. {title}: {title_counter[title]}首")
            if len(selected_titles) > 20:
                print(f"  ... 还有 {len(selected_titles) - 20} 个词牌")

        # 按词牌分组
        grouped = defaultdict(list)
        for item in data:
            if item.get("rhythmic") in selected_titles:
                grouped[item["rhythmic"]].append(item)

        return dict(grouped)

    def get_structure(self, item: Dict) -> Tuple[int, int, List[int]]:
        """
        获取词的结构信息

        智能处理：对于特定词牌（如浣溪沙），识别并拆分合并的段落

        Returns:
            (句数, 总字数, 每句字数列表)
        """
        rhythmic = item.get("rhythmic", "")
        paragraphs = item.get("paragraphs", [])
        if not paragraphs:
            return 0, 0, []

        # 清理每句（去除标点）
        sentences = []
        for para in paragraphs:
            # 移除常见标点
            cleaned = (
                para.replace("，", "")
                .replace("。", "")
                .replace("、", "")
                .replace("！", "")
                .replace("？", "")
                .replace("；", "")
                .replace("：", "")
                .replace('"', "")
                .replace('"', "")
                .replace('"', "")
                .replace('"', "")
                .replace("（", "")
                .replace("）", "")
                .replace("【", "")
                .replace("】", "")
                .replace("《", "")
                .replace("》", "")
                .replace("〈", "")
                .replace("〉", "")
                .replace(".", "")
                .replace(",", "")
                .replace("!", "")
                .replace("?", "")
                .replace(":", "")
                .replace(";", "")
                .strip()
            )
            if cleaned:
                length = len(cleaned)

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
                if length == 14 and rhythmic in split_14_titles:
                    sentences.extend([7, 7])
                else:
                    sentences.append(length)

        total_sentences = len(sentences)
        total_chars = sum(sentences)

        return total_sentences, total_chars, sentences

    def mode_filter(
        self, samples: List[Dict]
    ) -> Tuple[List[Dict], Tuple[int, int, List[int]]]:
        """
        众数过滤：保留符合最频繁结构的样本

        Returns:
            (过滤后的样本列表, 标准结构)
        """
        # 统计所有结构
        structures = []
        for item in samples:
            struct = self.get_structure(item)
            if struct[0] > 0:  # 过滤空样本
                structures.append((struct, item))

        if not structures:
            return [], (0, 0, [])

        # 找出众数结构 (句数, 总字数, 每句字数列表)
        structure_counter = Counter(
            (s[0][0], s[0][1], tuple(s[0][2])) for s in structures
        )
        mode_structure_tuple, mode_count = structure_counter.most_common(1)[0]
        mode_structure = (
            mode_structure_tuple[0],
            mode_structure_tuple[1],
            list(mode_structure_tuple[2]),
        )

        # 过滤：保留符合众数结构的样本
        filtered = []
        for struct, item in structures:
            if (
                struct[0] == mode_structure[0]
                and struct[1] == mode_structure[1]
                and struct[2] == mode_structure[2]
            ):
                filtered.append(item)

        print(
            f"    众数结构: {mode_structure[0]}句, {mode_structure[1]}字, 每句{mode_structure[2]}"
        )
        print(f"    过滤前: {len(samples)}首, 过滤后: {len(filtered)}首")

        return filtered, mode_structure

    def get_char_tone(self, char: str) -> str:
        """
        获取单个字的平仄

        Returns:
            'P' (平声), 'Z' (仄声), or '?' (无法判定)
        """
        if not char or len(char) != 1:
            return "?"

        try:
            # 获取拼音和声调
            py = pinyin(char, style=Style.TONE3, heteronym=False)
            if not py or not py[0]:
                return "?"

            py_str = py[0][0]
            if not py_str:
                return "?"

            # 提取声调
            tone = None
            for c in py_str:
                if c.isdigit():
                    tone = int(c)
                    break

            if tone is None:
                return "?"

            # 1,2声为平 (P)，3,4声为仄 (Z)
            if tone in [1, 2]:
                return "P"
            elif tone in [3, 4]:
                return "Z"
            else:
                return "?"
        except Exception:
            return "?"

    def get_rhyme_id(self, char: str) -> str:
        """
        获取字的韵部 ID

        Returns:
            韵部 ID (如 'AN', 'ANG', 'EN' 等)，无法识别返回原韵母
        """
        if not char or len(char) != 1:
            return "UNKNOWN"

        try:
            # 获取韵母
            py = pinyin(char, style=Style.FINALS, heteronym=False)
            if not py or not py[0]:
                return "UNKNOWN"

            final = py[0][0].lower()

            # 处理 ü 的情况
            if "ü" in final or "v" in final:
                # 尝试匹配含 ü 的韵母
                for rhyme, group in self.RHYME_MAPPING.items():
                    if rhyme in final:
                        return group

            # 查找映射
            return self.RHYME_MAPPING.get(final, final.upper() if final else "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    def get_sentences_from_paragraphs(
        self, item: Dict, structure: List[int]
    ) -> List[str]:
        """
        从段落中提取句子，支持智能拆分

        Returns:
            句子列表（清理后的文本）
        """
        rhythmic = item.get("rhythmic", "")
        paragraphs = item.get("paragraphs", [])
        sentences = []

        for para in paragraphs:
            # 清理标点
            cleaned = (
                para.replace("，", "")
                .replace("。", "")
                .replace("、", "")
                .replace("！", "")
                .replace("？", "")
                .replace("；", "")
                .replace("：", "")
                .replace('"', "")
                .replace('"', "")
                .replace('"', "")
                .replace('"', "")
                .replace("（", "")
                .replace("）", "")
                .replace("【", "")
                .replace("】", "")
                .replace("《", "")
                .replace("》", "")
                .replace("〈", "")
                .replace("〉", "")
                .replace(".", "")
                .replace(",", "")
                .replace("!", "")
                .replace("?", "")
                .replace(":", "")
                .replace(";", "")
                .strip()
            )

            if not cleaned:
                continue

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
            if len(cleaned) == 14 and rhythmic in split_14_titles:
                sentences.append(cleaned[:7])
                sentences.append(cleaned[7:])
            else:
                sentences.append(cleaned)

        return sentences

    def analyze_tonal_pattern(
        self, samples: List[Dict], structure: List[int]
    ) -> List[str]:
        """
        分析平仄模板

        Args:
            samples: 过滤后的样本
            structure: 每句字数列表

        Returns:
            平仄模板列表，每个元素是一个字符串 (如 "PZPZZPP")
        """
        if not samples or not structure:
            return []

        num_sentences = len(structure)

        # 统计每个位置的平仄
        tone_votes = [
            [{"P": 0, "Z": 0} for _ in range(sentence_len)]
            for sentence_len in structure
        ]

        for item in samples:
            # 使用智能拆分获取句子
            sentences = self.get_sentences_from_paragraphs(item, structure)

            for sentence_idx, cleaned in enumerate(sentences):
                if sentence_idx >= num_sentences:
                    break

                expected_len = structure[sentence_idx]
                if len(cleaned) != expected_len:
                    continue

                # 统计每个字的平仄
                for char_idx, char in enumerate(cleaned):
                    if char_idx < expected_len:
                        tone = self.get_char_tone(char)
                        if tone in ["P", "Z"]:
                            tone_votes[sentence_idx][char_idx][tone] += 1

        # 生成平仄模板
        tonal_template = []
        for sent_idx, sentence_votes in enumerate(tone_votes):
            sentence_pattern = []
            total_samples = len(samples)

            for char_votes in sentence_votes:
                p_count = char_votes["P"]
                z_count = char_votes["Z"]
                total = p_count + z_count

                if total == 0:
                    sentence_pattern.append("*")  # 无数据
                else:
                    p_ratio = p_count / total
                    z_ratio = z_count / total

                    if p_ratio > self.confidence:
                        sentence_pattern.append("P")
                    elif z_ratio > self.confidence:
                        sentence_pattern.append("Z")
                    else:
                        sentence_pattern.append("*")  # 平仄不拘

            tonal_template.append("".join(sentence_pattern))

        return tonal_template

    def analyze_rhyme_positions(
        self, samples: List[Dict], structure: List[int]
    ) -> List[int]:
        """
        分析押韵位置

        Args:
            samples: 过滤后的样本
            structure: 每句字数列表

        Returns:
            押韵句的索引列表 (0-based)
        """
        if not samples or not structure:
            return []

        num_sentences = len(structure)
        rhyme_votes = [defaultdict(int) for _ in range(num_sentences)]

        for item in samples:
            # 使用智能拆分获取句子
            sentences = self.get_sentences_from_paragraphs(item, structure)

            for sentence_idx, cleaned in enumerate(sentences):
                if sentence_idx >= num_sentences:
                    break

                expected_len = structure[sentence_idx]
                if len(cleaned) != expected_len:
                    continue

                # 获取句末字的韵部
                last_char = cleaned[-1]
                rhyme_id = self.get_rhyme_id(last_char)
                rhyme_votes[sentence_idx][rhyme_id] += 1

        # 判定押韵位置
        rhyme_indices = []
        total_samples = len(samples)

        for sent_idx, votes in enumerate(rhyme_votes):
            if not votes:
                continue

            # 找出最频繁的韵部
            most_common_rhyme, count = max(votes.items(), key=lambda x: x[1])
            ratio = count / total_samples

            # 如果某韵部占比超过阈值，且该韵部不是 UNKNOWN，则判定为押韵位置
            if ratio > self.confidence and most_common_rhyme != "UNKNOWN":
                rhyme_indices.append(sent_idx)

        return rhyme_indices

    def build(self) -> Dict:
        """
        构建格律库

        Returns:
            格律库字典，格式符合 metadata.json 规范
        """
        print("=" * 60)
        print("SongEval RegistryBuilder - 构建格律先验库")
        print("=" * 60)

        # 1. 加载数据
        data = self.load_data()

        # 2. 筛选词牌
        grouped = self.filter_titles(data)

        # 3. 对每个词牌进行分析
        print("\n" + "=" * 60)
        print("开始分析各词牌格律...")
        print("=" * 60)

        for title, samples in grouped.items():
            print(f"\n【{title}】")

            # 3.1 众数过滤
            filtered_samples, structure_info = self.mode_filter(samples)

            if not filtered_samples:
                print("    警告: 过滤后无有效样本，跳过")
                continue

            num_sentences, total_chars, sentence_lengths = structure_info

            # 3.2 平仄统计
            tonal_template = self.analyze_tonal_pattern(
                filtered_samples, sentence_lengths
            )

            # 3.3 韵律定位
            rhyme_indices = self.analyze_rhyme_positions(
                filtered_samples, sentence_lengths
            )

            # 保存结果
            self.registry[title] = {
                "structure": sentence_lengths,
                "tonal_template": tonal_template,
                "rhyme_indices": rhyme_indices,
                "sample_size": len(filtered_samples),
            }

            print(f"    平仄模板: {tonal_template}")
            print(f"    押韵位置: {rhyme_indices}")

        print("\n" + "=" * 60)
        print(f"格律库构建完成！共 {len(self.registry)} 个词牌")
        print("=" * 60)

        return self.registry

    def save(self, output_path: str):
        """保存格律库到 JSON 文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
        print(f"\n格律库已保存到: {output_path}")


if __name__ == "__main__":
    # 测试代码
    builder = RegistryBuilder("../../dataset/宋词/*.json", top_n=50, confidence=0.8)
    registry = builder.build()
    builder.save("standard.json")
