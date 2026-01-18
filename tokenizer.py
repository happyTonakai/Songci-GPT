import os
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import jieba
import orjson
from tqdm import tqdm

jieba.dt.tmp_dir = "./.jieba_cache"


class BPETokenizer:
    def __init__(self) -> None:
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.id = 0
        # Initialize special tokens
        self.special_tokens = ["<bos>", "<eos>", "<unknown>", "</w>", "<mask>", "<sep>", "<pad>"]
        for token in self.special_tokens:
            self._add_to_vocab(token)
        self.bos_id = self.encode("<bos>")
        self.eos_id = self.encode("<eos>")
        self.mask_id = self.encode("<mask>")
        self.pad_id = self.encode("<pad>")
        self.sep_id = self.encode("<sep>")

    def _add_to_vocab(self, token: str) -> None:
        if token not in self.token2id:
            self.token2id[token] = self.id
            self.id2token[self.id] = token
            self.id += 1

    def _replace_new_token(self, sequence: list[str], new_token: str) -> list[str]:
        # ['h', 'e', 'l', 'l', 'o', '</w>', 'l', 'o', 'r', 'a', '</w>']
        # 假设new_token是'lo', i = 3
        new_sequence = []
        i = 0
        while i < len(sequence) - 1:
            pair = sequence[i] + sequence[i + 1]
            if pair == new_token:
                new_sequence.append(pair)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        if i == len(sequence) - 1:
            # 说明最后一对不是new_token，把最后一个token也加回去
            new_sequence.append(sequence[-1])
        return new_sequence

    def train(
        self, text_list: list[str], vocab_size: int, max_itr: int, chinese_only: bool = False
    ) -> None:
        self.vocab_size = vocab_size

        # 1. 首先，将所有语料中的独特字符添加到词汇表
        # sorted(list(set(...))) 确保了顺序一致性，这在不同运行环境下很重要
        unique_chars = sorted(list(set(list("".join(text_list)))))
        for char in unique_chars:
            self._add_to_vocab(char)

        # 2. 对语料库做预处理
        # 如果你不进行预处理和添加 </w>，BPE 仍然会运行，它会合并整个文本流中最常见的字符对。
        # 但是，你得到的子词可能会包含很多跨越单词边界的无意义组合，这会使得后续的 NLP 模型（如翻译模型、文本生成模型）更难理解和生成高质量的文本。

        # 为所有英文单词加上Special token </w> 表示一个单词的结束
        word_sequences = []
        for text in text_list:
            if not chinese_only:
                en_words = re.findall(r"[a-zA-Z0-9]+", text)
                for word in en_words:
                    word_sequences.append(list(word) + ["</w>"])
                # [['h', 'e', 'l', 'l', 'o', '</w>'], ['w', 'o', 'r', 'l', 'd', '</w>'], ...]
            text = text.replace("<sep>", " <sep> ")  # 确保 <sep> 被独立切分
            ch_words = jieba.lcut(text)
            # 检查是否是标点符号或空白，如果不是，则添加 </w>
            # 对于中文，有些时候标点也会作为单独的token
            for word in ch_words:
                if word.strip():  # 忽略空格
                    if word.strip() == "<sep>":
                        word_sequences.append(["<sep>"])  # <sep> 是一个原子词元
                    elif re.fullmatch(r"[\u4e00-\u9fa5]+", word):  # 是否为纯汉字
                        word_sequences.append(list(word) + ["</w>"])
                    else:  # 可能是数字，英文，或者标点符号，作为独立词元
                        word_sequences.append(list(word))
            # [['你', '好', '</w>'], ['世', '界', '</w>'], ...]

        # 3. 开始合并
        pbar = tqdm(range(max_itr), desc="BPE Merging")
        for _ in pbar:
            # 实时更新 desc
            pbar.set_description(f"Current vocab size: {len(self.token2id)}")
            if len(self.token2id) > self.vocab_size:
                print("词表大小达到上限，停止训练")
                break  # 如果词表大小达到上限，停止训练

            # 统计频率： 遍历所有当前的“词”（无论是单个字符还是已合并的子词序列），统计所有相邻字符对的出现频率。
            pair_count = {}
            for sequence in word_sequences:
                # ['h', 'e', 'l', 'l', 'o', '</w>', 'w', 'o', 'r', 'l', 'd', '</w>']
                for i in range(len(sequence) - 1):
                    pair = (sequence[i], sequence[i + 1])  # ('h', 'e')
                    pair_count[pair] = pair_count.get(pair, 0) + 1

            if not pair_count:  # 所有组合都已经在词表中，没有可以继续合并的对了，停止训练
                print("所有组合都已经在词表中，没有可以继续合并的对了，停止训练")
                break

            # 选择最常见对： 找到频率最高的那个字符对。
            best_pair = max(pair_count, key=lambda k: pair_count[k])

            # 合并： 将这个最常见的字符对合并成一个新的词元。这个新的词元将被添加到你的词汇表中。
            new_token = "".join(best_pair)
            self._add_to_vocab(new_token)

            # 替换： 在整个语料中，将所有出现的最常见字符对都替换为这个新的词元。
            new_word_sequences = []
            for sequence in word_sequences:
                new_sentence = self._replace_new_token(sequence, new_token)
                new_word_sequences.append(new_sentence)

            # 重复： 重复上述过程（统计频率，选择最常见，合并，替换），直到：
            #       你达到了预设的目标词汇表大小（例如，你希望最终词汇表有 10000 个词元）。
            #       你达到了预设的合并操作次数。
            #       或者，没有更多的字符对可以合并了（所有字符对的频率都只有1）。
            word_sequences = new_word_sequences

    def encode(self, text: str) -> list[int]:
        # 将文本编码为整数序列，返回token id
        encoded_ids = []
        i = 0
        while i < len(text):
            matched = False
            # 尝试匹配特殊标记
            for st_token in self.special_tokens:
                if text[i:].startswith(st_token):
                    encoded_ids.append(self.token2id.get(st_token, self.token2id["<unknown>"]))
                    i += len(st_token)
                    matched = True
                    break
            if not matched:
                # 如果没有匹配到特殊标记，则按单个字符处理
                encoded_ids.append(self.token2id.get(text[i], self.token2id["<unknown>"]))
                i += 1
        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        # 将整数序列解码为文本
        return "".join([self.id2token.get(id, "<unknown>") for id in ids])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = self.__dict__
        state_dict["id2token"] = {str(k): v for k, v in self.id2token.items()}
        if "id" in state_dict:
            del state_dict["id"]
        json_bytes = orjson.dumps(
            state_dict, option=orjson.OPT_INDENT_2  # 可选：格式化输出（缩进2个空格），方便阅读
        )
        with open(path, "wb") as f:
            f.write(json_bytes)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            self.__dict__ = orjson.loads(f.read())
            self.id2token = {int(k): v for k, v in self.id2token.items()}


def test_replace_new_token():
    tokenizer = BPETokenizer()
    sequence = ["h", "e", "l", "l", "o", "</w>", "l", "o", "r", "a", "</w>"]
    new_token = "lo"
    new_sequence = tokenizer._replace_new_token(sequence, new_token)
    print(new_sequence)
    new_token = "a</w>"
    new_sequence = tokenizer._replace_new_token(sequence, new_token)
    print(new_sequence)


def train_simple_tokenizer():
    tokenizer = BPETokenizer()
    with open("./dataset/train-en.txt", "r", encoding="utf-8") as f:
        en_text_list = f.readlines()
    with open("./dataset/train-zh.txt", "r", encoding="utf-8") as f:
        zh_text_list = f.readlines()
    tokenizer.train(en_text_list + zh_text_list, vocab_size=5000, max_itr=1000)

    tokenizer.save("./ckpt/tokenizer.json")
    tokenizer.load("./ckpt/tokenizer.json")
    test_text = "hello world! 你好世界。"
    token_ids = tokenizer.encode(test_text)
    print(token_ids)
    decoded_text = tokenizer.decode(token_ids)
    print(decoded_text)
    for id in token_ids:
        print(id, tokenizer.decode([id]))


def train_songci_tokenizer():
    from glob import glob

    tokenizer = BPETokenizer()  # 初始化词表大小 6103

    dataset = []
    for file in glob("./dataset/宋词/*.json"):
        with open(file, "rb") as f:
            data = orjson.loads(f.read())
            dataset += [item["rhythmic"] + "<sep>" + "".join(item["paragraphs"]) for item in data]

    try:
        tokenizer.train(dataset, 10000, 1000)
    except KeyboardInterrupt:
        print("训练被中断，保存词表")
    except Exception as e:
        print(f"训练发生错误：{e}，准备保存词表")
    finally:
        tokenizer.save("./ckpt/songci_tokenizer.json")
        tokenizer.load("./ckpt/songci_tokenizer.json")
        test_text = "春江潮水连海平，海上明月共潮生。"
        token_ids = tokenizer.encode(test_text)
        print(token_ids)
        decoded_text = tokenizer.decode(token_ids)
        print(decoded_text)


if __name__ == "__main__":

    # train_simple_tokenizer()

    train_songci_tokenizer()
