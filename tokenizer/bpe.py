import re
import json
from collections import Counter


class BPETokenizer:

    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>"
    ]

    def __init__(self, merges=None):
        self.merges = merges or []

        self.token_to_id = {}
        self.id_to_token = {}

    # ==========================================================
    # BPE TRAINING
    # ==========================================================

    @staticmethod
    def get_vocab(text):
        vocab = Counter()

        words = re.findall(r"\w+|[^\w\s]", text)

        for word in words:
            tokens = " ".join(list(word)) + " </w>"
            vocab[tokens] += 1

        return vocab

    @staticmethod
    def get_pair_frequencies(vocab):

        pairs = Counter()

        for word, freq in vocab.items():

            symbols = word.split()

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq

        return pairs

    @staticmethod
    def merge_vocab(pair, vocab):

        new_vocab = {}

        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]

        return new_vocab

    @classmethod
    def train_bpe(cls, text, num_merges=1000):

        vocab = cls.get_vocab(text)

        merges = []

        for _ in range(num_merges):

            pairs = cls.get_pair_frequencies(vocab)

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            vocab = cls.merge_vocab(best_pair, vocab)

            merges.append(best_pair)

        return merges

    # ==========================================================
    # TOKENIZATION
    # ==========================================================

    def encode_word(self, word):

        tokens = list(word) + ["</w>"]

        for pair in self.merges:

            i = 0

            while i < len(tokens) - 1:

                if (tokens[i], tokens[i + 1]) == pair:
                    tokens[i:i + 2] = ["".join(pair)]
                else:
                    i += 1

        return tokens

    def encode(self, text, add_special_tokens=False):

        words = re.findall(r"\w+|[^\w\s]", text)

        output_tokens = []

        if add_special_tokens:
            output_tokens.append("<BOS>")

        for word in words:
            output_tokens.extend(self.encode_word(word))

        if add_special_tokens:
            output_tokens.append("<EOS>")

        return output_tokens

    def decode(self, tokens):

        filtered = []

        for token in tokens:

            if token in self.SPECIAL_TOKENS:
                continue

            filtered.append(token)

        text = "".join(filtered)

        text = text.replace("</w>", " ")

        return text.strip()

    # ==========================================================
    # VOCABULARY
    # ==========================================================

    def build_vocab(self, corpus_tokens):

        unique_tokens = sorted(set(corpus_tokens))

        self.token_to_id = {}
        self.id_to_token = {}

        idx = 0

        for token in self.SPECIAL_TOKENS:

            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        for token in unique_tokens:

            if token in self.token_to_id:
                continue

            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

            idx += 1

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    # ==========================================================
    # ID CONVERSION
    # ==========================================================

    def encode_to_ids(self, text, add_special_tokens=True):

        tokens = self.encode(
            text,
            add_special_tokens=add_special_tokens
        )

        unk_id = self.token_to_id["<UNK>"]

        ids = [
            self.token_to_id.get(token, unk_id)
            for token in tokens
        ]

        return ids

    def decode_from_ids(self, ids):

        tokens = [
            self.id_to_token.get(
                idx,
                "<UNK>"
            )
            for idx in ids
        ]

        return self.decode(tokens)

    # ==========================================================
    # SAVE / LOAD
    # ==========================================================

    def save_vocab(self, path):

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.token_to_id,
                f,
                indent=2,
                ensure_ascii=False
            )

    def load_vocab(self, path):

        with open(path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)

        self.id_to_token = {
            int(v): k
            for k, v in self.token_to_id.items()
        }

    def save_merges(self, path):

        serializable = [
            list(pair)
            for pair in self.merges
        ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                serializable,
                f,
                indent=2,
                ensure_ascii=False
            )

    def load_merges(self, path):

        with open(path, "r", encoding="utf-8") as f:

            loaded = json.load(f)

        self.merges = [
            tuple(pair)
            for pair in loaded
        ]