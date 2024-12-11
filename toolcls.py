import csv
import os
import random
import numpy as np
import torch
import json
import copy
from torchvision import transforms
from PIL import Image
from logs import logger
from timm.data.auto_augment import RandAugment, rand_augment_ops
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def process_tsv_file(file_path, tokenizer, max_img_length=128):
    """
    读取TSV文件，提取指定列并返回分词结果。

    参数：
    - file_path (str): TSV文件的路径。
    - max_length (int): 最大分词长度（默认128）。

    返回：
    - image_text_tokens (list): 第二列分词结果。
    - text_entity_tokens (list): 第三列和第四列拼接后的分词结果。
    """

    # 读取TSV文件，不设置列标题
    df = pd.read_csv(file_path, sep="\t", header=None, skiprows=1)

    # 初始化结果列表
    labels = []
    image_text_tokens = []
    for i, row in df.iterrows():
        # 读取第1列作为label
        label = row[1]  # 第1列（列号0）
        labels.append(label)

        # 读取第2列作为图像描述分词
        image_text = row[2]  # 第2列（列号1）
        text = row[3]
        entity_text = row[4]  # 第4列（列号3）
        entity_text_img = text + image_text
        image_text_encoded = tokenizer(
            entity_text,
            entity_text_img,
            padding="max_length",
            truncation=True,
            max_length=max_img_length,
            return_tensors="pt"
        )
        image_text_tokens.append(image_text_encoded)

        if i == len(df) - 1:
            print("the last row of data: ", row.tolist())
    return labels, image_text_tokens


class TokenizedDataset(Dataset):
    """
    自定义Dataset，用于存储分词后的结果以及标签。
    """

    def __init__(self, labels, image_text_tokens):
        """
        参数：
        - labels (list): 标签列表。
        - image_text_tokens (list of dict): 图像文本分词结果，每项是字典。
        - text_entity_tokens (list of dict): 文本拼接后分词结果，每项是字典。
        """
        self.labels = labels
        self.image_text_tokens = image_text_tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回指定索引的数据项。
        """
        return {
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "image_text_input_ids": self.image_text_tokens[idx]["input_ids"].squeeze(0),
            "image_text_attention_mask": self.image_text_tokens[idx]["attention_mask"].squeeze(0),
            "image_text_token_type_ids": self.image_text_tokens[idx]["token_type_ids"].squeeze(0),
        }


def create_dataloader(labels, image_text_tokens, shuffle, batch_size=16):
    """
    根据分词结果和标签构造 DataLoader。

    参数：
    - labels (list): 标签列表。
    - image_text_tokens (list of dict): 图像文本分词结果。
    - text_entity_tokens (list of dict): 文本拼接后分词结果。
    - batch_size (int): DataLoader 的批量大小。

    返回：
    - dataloader (DataLoader): PyTorch DataLoader。
    """
    dataset = TokenizedDataset(labels, image_text_tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def save_dataloader(dataloader, save_path, expand):
    """
    将 DataLoader 的数据保存到文件。
    """
    # 提取 DataLoader 中的所有数据并保存为列表
    data = list(dataloader)
    file_path = os.path.join(save_path, expand)
    torch.save(data, file_path)


def load_dataloader(save_path, expand, shuffle, batch_size=16):
    """
    从文件加载 DataLoader。
    """
    # 加载保存的数据
    file_path = os.path.join(save_path, expand)
    data = torch.load(file_path)

    # 将数据重新封装为 Dataset
    dataset = TokenizedDataset(
        labels=[item["labels"].item() for batch in data for item in batch],
        image_text_tokens=[
            {"input_ids": item["image_text_input_ids"],
             "attention_mask": item["image_text_attention_mask"],
             "token_type_ids": item["image_text_token_type_ids"]
             }
            for batch in data for item in batch
        ],
    )

    # 重新构造 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
