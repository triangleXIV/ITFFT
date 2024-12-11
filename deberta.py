from mpmath import extend

from logs import logger
import math
import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoModel
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        """
        SwiGLU 激活函数模块。

        Args:
            hidden_dim (int): 分割后每个部分的维度。
            dropout_rate (float): Dropout 的概率。
        """
        super(SwiGLU, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (..., 2 * hidden_dim)

        Returns:
            Tensor: 输出张量，形状为 (..., hidden_dim)
        """
        x1, x2 = x.chunk(2, dim=-1)  # 将张量沿最后一个维度拆分为两部分
        return self.dropout(F.silu(x1) * x2)  # 应用 SiLU 激活并逐元素相乘，然后应用 Dropout


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        """
        前馈神经网络（FFN）层，使用 SwiGLU 作为激活函数。

        Args:
            input_dim (int): 输入维度。
            hidden_dim (int, optional): 隐藏层维度。默认为 4 * input_dim。
            dropout_rate (float, optional): Dropout 的概率。默认为 0.1。
        """
        super(FFN, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * input_dim  # 通常 FFN 的隐藏层维度是输入维度的 4 倍

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),  # 第一个线性层，扩展维度以适应 SwiGLU
            SwiGLU(hidden_dim, dropout_rate),  # SwiGLU 激活函数
            nn.Linear(hidden_dim, input_dim),  # 第二个线性层，恢复到输入维度
            nn.Dropout(dropout_rate)  # 最后的 Dropout
        )

    def forward(self, x):
        """
        前向传播。
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, input_dim)

        Returns:
            Tensor: 输出张量，形状为 (batch_size, input_dim)
        """
        return self.ffn(x)


# 这一块是自注意力融合多模态特征
class MultimodalEncoder(nn.Module):
    def __init__(self, config):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(2)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers[-1]


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        # 检查 hidden_size 是否能被 3 * num_attention_heads 整除
        if config.hidden_size % 3  != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of 3 times the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = config.hidden_size
        self.attention_head_size=self.all_head_size//self.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)  # 输出维度为 all_head_size
        self.key = nn.Linear(config.hidden_size, config.hidden_size // 3)  # 输出维度为 config.hidden_size / 3
        self.value = nn.Linear(config.hidden_size, config.hidden_size // 3)  # 输出维度为 config.hidden_size / 3

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

    def forward(self, hidden_states, attention_mask):
        # 线性变换
        mixed_query_layer = self.query(hidden_states)  # (batch_size, seq_length, all_head_size)
        mixed_key_layer = self.key(hidden_states)  # (batch_size, seq_length, all_head_size / 3)
        mixed_value_layer = self.value(hidden_states)  # (batch_size, seq_length, all_head_size / 3)

        mixed_key_layer = torch.cat([mixed_key_layer, mixed_key_layer, mixed_key_layer],
                                    dim=-1)  # (batch_size, num_heads, seq_length_k, head_dim)
        mixed_value_layer = torch.cat([mixed_value_layer, mixed_value_layer, mixed_value_layer],
                                      dim=-1)  # (batch_size, num_heads, seq_length_k, head_dim)

        # 转置为多头格式
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (batch_size, num_heads, seq_length, head_dim)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (batch_size, num_heads, seq_length, head_dim_k)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (batch_size, num_heads, seq_length, head_dim_k)

        # 计算注意力分数
        # query_layer: (batch_size, num_heads, seq_length, all_head_size / 3)
        # key_layer.transpose(-1, -2): (batch_size, num_heads, all_head_size / 3, seq_length)
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))  # (batch_size, num_heads, seq_length, seq_length)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力掩码
        attention_scores = attention_scores + attention_mask  # attention_mask 应为 (batch_size, 1, 1, seq_length) 或兼容形状

        # 归一化注意力分数
        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # 应用 Dropout
        attention_probs = self.dropout(attention_probs)

        # 计算上下文层
        # attention_probs: (batch_size, num_heads, seq_length, seq_length)
        # value_layer: (batch_size, num_heads, seq_length, all_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)  # (batch_size, num_heads, seq_length, all_head_size)

        # 转置并合并多头
        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # (batch_size, seq_length, num_heads, all_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch_size, seq_length, all_head_size)

        return context_layer


# 这一块是交叉注意力融合多模态特征
class BertCrossEncoder(nn.Module):
    def __init__(self, config):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(4)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers[-1]


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % 3  != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of 3 times the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size=config.hidden_size
        self.attention_head_size = self.all_head_size // self.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)  # 输出维度为 all_head_size
        self.key = nn.Linear(config.hidden_size, config.hidden_size // 3)  # 输出维度为 config.hidden_size / 3
        self.value = nn.Linear(config.hidden_size, config.hidden_size // 3)  # 输出维度为 config.hidden_size / 3

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        # 线性变换
        mixed_query_layer = self.query(s1_hidden_states)  # (batch_size, seq_length_q, all_head_size)
        mixed_key_layer = self.key(s2_hidden_states)  # (batch_size, seq_length_k, all_head_size / 3)
        mixed_value_layer = self.value(s2_hidden_states)  # (batch_size, seq_length_k, all_head_size / 3)

        mixed_key_layer = torch.cat([mixed_key_layer, mixed_key_layer, mixed_key_layer],
                              dim=-1)  # (batch_size, num_heads, seq_length_k, head_dim)
        mixed_value_layer = torch.cat([mixed_value_layer, mixed_value_layer, mixed_value_layer],
                                dim=-1)  # (batch_size, num_heads, seq_length_k, head_dim)

        # 转置为多头格式
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (batch_size, num_heads, seq_length_q, head_dim)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (batch_size, num_heads, seq_length_k, head_dim_k)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (batch_size, num_heads, seq_length_k, head_dim_k)

        # 计算注意力分数
        # query_layer: (batch_size, num_heads, seq_length_q, head_dim)
        # key_layer.transpose(-1, -2): (batch_size, num_heads, head_dim, seq_length_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # (batch_size, num_heads, seq_length_q, seq_length_k)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力掩码
        attention_scores = attention_scores + s2_attention_mask  # s2_attention_mask 应为 (batch_size, 1, 1, seq_length_k) 或兼容形状

        # 归一化注意力分数
        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_length_q, seq_length_k)

        # 应用 Dropout
        attention_probs = self.dropout(attention_probs)

        # 计算上下文层
        # attention_probs: (batch_size, num_heads, seq_length_q, seq_length_k)
        # value_layer: (batch_size, num_heads, seq_length_k, head_dim)
        context_layer = torch.matmul(attention_probs, value_layer)  # (batch_size, num_heads, seq_length_q, head_dim)

        # 转置并合并多头
        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # (batch_size, seq_length_q, num_heads, head_dim)
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,)  # (batch_size, seq_length_q, all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch_size, seq_length_q, all_head_size)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}
AlbertLayerNorm = torch.nn.LayerNorm
BertLayerNorm = torch.nn.LayerNorm


class ITFFT(nn.Module):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, args, bert_config):
        super(ITFFT, self).__init__()
        self.num_labels = 3
        self.sentence_extract = AutoModel.from_pretrained(args.deberta_dir)

        self.LN1 = nn.LayerNorm(bert_config.hidden_size, eps=1e-6)
        self.ffn = FFN(bert_config.hidden_size)

        self.classifier = nn.Linear(bert_config.hidden_size, 3)

    def forward(self, img_text_ids, img_text_mask,image_text_type_ids, labels=None):
        sequence_output = self.sentence_extract(input_ids=img_text_ids,
                                                attention_mask=img_text_mask,
                                                token_type_ids=image_text_type_ids)
        # output返回的是 特征提取后的完整结果
        sequence_feature = sequence_output[0]

        output = self.LN1(sequence_feature + self.ffn(sequence_feature))

        cls_token = output[:, 0, :]
        logits = self.classifier(cls_token)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss,logits
        else:
            return logits
