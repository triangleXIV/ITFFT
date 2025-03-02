# ITFFT, Image-to-Text Feature Fusion Transformer
<hr>
随着大模型的发展，似乎在预处理阶段将图像转换为一段文本的描述已经变得愈发准确了，所以这个项目就是通过通义千问大模型，把Twitter15/17的情感分析数据集中的图像，通过提示词转换为了一段文本描述，这样的话，多模态情感分析任务就回归到了单模态情感分析，甚至不用再编写过多的特征融合，图像处理模块，只需要专注于提取文本特征就行了。

# 图像转文本（非必要）
执行qwen.py的代码就行了，需要自己在函数里指定原始的Twitter15/17的tsv文件路径，图像路径，输出路径。我跑了一遍代码已经转换好了，目前twitter15/17文件夹里面的new_*.tsv文件都是文本，实体，图像转为文本描述后的结果。只需要处理那个new_*的结果就行了，7B的int8量化需要16G显存，int4需要8G显存，无量化24G显存不够用，推测需要26G左右显存才可以运行，下载完qwen2以后要确认qwen.py的model变量里面的路径是否正确。

```
git clone https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8
```

# 模型准备
执行以下命令下载依赖库

```
pip install scikit-learn timm transformers pandas tiktoken protobuf SentencePiece
```

通过git命令下载预训练模型,修改args里--deberta_dir的路径为git clone后的预训练模型的文件夹路径

```
git clone https://huggingface.co/microsoft/deberta-v3-base
```

# 运行
通常来说直接运行main.py就可以了
