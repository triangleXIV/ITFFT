import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os

# 加载模型和处理器（全局加载，避免重复加载）
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("./Qwen2-VL-7B-Instruct")


def process_tsv(input_file, output_file, img_base_path):
    """
    处理TSV文件，将第三列的图片路径传入Qwen2-VL模型生成图像描述，并替换第三列内容。

    参数:
        input_file (str): 输入的TSV文件路径
        output_file (str): 输出的TSV文件路径
    """

    def generate_image_description(image_path):
        """
        使用Qwen2-VL模型生成图像描述。

        参数:
            image_path (str): 图像路径

        返回:
            str: 生成的图像描述
        """
        try:
            # 构造消息结构
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "Identify the key elements in the image and briefly describe each element's related emotions, atmosphere, or actions in up to 60 words."},
                    ],
                }
            ]
            # 处理输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # 模型推理
            generated_ids = model.generate(**inputs, max_new_tokens=192)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]  # 返回生成的描述
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return "Error generating description"

    # 读取TSV文件
    data = pd.read_csv(input_file, sep="\t")

    # 替换第三列内容
    new_descriptions = []
    for index, row in data.iterrows():
        img_path = os.path.join(img_base_path, row[2])
        description = generate_image_description(img_path)
        new_descriptions.append(description)
        print(f"Processed row {index + 1}/{len(data)}: {description}")
        torch.cuda.empty_cache()

    # 更新第三列
    data.iloc[:, 2] = new_descriptions

    # 保存新的TSV文件
    data.to_csv(output_file, sep="\t", index=False)
    print(f"Updated TSV file saved to {output_file}")

# 三个参数分别为 原始的twitter15/17的具体tsv文件路径，处理后的图像转为文本描述的文件名，图像的路径 按实际情况调整
process_tsv("dev.tsv", "new_dev.tsv", "./twitterdataset/img_data/twitter2017_images/")



