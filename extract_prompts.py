import os
import base64
import requests

API_URL = "http://127.0.0.1:8080/v1/chat/completions"

# 相对于脚本所在目录的路径，保证项目可移植
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(SCRIPT_DIR, "照片")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "提示词目录")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
PROMPT_TEMPLATE = "请只输出这张图片的内容描述，要求包含光影、风格、构图等细节，不要有其他解释性文字或格式标记。用中文回复。"


def extract_prompt(image_path: str) -> str:
    """通过 llama.cpp 多模态 API 提取图片提示词"""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            }
        ]
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [
        f for f in os.listdir(PHOTO_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not image_files:
        print("照片目录中没有找到支持的图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片，开始提取提示词...\n")

    for filename in image_files:
        image_path = os.path.join(PHOTO_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        output_txt = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

        print(f"处理: {filename}")
        try:
            prompt = extract_prompt(image_path)
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"  ✓ 已保存: {output_txt}")
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")

    print("\n完成！")


if __name__ == "__main__":
    main()
