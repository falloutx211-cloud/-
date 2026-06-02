import os
import base64
import mimetypes
import requests
import logging
import time
from typing import Optional

# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# 配置区域（可按需修改）
# =========================
API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "local-vision-model"  # llama.cpp / LM Studio 需要

# 相对于脚本所在目录的路径（保证可移植）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(SCRIPT_DIR, "照片")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "提示词目录")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 2  # 重试延迟（秒）

PROMPT_TEMPLATE = (
    "请只输出这张图片的内容描述，要求包含光影、风格、构图等细节，"
    "不要有其他解释性文字或格式标记。用中文回复。"
)


# =========================
# 工具函数
# =========================
def check_api_connectivity() -> bool:
    """
    检查本地 llama.cpp 服务是否运行
    """
    try:
        logger.info(f"检查 API 连接: {API_URL}")
        response = requests.get(
            API_URL.replace("/v1/chat/completions", "/health"),
            timeout=5
        )
        if response.status_code == 200:
            logger.info("✓ API 连接成功")
            return True
    except requests.exceptions.ConnectionError:
        logger.error(f"✗ 无法连接到 API: {API_URL}")
        logger.error("请确保 llama.cpp / LM Studio 已启动")
        return False
    except Exception as e:
        logger.warning(f"API 健康检查失败: {e}，继续尝试...")
        return True  # 继续尝试，因为有些服务器可能不支持 /health


def ensure_directories():
    """
    创建必要的目录
    """
    try:
        os.makedirs(PHOTO_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"✓ 目录已就绪: {PHOTO_DIR}, {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"✗ 无法创建目录: {e}")
        raise


# =========================
# 核心函数
# =========================
def extract_prompt(image_path: str) -> Optional[str]:
    """
    通过本地 llama.cpp / LM Studio Vision API 提取图片提示词
    支持重试机制
    """
    # 文件大小检查
    if os.path.getsize(image_path) > MAX_FILE_SIZE:
        raise ValueError("图片文件过大（超过 10MB）")

    # 读取并编码图片
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 自动识别 MIME 类型
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    }

    # 带重试的 API 调用
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(f"API 调用 (尝试 {attempt}/{MAX_RETRIES})")
            response = requests.post(API_URL, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            logger.warning(f"✗ 请求超时 (尝试 {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"✗ 连接失败 (尝试 {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise
        except Exception as e:
            logger.error(f"✗ API 错误: {e}")
            raise


def main():
    logger.info("=" * 50)
    logger.info("图片提示词提取工具")
    logger.info("=" * 50)

    # 创建必要目录
    try:
        ensure_directories()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    # 检查 API 连接
    if not check_api_connectivity():
        logger.error("无法继续，请启动 llama.cpp 服务")
        return

    # 获取图片列表
    try:
        image_files = [
            f for f in os.listdir(PHOTO_DIR)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
    except Exception as e:
        logger.error(f"✗ 读取照片目录失败: {e}")
        return

    if not image_files:
        logger.info("照片目录中没有找到支持的图片文件。")
        logger.info(f"请将图片放到: {PHOTO_DIR}")
        return

    logger.info(f"找到 {len(image_files)} 张图片，开始提取提示词...\n")

    success_count = 0
    failed_count = 0

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(PHOTO_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        output_txt = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

        logger.info(f"[{idx}/{len(image_files)}] 处理: {filename}")
        try:
            prompt = extract_prompt(image_path)
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(prompt)
            logger.info(f"  ✓ 已保存: {output_txt}")
            success_count += 1
        except ValueError as e:
            logger.error(f"  ✗ 文件错误: {e}")
            failed_count += 1
        except Exception as e:
            logger.error(f"  ✗ 处理失败: {e}")
            failed_count += 1

    logger.info("\n" + "=" * 50)
    logger.info(f"完成! 成功: {success_count}, 失败: {failed_count}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
