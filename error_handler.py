"""
Unified error handling for Gradio application.
Provides friendly error messages and gr.Error() popup mechanism.
"""

import functools
import logging
from typing import Callable

import gradio as gr

logger = logging.getLogger(__name__)


class UserInputError(Exception):
    """Error caused by invalid user input."""
    pass


class ResourceError(Exception):
    """Error related to missing resources (model, voice library, etc.)."""
    pass


class SystemBusyError(Exception):
    """Engine is busy with another generation."""
    pass


class GenerationTimeoutError(Exception):
    """Generation timeout error."""
    pass


ERROR_MESSAGES = {
    "empty_text": "请输入要朗读的文本",
    "empty_instruct": "请描述您想要的音色特征",
    "empty_ref_audio": "请上传参考音频或从音色库中选择",
    "empty_ref_text": "参考音频的文本转录是必需的",
    "voice_not_found": "音色 '{name}' 不存在于音色库中",
    "model_not_found": "模型未找到，请检查是否已下载到本地缓存",
    "engine_busy": "系统正忙于处理其他请求，请稍后重试",
    "timeout": "生成超时，建议：缩短文本长度或在设置中增加超时时间",
    "yt_url_empty": "请输入 YouTube 视频链接",
    "yt_fetch_first": "请先获取视频信息（步骤1）",
    "yt_invalid_time": "时间格式无效，请使用 mm:ss 或秒数格式",
    "yt_end_before_start": "结束时间必须大于开始时间",
    "yt_clip_short": "片段过短（最少需要 3 秒）",
    "yt_clip_long": "片段超过 60 秒限制，已自动截取前 60 秒",
    "yt_time_exceeds": "结束时间超过视频长度",
    "yt_transcribe_empty": "转录结果为空，请尝试更清晰的音频片段",
    "script_empty": "请先输入剧本内容",
    "script_parse_first": "请先解析剧本",
    "script_error": "剧本解析错误：{detail}",
    "asr_empty": "请上传音频文件",
    "asr_result_empty": "转录结果为空，请尝试更清晰的音频",
    "save_empty": "没有可保存的内容",
    "import_empty_audio": "请先上传音频文件",
    "import_empty_name": "请输入音色名称",
    "import_empty_transcript": "请输入文本转录",
    "rename_empty": "请输入新的音色名称",
    "select_voice": "请先选择一个音色",
    "select_entry": "请先选择一条历史记录",
    "clone_missing_fields": "缺少必需字段：{fields}",
}


def get_friendly_message(error: Exception, context: str = "") -> str:
    """Convert exception to user-friendly Chinese message."""
    error_str = str(error)
    error_type = type(error).__name__
    
    if isinstance(error, UserInputError):
        return error_str
    
    if isinstance(error, ResourceError):
        return f"资源错误：{error_str}"
    
    if isinstance(error, GenerationTimeoutError):
        return ERROR_MESSAGES["timeout"]
    
    if isinstance(error, SystemBusyError):
        return ERROR_MESSAGES["engine_busy"]
    
    if "Engine is busy" in error_str or "previous generation" in error_str:
        return ERROR_MESSAGES["engine_busy"]
    
    if "timed out" in error_str.lower() or error_type == "GenerationTimeout":
        return ERROR_MESSAGES["timeout"]
    
    if error_type == "FileNotFoundError":
        if "voice" in error_str.lower():
            parts = error_str.split("'")
            name = parts[-2] if len(parts) >= 2 else ""
            return ERROR_MESSAGES["voice_not_found"].format(name=name)
        return ERROR_MESSAGES["model_not_found"]
    
    if error_type == "ValueError":
        return f"输入错误：{error_str}"
    
    if error_type == "RuntimeError":
        if "busy" in error_str.lower():
            return ERROR_MESSAGES["engine_busy"]
        return f"运行错误：{error_str}"
    
    logger.warning(f"Unhandled error type: {error_type}: {error}")
    return f"发生错误：{error_str}"


def raise_user_error(message_key: str, **kwargs) -> None:
    """Raise a UserInputError with a pre-defined message."""
    msg = ERROR_MESSAGES.get(message_key, message_key)
    if kwargs:
        msg = msg.format(**kwargs)
    raise UserInputError(msg)


def raise_resource_error(message_key: str, **kwargs) -> None:
    """Raise a ResourceError with a pre-defined message."""
    msg = ERROR_MESSAGES.get(message_key, message_key)
    if kwargs:
        msg = msg.format(**kwargs)
    raise ResourceError(msg)


def with_error_popup(func: Callable) -> Callable:
    """
    Decorator that wraps handler functions with gr.Error() popup mechanism.
    
    Usage:
        @with_error_popup
        def generate_custom_voice(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (UserInputError, ResourceError, GenerationTimeoutError, SystemBusyError) as e:
            raise gr.Error(str(e), duration=6)
        except Exception as e:
            if "GenerationTimeout" in type(e).__name__:
                raise gr.Error(ERROR_MESSAGES["timeout"], duration=8)
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            friendly_msg = get_friendly_message(e, func.__name__)
            raise gr.Error(friendly_msg, duration=8)
    
    return wrapper


def with_error_popup_yield(func: Callable) -> Callable:
    """
    Decorator for generator functions (using yield) with gr.Error() popup.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            gen = func(*args, **kwargs)
            yield from gen
        except (UserInputError, ResourceError, GenerationTimeoutError, SystemBusyError) as e:
            raise gr.Error(str(e), duration=6)
        except Exception as e:
            if "GenerationTimeout" in type(e).__name__:
                raise gr.Error(ERROR_MESSAGES["timeout"], duration=8)
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            friendly_msg = get_friendly_message(e, func.__name__)
            raise gr.Error(friendly_msg, duration=8)
    
    return wrapper