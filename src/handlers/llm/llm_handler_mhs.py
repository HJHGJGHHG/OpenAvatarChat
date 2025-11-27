# llm_handler_mhs.py
import os
import re
import base64
from typing import Dict, Optional, cast, Generato
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from loguru import logger
from pydantic import BaseModel, Field
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from dataclasses import dataclass
from typing import List as TypingList
@dataclass
class HistoryMessage:
    role: str
    content: str

class ChatHistory:
    def __init__(self, history_length=10):
        self.history_length = history_length
        self.messages: TypingList[HistoryMessage] = []
    
    def add_message(self, message: HistoryMessage):
        self.messages.append(message)
        if len(self.messages) > self.history_length:
            self.messages.pop(0)
    
    def get_formatted_history(self) -> str:
        return "\n".join([f"{msg.role}: {msg.content}" for msg in self.messages])
    
    def to_list(self):
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

# --- 配置项 ---
class MHSChatConfig(HandlerBaseConfigModel):
    model_name: str = Field(default="qwen-vl-max")
    system_prompt: str = Field(default="")
    api_key: str = Field(default="")
    api_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    enable_video_input: bool = Field(default=True)
    history_length: int = Field(default=10)
    mhs_enable: bool = Field(default=True)
    mhs_few_shot_k: int = Field(default=2)
    mhs_aux_model_name: str = Field(default="qwen-vl-plus")

class MHSChatContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.model_name: str = None
        self.system_prompt: Dict = None
        self.api_key: str = None
        self.api_url: str = None
        self.client: OpenAI = None
        self.input_texts: str = ""
        self.output_texts: str = ""
        self.current_image: Optional[np.ndarray] = None
        self.history: ChatHistory = None
        self.enable_video_input: bool = False
        self.mhs_enable: bool = True
        self.mhs_few_shot_k: int = 2
        self.mhs_aux_model_name: str = None

class MHSChatHandler(HandlerBase):
    """MHS处理器 - 心理健康支持处理器"""
    sentence_model: Optional[SentenceTransformer] = None
    corpus_embeddings: Optional[torch.Tensor] = None
    example_corpus: list = [
        "Seeker: 我最近总是感到很焦虑，晚上睡不着。 Counselor: 听到你这样说，我感到很难过。可以多告诉我一些关于焦虑的感觉吗？",
        "Seeker: 我觉得工作压力太大了，每天都喘不过气来。 Counselor: 听起来你承受了很多。这种压力具体体现在哪些方面呢？",
        "Seeker: 我和家人的关系最近很紧张。 Counselor: 家庭关系的确会深刻影响我们的情绪。你愿意和我聊聊具体发生了什么吗？",
    ]
    comet_model: Optional[AutoModelForSeq2SeqLM] = None
    comet_tokenizer: Optional[AutoTokenizer] = None

    def __init__(self):
        super().__init__()
        self.name = "MHSChatHandler"
        logger.info("MHS处理器初始化")

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=MHSChatConfig)

    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        entries = {"avatar_text": DataBundleEntry.create_text_entry("avatar_text")}
        definition = DataBundleDefinition(entries)
        
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(type=ChatDataType.HUMAN_TEXT),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(type=ChatDataType.CAMERA_VIDEO),
        }
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(type=ChatDataType.AVATAR_TEXT, definition=definition)
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        logger.info("加载MHS处理器模型...")
        if MHSChatHandler.sentence_model is None:
            try:
                model_path = os.path.join(engine_config.model_root, "paraphrase-multilingual-MiniLM-L12-v2")
                MHSChatHandler.sentence_model = SentenceTransformer(model_path)
                logger.info(f"加载sentence-transformer模型: {model_path}")
                MHSChatHandler.corpus_embeddings = MHSChatHandler.sentence_model.encode(
                    MHSChatHandler.example_corpus, convert_to_tensor=True)
            except Exception as e:
                logger.warning(f"无法加载sentence-transformer模型: {e}")
        if MHSChatHandler.comet_model is None:
            try:
                model_path = os.path.join(engine_config.model_root, "comet-atomic_2020_BART")
                MHSChatHandler.comet_tokenizer = AutoTokenizer.from_pretrained(model_path)
                MHSChatHandler.comet_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                if torch.cuda.is_available():
                    MHSChatHandler.comet_model.to('cuda')
                logger.info(f"加载COMET模型: {model_path}")
            except Exception as e:
                logger.warning(f"无法加载COMET模型: {e}")
        if isinstance(handler_config, MHSChatConfig):
            if not handler_config.api_key:
                raise ValueError('MHS处理器需要api_key配置')

    def create_context(self, session_context, handler_config=None) -> MHSChatContext:
        context = MHSChatContext(session_context.session_info.session_id)
        config = cast(MHSChatConfig, handler_config)
        context.model_name = config.model_name
        context.system_prompt = {'role': 'system', 'content': config.system_prompt}
        context.api_key = config.api_key
        context.api_url = config.api_url
        context.enable_video_input = config.enable_video_input
        context.history = ChatHistory(history_length=config.history_length)
        context.client = OpenAI(api_key=context.api_key, base_url=context.api_url)
        context.mhs_enable = config.mhs_enable
        context.mhs_few_shot_k = config.mhs_few_shot_k
        context.mhs_aux_model_name = config.mhs_aux_model_name
        return context

    def _pil_to_base64(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _preprocess_image_data(self, image_data):
        try:
            if image_data is None: return None
            if isinstance(image_data, np.ndarray):
                logger.info(f"原始图像形状: {image_data.shape}, 数据类型: {image_data.dtype}")
                if len(image_data.shape) == 4 and image_data.shape[0] == 1: image_data = image_data[0]
                if len(image_data.shape) == 3 and image_data.shape[0] in [1, 3]:
                    if image_data.shape[0] == 1: image_data = image_data[0]
                    else: image_data = np.transpose(image_data, (1, 2, 0))
                if image_data.dtype != np.uint8:
                    if image_data.max() <= 1.0: image_data = (image_data * 255).astype(np.uint8)
                    else: image_data = image_data.astype(np.uint8)
                if len(image_data.shape) == 2: pil_image = Image.fromarray(image_data, mode='L').convert('RGB')
                elif len(image_data.shape) == 3:
                    if image_data.shape[2] == 1: pil_image = Image.fromarray(image_data[:,:,0], mode='L').convert('RGB')
                    elif image_data.shape[2] == 3: pil_image = Image.fromarray(image_data, mode='RGB')
                    elif image_data.shape[2] == 4: pil_image = Image.fromarray(image_data, mode='RGBA').convert('RGB')
                    else:
                        logger.warning(f"不支持的通道数: {image_data.shape[2]}"); return None
                else:
                    logger.warning(f"不支持的图像维度: {len(image_data.shape)}"); return None
                pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                return pil_image
            else:
                logger.warning(f"不支持的图像数据类型: {type(image_data)}"); return None
        except Exception as e:
            logger.error(f"图像预处理错误: {e}"); return None

    def handle(self, context: HandlerContext, inputs: ChatData, 
               output_definitions: Dict[ChatDataType, HandlerDataInfo]) -> Generator[DataBundle, None, None]:
        context = cast(MHSChatContext, context)
        if inputs.type == ChatDataType.CAMERA_VIDEO and context.enable_video_input:
            context.current_image = inputs.data.get_main_data(); return
        if inputs.type == ChatDataType.HUMAN_TEXT:
            text = inputs.data.get_main_data()
            if text: context.input_texts += text
        speech_id = inputs.data.get_meta("speech_id", context.session_id)
        text_end = inputs.data.get_meta("human_text_end", False)
        if not text_end or not context.input_texts.strip(): return
        chat_text = context.input_texts.strip(); context.input_texts = ''
        dialogue_history = context.history.get_formatted_history()
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        try:
            if context.mhs_enable:
                similar_examples = self._find_similar_examples(dialogue_history, context.mhs_few_shot_k)
                visual_desc = self._get_visual_description(context.client, context.mhs_aux_model_name, context.current_image)
                context.current_image = None
                final_prompt = self._build_mhs_prompt(chat_text, dialogue_history, similar_examples, visual_desc)
                messages = [{"role": "user", "content": final_prompt}]
            else:
                context.history.add_message(HistoryMessage(role="user", content=chat_text))
                messages = [context.system_prompt] + context.history.to_list()
            
            completion = context.client.chat.completions.create(model=context.model_name, messages=messages, stream=True)
            context.output_texts = ''
            full_response = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices and chunk.choices[0].delta.content)
            context.output_texts = full_response
            output = DataBundle(output_definition)
            output.set_data("avatar_text", full_response) 
            output.add_meta("avatar_text_end", True)
            output.add_meta("speech_id", speech_id)
            yield output

            context.history.add_message(HistoryMessage(role="assistant", content=context.output_texts))
            logger.info(f"MHS回复并发送: {context.output_texts}")

        except Exception as e:
            logger.error(f"MHS处理错误: {e}")
            error_text = "抱歉，我遇到了一些问题，请稍后再试。"
            output = DataBundle(output_definition)
            output.set_data("avatar_text", error_text)
            output.add_meta("avatar_text_end", True)
            output.add_meta("speech_id", speech_id)
            yield output

    def _find_similar_examples(self, query_text: str, k: int) -> str:
        if not self.sentence_model or k == 0: return "No examples available."
        try:
            query_embedding = self.sentence_model.encode(query_text, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(k, len(self.example_corpus)))
            return "\n".join([f"Instance {i+1}: {self.example_corpus[idx]}" for i, idx in enumerate(top_results.indices)])
        except Exception as e:
            logger.error(f"查找相似示例错误: {e}"); return "Error finding examples."

    def _get_visual_description(self, client: OpenAI, model: str, image_frame: Optional[np.ndarray]) -> str:
        if image_frame is None: return "无视觉输入"
        try:
            pil_image = self._preprocess_image_data(image_frame)
            if pil_image is None: return "图像预处理失败"
            base64_image = self._pil_to_base64(pil_image)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "请详细描述这张图像中的场景、物体、人物、动作和情感。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}],
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"视觉描述错误: {e}"); return "视觉描述生成失败"

    def _build_mhs_prompt(self, chat_text: str, dialogue_history: str, similar_examples: str, visual_desc: str) -> str:
        return f"""**心理咨询任务定义:**
你是一位心理咨询师，为来访者提供心理和情感支持。通过积极倾听、共情和开放式提问，促进来访者对自身感受的探索。目标是促进自我理解和成长。
**相似对话示例:**
{similar_examples}
**表情描述:**
{visual_desc}
**对话上下文:**
{dialogue_history}
来访者: {chat_text}
咨询师:"""
    def destroy_context(self, context: HandlerContext): pass
    def on_before_register(self): pass
    def start_context(self, session_context, context): pass
    def destroy(self): pass

