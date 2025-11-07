from llama_index.embeddings.dashscope import DashScopeEmbedding
import logging
from typing import List, Tuple

import numpy as np
from dashscope import get_tokenizer
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from pydantic import Field
logger = logging.getLogger(__name__)

# 此处填写千问文本嵌入模型: https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2842587
QWEN_GPT_API_KEY = ""

def _combine_chunk_embeddings(num_texts: int, chunk_embeddings: List[List[float]],
                              indices: List[int]) -> List[List[float]]:
    """合并分块嵌入结果（加权平均）"""
    # 简单实现：对属于同一原始文本的所有分块嵌入进行平均
    # 更复杂的实现可以考虑根据分块长度进行加权
    final_embeddings = []

    for i in range(num_texts):
        # 找到属于第i个文本的所有块
        chunk_indices = [idx for idx, orig_idx in enumerate(indices) if orig_idx == i]
        if not chunk_indices:
            # 处理空文本情况
            final_embeddings.append([0.0] * len(chunk_embeddings[0]) if chunk_embeddings else [])
            continue

        # 对同一文本的所有分块嵌入进行平均
        combined_embedding = np.mean([chunk_embeddings[idx] for idx in chunk_indices], axis=0)
        final_embeddings.append(combined_embedding.tolist())

    return final_embeddings


class DashScopeEmbeddingsExt(DashScopeEmbeddings):
    """扩展的DashScope嵌入模型，支持长度安全的分块处理"""

    embedding_ctx_length: int = Field(default=8100, description="嵌入最大token长度")
    check_embedding_ctx_length: bool = Field(default=True, description="是否检查嵌入上下文长度")
    tokenizer_model: str = Field(default='qwen-turbo', description="分词器模型")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表，支持长文本自动分块"""
        if not self.check_embedding_ctx_length:
            # 不检查长度，使用原始逻辑
            return super().embed_documents(texts)

        # 使用长度安全的嵌入函数
        return self._get_len_safe_embeddings(texts)

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        DashScope版本的长度安全嵌入实现
        参考OpenAI的实现逻辑，但适配DashScope API
        """
        # 1. 分词和分块
        batch_size = 16  # 可根据API限制调整
        iter_range, token_chunks, indices = self._tokenize(texts, batch_size)

        # 2. 批量获取嵌入
        batched_embeddings = []
        for i in range(0, len(token_chunks), batch_size):
            batch_chunks = token_chunks[i:i + batch_size]
            try:
                # 调用父类的embed_documents方法处理每个分块
                batch_embeddings = super().embed_documents(batch_chunks)
                batched_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"处理分块时出错: {e}")
                # 错误处理：返回空嵌入或抛出异常
                raise

        # 3. 合并分块嵌入结果
        return _combine_chunk_embeddings(len(texts), batched_embeddings, indices)

    def _tokenize(self, texts: List[str], batch_size: int) -> Tuple[range, List[str], List[int]]:
        """使用DashScope tokenizer进行分词和分块"""
        tokens = []
        indices = []

        try:
            tokenizer = get_tokenizer(self.tokenizer_model)
            for i, text in enumerate(texts):
                # 使用DashScope tokenizer对文本进行标记化
                tokenized = tokenizer.encode(text)

                # 将tokens拆分为遵循embedding_ctx_length的块
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk = tokenized[j: j + self.embedding_ctx_length]
                    # 将token ID转换回字符串
                    chunk_text = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)

        except Exception as e:
            # 如果tokenization失败，回退到字符级分块
            logger.warning(f"Tokenization失败，使用基于字符的分块: {e}")
            for i, text in enumerate(texts):
                for j in range(0, len(text), self.embedding_ctx_length * 4):  # 粗略估计
                    chunk_text = text[j: j + self.embedding_ctx_length * 4]
                    tokens.append(chunk_text)
                    indices.append(i)

        return range(0, len(tokens), batch_size), tokens, indices

# 为适应长上下文情况的扩展版本 (自行实现的逻辑)
def get_embedder_ext():
    return DashScopeEmbeddingsExt(
        dashscope_api_key=QWEN_GPT_API_KEY,
        model="text-embedding-v4",
        check_embedding_ctx_length=True,
        embedding_ctx_length=8100,  # 保守设置，确保稳定性
    )

# 原生版本
def get_embedder():
    return DashScopeEmbedding(
        model_name="text-embedding-v4",
        api_key=QWEN_GPT_API_KEY
    )