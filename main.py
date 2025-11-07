# 使用案例
from dashscope import get_tokenizer

from extClassStruct import get_embedder_ext, get_embedder, QWEN_GPT_API_KEY

TEXT_EMBEDDING_MODEL_MAX_CONTENT = 8100  # 文本嵌入模型所能支持的最大上下文(理论值为8192)
long_text = "这是一个很长的测试文本..." * 2000


def main():
    if len(QWEN_GPT_API_KEY) == 0:
        print("[-] 请先填写 QWEN_GPT_API_KEY !!! 否则无法使用文本嵌入模型(或者你自己本地部署文本嵌入模型)")
        return

    tokenizer = get_tokenizer('qwen-turbo')
    tokenized = tokenizer.encode(long_text)
    token_len = len(tokenized)
    print(f"long_text 文本的token长度为: {token_len}")

    if token_len < TEXT_EMBEDDING_MODEL_MAX_CONTENT:  # 不超过限制，则直接使用原生版本
        embedder = get_embedder()
        content_list = [long_text]
        vectors = embedder.get_text_embedding_batch(content_list)
        print("[+] 文本没有超过最大上下文,使用原生版本获取其向量数据:\n")
        print(vectors)

    # 文本超过最大上下文,则使用自行扩展的版本
    embedder_ext = get_embedder_ext()
    result_ext_vector = embedder_ext.embed_documents([long_text])
    print("[+] 文本超过最大上下文,使用扩展版本获取其向量数据:\n")
    print(result_ext_vector)


if __name__ == "__main__":
    main()
