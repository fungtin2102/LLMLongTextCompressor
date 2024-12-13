#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference to https://github.com/taylorbayouth/llm-text-compressor code

import logging

import tiktoken
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongTextCompressor:
    def __init__(self, base_url, api_key, model_type="gpt-3.5-turbo", max_tokens=4096):
        self.base_url = base_url
        self.api_key = api_key
        self.client  = OpenAI(
            api_key = self.api_key,
            base_url = self.base_url,
        )
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.tiktoken_model="gpt-3.5-turbo"
        self.encoding = tiktoken.encoding_for_model(self.tiktoken_model)

    def call_llm_model_api(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model = self.model_type,
                messages = messages,
                stream = True,
                stream_options={"include_usage": True}
            )
            response_text=[]
            for chunk in completion:
                # print(chunk.model_dump_json())
                chunk_text = chunk.choices[0].delta.content if len(chunk.choices)>0 else ""
                response_text.append(chunk_text) if chunk_text and len(chunk_text)>0 else None
            return "".join(response_text)
        except Exception as e:
            # logger.error(f"[Error] Failed to call LLM API: {e}")
            logger.error(f"[错误] 无法调用 LLM API：{e}")
            # return "An error occurred while processing the request."
            return "处理请求时发生错误。"
    
    def get_prompts(self):
        # Define a system message to establish context and ensure consistency
        system_message = (
            # "You are a compression assistant. Your task is to compress text to a specified word count or summary format. "
            "你是一名压缩助理，你的任务是将文本压缩为指定的字数或摘要格式。"
            # "Follow the specified compression style, using concise language while retaining essential details."
            "遵循指定的压缩风格，使用简洁的语言同时保留必要的细节。"
        )
        # Define the prompts
        prompts = {
            # 'auto_detect': "Review the following text and choose a mode of compression that best suits the type of content provided, aiming for around {target_word_count} words{json_spec}. Here is the text to analysize and compress:\n\n{chunk_string}",
            "auto_detect": "查看以下文本并选择最适合所提供内容类型的压缩模式，目标是大约 {target_word_count} 个单词{json_spec}。以下是要分析和压缩的文本：\n\n{chunk_string}",
            # 'bullet_points': "Summarize the following text into clear bullet points, with each point capturing an essential idea. Aim for approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
            "bullet_points": "将以下文本总结为清晰的要点，每个要点都抓住一个基本思想。目标是大约 {target_word_count} 个字{json_spec}:\n\n{chunk_string}",
            # 'glossary_terms': "Extract and define key terms and concepts from the following text, presenting them as a glossary list. Aim for around {target_word_count} words{json_spec}:\n\n{chunk_string}",
            "glossary_terms": "从以下文本中提取关键术语和概念，并以词汇列表的形式呈现。目标是大约 {target_word_count} 个字{json_spec}:\n\n{chunk_string}",
            # 'outline': "Create a structured outline with headings and subheadings, capturing the primary structure and flow of the text. Aim for around {target_word_count} words{json_spec}. Use hierarchical headings to emphasize key points and their relationships:\n\n{chunk_string}",
            "outline": "创建带有标题和副标题的结构化大纲，捕捉文本的主要结构和流程。目标是约 {target_word_count} 个单词{json_spec}。使用分层标题来强调要点及其关系：\n\n{chunk_string}",
            # 'critical_analysis': "Provide a brief analysis of the main points, discussing strengths, weaknesses, or important themes present in the text. Aim for around {target_word_count} words{json_spec}:\n\n{chunk_string}",
            "critical_analysis": "对主要观点进行简要分析，讨论文章中的优点、缺点或重要主题。目标是约 {target_word_count} 个单词{json_spec}:\n\n{chunk_string}",
            # 'facts_database': "Extract factual statements from the following text, summarizing key details, statistics, and verifiable information. Aim for around {target_word_count} words{json_spec}:\n\n{chunk_string}",
            "facts_database": "从以下文本中提取事实陈述，总结关键细节、统计数据和可验证信息。目标是约 {target_word_count} 个单词{json_spec}:\n\n{chunk_string}",
            # 'keywords_keyphrases': "List key terms and phrases that represent the main ideas of the following text. Limit the list to approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
            "keywords_keyphrases": "列出代表以下文本主要思想的关键术语和短语。将列表限制为大约 {target_word_count} 个单词{json_spec}:\n\n{chunk_string}"
        }
        for key in prompts:
            prompts[key] = prompts[key].replace("{json_spec}", "")
        return system_message, prompts
    
    def calculate_prompt_tokens(self, system_message, prompts, encoding):
        max_prompt_length = 0
        for prompt in prompts.values():
            full_prompt = system_message + prompt.format(target_word_count=0, chunk_string="")
            prompt_tokens = len(encoding.encode(full_prompt))
            if prompt_tokens > max_prompt_length:
                max_prompt_length = prompt_tokens
        #
        return max_prompt_length
    
    def split_text_into_chunks(self, text, max_chunk_size, encoding):
        # Split the text into chunks of at most max_chunk_size tokens
        # 将文本拆分成最多 max_chunk_size 个标记的块
        tokens = encoding.encode(text)
        chunks = []
        chunk_token_counts = []
        start = 0
        while start < len(tokens):
            end = min(start + max_chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            chunk_token_counts.append(len(chunk_tokens))
            start = end
        return chunks, chunk_token_counts
    
    def compute_target_word_count_per_chunk(self, chunk_tokens_len, total_tokens_len, token_target, aggression_factor):
        # Compute the proportion of the chunk relative to the entire text
        # 计算块相对于整个文本的比例
        chunk_proportion = chunk_tokens_len / total_tokens_len if total_tokens_len > 0 else 0

        # Compute the target total output tokens per chunk, applying the aggression factor
        # 应用 aggression 因子计算每个块的目标总输出 token 数
        target_tokens_per_chunk = (chunk_proportion * token_target) / aggression_factor if aggression_factor != 0 else 0

        # Convert tokens to words, assuming average tokens per word is 1.3
        # 将 token 转换为单词，假设每个单词的平均 token 为 1.3
        target_words_per_chunk = max(int(target_tokens_per_chunk / 1.3), 1)

        return target_words_per_chunk

    def compress_chunk_parallel(self, chunks, prompts, system_message):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.compress_chunk, chunk, prompts, system_message
                ) for chunk in chunks
            ]
            return [future.result() for future in concurrent.futures.as_completed(futures)]

    def compress_chunk(self, chunk, compressor_type, target_word_count, prompts, system_message):
        # Prepare the prompt
        # 准备提示
        prompt = prompts[compressor_type].format(
            target_word_count=target_word_count,
            chunk_string=chunk
        )
        messages=[
            {"role": "system","content": system_message},
            {"role": "user","content": prompt}
        ]
        compressed_chunk = self.call_llm_model_api(messages)
        return compressed_chunk
    
    def do_compress_large_text(self, large_text, token_target=200, compressor_type="auto_detect", model_max_tokens=32768):
        '''
        @param token_target The target length after compression is 200 words by default, that is, the content that exceeds the input size of the model will be compressed to this size.
            压缩后的目标长度默认为200个字，也就是说，超出模型输入大小的内容都会被压缩到这个大小。
        @param compressor_type The compression type is set to 'auto_detect' by default, which can be changed to other options such as 'bullet_points', 'glossary_terms', 'outline', 'critical_analysis', 'facts_database', 'keywords_keyphrases'.
            压缩类型默认设置为'auto_detect'，可以更改为其他选项，例如'bullet_points'，'glossary_terms'，'outline'，'critical_analysis'，'facts_database'，'keywords_keyphrases'。
        @param model_max_tokens The maximum number of tokens that the model can input is set to 4096 by default, which is used for testing. In reality, most are 32768, and qwen-plus is 128k.
            模型最大可以输入的token数，默认设置是4096，用于测试，现实中大部分都是32768，qwen-plus是128k。
        '''
        system_message, prompts = self.get_prompts()
        encoding = self.encoding
        tokens_reserved_for_prompt = self.calculate_prompt_tokens(system_message, prompts, encoding)
        max_chunk_size = model_max_tokens - tokens_reserved_for_prompt
        total_tokens = len(encoding.encode(large_text))
        compressed_text = large_text
        current_tokens = total_tokens
        iteration = 0
        max_iterations = 100
        initial_aggression_factor = 1.2
        aggression_factor = initial_aggression_factor
        while current_tokens > token_target and iteration < max_iterations:
            # logger.info(f"[X] Iteration {iteration + 1}: Current token count: {current_tokens}, target: {token_target}. Compressing...")
            logger.info(f"[X] 迭代 {iteration + 1}：当前token数：{current_tokens}，目标：{token_target}。正在压缩...")
            chunks, chunk_token_counts = self.split_text_into_chunks(compressed_text, max_chunk_size, encoding)
            total_chunk_tokens = sum(chunk_token_counts)
            compressed_chunks = []
            for idx, (chunk, chunk_tokens_len) in enumerate(zip(chunks, chunk_token_counts)):
                target_word_count = self.compute_target_word_count_per_chunk(chunk_tokens_len, total_chunk_tokens, token_target, aggression_factor)
                # logger.info(f"[X] Compressing chunk {idx + 1}/{len(chunks)} with target word count {target_word_count}...")
                logger.info(f"[X] 压缩块 {idx + 1}/{len(chunks)}，目标字数为 {target_word_count}...")
                compressed_chunk = self.compress_chunk(chunk, compressor_type, target_word_count, prompts, system_message)
                compressed_chunks.append(compressed_chunk)
            compressed_text = '\n'.join(compressed_chunks)
            current_tokens = len(encoding.encode(compressed_text))
            # Increase aggression_factor by 10% for the next iteration
            aggression_factor *= 1.1
            iteration += 1
        if iteration >= max_iterations and current_tokens > token_target:
            # logger.info("[X] Maximum iterations reached. Compression may not have reached the target token count.")
            logger.info("[X] 已达到最大迭代次数。压缩可能未达到目标标记数。")
        else:
            # logger.info("[X] Compression successful.")
            logger.info("[X] 压缩成功。")
        return compressed_text

    def do_chat_long_history(self, chat_history, new_message, token_target=200, compressor_type='auto_detect', model_max_tokens=4096):
        '''
        @param token_target The target length after compression is 200 words by default, that is, the content that exceeds the input size of the model will be compressed to this size.
            压缩后的目标长度默认为200个字，也就是说，超出模型输入大小的内容都会被压缩到这个大小。
        @param compressor_type The compression type is set to 'auto_detect' by default, which can be changed to other options such as 'bullet_points', 'glossary_terms', 'outline', 'critical_analysis', 'facts_database', 'keywords_keyphrases'.
            压缩类型默认设置为'auto_detect'，可以更改为其他选项，例如'bullet_points'，'glossary_terms'，'outline'，'critical_analysis'，'facts_database'，'keywords_keyphrases'。
        @param model_max_tokens The maximum number of tokens that the model can input is set to 4096 by default, which is used for testing. In reality, most are 32768, and qwen-plus is 128k.
            模型最大可以输入的token数，默认设置是4096，用于测试，现实中大部分都是32768，qwen-plus是128k。
        '''
        encoding = self.encoding
        new_message_tokens = len(encoding.encode(new_message))
        # logger.info(f"[X] new_message is {new_message_tokens} tokens")
        logger.info(f"[X] new_message 有 {new_message_tokens} 个token")
        if new_message_tokens>model_max_tokens:
            # logger.info(f"[X] new_message is {new_message_tokens} tokens, the maximum allowed length for a llm model is {model_max_tokens}, please simplify the message content.")
            logger.info(f"[X] new_message 为 {new_message_tokens} 个 token，大模型允许的最大长度为 {model_max_tokens}，请简化消息内容。")
            return "Sorry, the new message is too long."
        # The number of tokens available under the current model
        # 当前模式下可用的token数量
        available_tokens = model_max_tokens-new_message_tokens
        # logger.info(f"[X] The maximum input length of the model is {model_max_tokens}, after deducting {new_message_tokens} tokens of new_message, there are {available_tokens} available tokens")
        logger.info(f"[X] 模型最大输入长度为 {model_max_tokens}，扣除 new_message 的 {new_message_tokens} 个 token 后，还有 {available_tokens} 个可用 token")
        # The number of tokens that do not need to be compressed, that is, the message list is looped in reverse order, and the part longer than this value needs to be compressed
        # 不需要压缩的token数量，即消息列表倒序循环，长度超过此值的部分需要进行压缩
        no_compression_required_tokens = (available_tokens - token_target)
        # logger.info(f"[X] After deducting the size {token_target} characters reserved for compressing a piece of content, there are still {no_compression_required_tokens} tokens in {available_tokens}, which is the number of characters that do not need to be compressed")
        logger.info(f"[X] 扣除为压缩一段内容预留的大小{token_target}个字符后，{available_tokens}中仍有{no_compression_required_tokens}个token，即不需要压缩的字符数")
        # logger.info(f"[X] Start calculating...Calculate the number of words and messages that need to be compressed by excluding the number of words that do not need to be compressed, totaling {len(chat_history)} messages")
        logger.info(f"[X] 开始计算...计算需要压缩的单词数和消息数，排除不需要压缩的单词数，共计{len(chat_history)}条消息")
        compression_required_message_index = len(chat_history)
        preseved_messages_token_counter = 0
        preseved_messages_token_counter += new_message_tokens
        compressed_history=[]
        if len(chat_history)>0:
            for idx in range(len(chat_history)-1,0,-1):
                message = chat_history[idx]
                # The number of tokens in the latest chat message
                # 最新聊天消息中的token数量
                message_tokens = len(encoding.encode(message["content"]))
                preseved_messages_token_counter+=message_tokens
                if preseved_messages_token_counter>no_compression_required_tokens:
                    # logger.info(f"[X] When calculating the {idx}th historical message, the accumulated {preseved_messages_token_counter} tokens exceeded the {no_compression_required_tokens} tokens that do not need to be compressed. All the tokens before this one (including) need to be compressed")
                    logger.info(f"[X] 计算第 {idx} 条历史消息时，累计的 {preseved_messages_token_counter} 个 token 超出了 {no_compression_required_tokens} 个不需要压缩的 token，此 token 之前的所有 token（包括）都需要压缩")
                    # All messages before this one need to be compressed, including this one
                    # 此消息之前的所有消息都需要压缩，包括此消息
                    compression_required_message_index=idx
                    break
                compressed_history.insert(0,message)
                # logger.info(f"[X] Calculate the {idx}th historical message, accumulate to {preseved_messages_token_counter} words, no compression required, keep directly, the current compressed_history length is {len(compressed_history)}")
                logger.info(f"[X] 计算第{idx}条历史消息，累计至{preseved_messages_token_counter}个字，不需要压缩，直接保留，当前compressed_history长度为{len(compressed_history)}")
        if compression_required_message_index>0:
            # logger.info(f"[X] The number of messages that need to be compressed, from 0 to {compression_required_message_index}")
            logger.info(f"[X] 需要压缩的消息数量，从 0 到 {compression_required_message_index}")
            large_text = "\n".join([f"{chat_history[idx]['role']}: {chat_history[idx]['content']}" for idx in range(0,compression_required_message_index)])
            summary = self.do_compress_large_text(large_text, token_target, compressor_type, model_max_tokens)
            # logger.info(f"[X] The number of messages that need to be compressed is from 0 to {compression_required_message_index}, the summary is: \n[\n{summary}\n], which will be placed at the front of the history that will be kept without compression")
            logger.info(f"[X] 需要压缩的消息数量从 0 到 {compression_required_message_index}，摘要为：\n===\n{summary}\n===，将放置在不压缩保留的历史记录的最前面做为前情提要")
            # compressed_messages = f"The following is the conversation history:\n{summary}\nAssistant, please continue to reply to the user:"
            compressed_messages = f"以下是对话历史记录:\and{summary}\助手，请继续回复用户："
            compressed_history.insert(0,{"role": "assistant", "content": compressed_messages})
        # logger.info(f"[X] The current compressed_history length is {len(compressed_history)}")
        logger.info(f"[X] 当前 compressed_history 长度为 {len(compressed_history)}")
        compressed_history.append({"role": "user", "content": new_message})
        # logger.info(f"[X] The current compressed_history after adding new_message is {len(compressed_history)}")
        logger.info(f"[X] 添加 new_message 之后的当前 compressed_history 为 {len(compressed_history)}")
        response_text = self.call_llm_model_api(compressed_history)
        # logger.info(f"[X] Current inference result:\n====\n{response_text}\n====\n")
        logger.info(f"[X] 当前推理结果:\n====\n{response_text}\n====\n")
        return response_text

def test_compress_large_text(compressor):
    # 示例对话历史
    chat_history = [
        {"role": "user", "content": "你好，能帮我查一下天气吗？"},
        {"role": "assistant", "content": "当然，请问您需要查询哪个城市的天气？"},
        {"role": "user", "content": "上海的天气。"}
    ]
    
    # 生成的一组超长的项目管理相关高仿真聊天记录
    for i in range(50):
        if i % 2 == 0:
            chat_history.append({"role": "user", "content": f"项目组成员{i}的任务进展如何？是否有遇到困难？"})
        else:
            chat_history.append({"role": "assistant", "content": f"项目成员{i}反馈进展顺利，但需要额外的资源支持来加快开发速度。"})
        chat_history.append({"role": "user", "content": f"关于项目计划第{i}阶段的时间安排，需要调整吗？"})
        chat_history.append({"role": "assistant", "content": f"第{i}阶段的时间安排可以维持，但需要密切跟踪以确保不拖延。"})
    large_text = "\n".join([f"{item['role']}: {item['content']}" for item in chat_history])

    compressed_text = compressor.do_compress_large_text(large_text, model_max_tokens=1024)

    print("Original Text:")
    print(large_text)
    print("\nCompressed Text:")
    print(compressed_text)

def test_chat_long_history(compressor):
    # 示例对话历史
    chat_history = [
        {"role": "user", "content": "你好，能帮我查一下天气吗？"},
        {"role": "assistant", "content": "当然，请问您需要查询哪个城市的天气？"},
        {"role": "user", "content": "深圳的天气。"}
    ]
    
    # 生成的一组超长的项目管理相关高仿真聊天记录
    for i in range(50):
        if i % 2 == 0:
            chat_history.append({"role": "user", "content": f"项目组成员{i}的任务进展如何？是否有遇到困难？"})
        else:
            chat_history.append({"role": "assistant", "content": f"项目成员{i}反馈进展顺利，但需要额外的资源支持来加快开发速度。"})
        chat_history.append({"role": "user", "content": f"关于项目计划第{i}阶段的时间安排，需要调整吗？"})
        chat_history.append({"role": "assistant", "content": f"第{i}阶段的时间安排可以维持，但需要密切跟踪以确保不拖延。"})
    # new_message = "请问，项目组需要哪些资源来完成这个任务？"
    new_message = "所以，深圳的天气到底怎么样了？"
    response_text = compressor.do_chat_long_history(chat_history, new_message)
    print(response_text)
    pass
if __name__ == "__main__":
    # Example test for the LongTextCompressor class
    base_url = "https://api.openai.com/v1/chat/completions"
    api_key="sk-***********"
    compressor = LongTextCompressor(base_url, api_key)
    test_compress_large_text(compressor)
    test_chat_long_history(compressor)
