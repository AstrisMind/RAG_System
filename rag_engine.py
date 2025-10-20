from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from utils import logger
import os
import httpx
from openai import OpenAI

# 定义Qwen LLM类
class QwenLLM(LLM):
    """Qwen API LLM实现"""
    
    client: Any = None
    
    def __init__(self):
        """初始化Qwen LLM"""
        super().__init__()
        api_key = os.getenv("DASHSCOPE_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
    
    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model="qwen3-max",
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 1500),
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"调用Qwen API时出错: {str(e)}")
            raise

class RAGEngine:
    """RAG引擎:向量检索和回答生成"""
    def __init__(self, text_processor=None):
        try:
            self.llm = QwenLLM()
            logger.info("成功初始化Qwen API")
        except Exception as e:
            logger.error(f"初始化Qwen API失败: {str(e)}")
            raise
        self.text_processor = text_processor
        self.conversation_history: List[Dict[str, str]] = []
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            你是一个专业的助手，基于提供的参考资料回答用户问题。
            参考资料：
            {context}           
            对话历史：
            {history} 
            用户当前问题：
            {query}
            请基于参考资料和对话历史，以自然、友好的语言回答用户问题。
            若回答中包含引用来源，请必须包含引用来源，格式为"（来源：文件名）",
            若回答中不包含引用来源，请不要在回答中添加任何来源信息。
            如果参考资料中没有相关信息，请坦诚告知用户。
            """
        )
        logger.info("RAG引擎初始化完成")
    
    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        格式化检索结果作为上下文
        
        Args:
            search_results: 检索结果列表
            
        Returns:
            格式化的上下文文本
        """
        context_parts = []
        for i, result in enumerate(search_results):
            metadata = result.get("metadata", {})
            source_info = f"来源: {metadata.get('file_name', 'unknown')}"
            context_parts.append(f"[{i+1}]")
            context_parts.append(result["text"])
            context_parts.append(f"({source_info})")
            context_parts.append("---")
        
        return "\n".join(context_parts)

    
    def format_history(self) -> str:
        """
        格式化对话历史
        
        Returns:
            格式化的对话历史文本
        """
        if not self.conversation_history:
            return "暂无"
        
        history_parts = []
        for i, turn in enumerate(self.conversation_history):
            if turn["role"] == "user":
                history_parts.append(f"用户 {i//2 + 1}: {turn['content']}")
            elif turn["role"] == "assistant":
                history_parts.append(f"助手 {i//2 + 1}: {turn['content']}")
        
        return "\n".join(history_parts)
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        生成回答
        Args:
            query: 用户查询
            search_results: 检索结果   
        Returns:
            生成的回答
        """
        # 格式化上下文和历史
        context = self.format_context(search_results)
        history = self.format_history()
        
        logger.info(f"开始生成回答，查询: {query[:50]}..., 上下文长度: {len(context)}")
        
        try:
            chain = (
                {
                    "context": RunnablePassthrough(),
                    "history": RunnablePassthrough(),
                    "query": RunnablePassthrough()
                }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            answer = chain.invoke({
                "context": context,
                "history": history,
                "query": query
            })
            return answer
        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            raise
    
    def rag_pipeline(self, query: str, k: int = 3) -> Dict[str, Any]:
        # if not self.text_processor or not self.text_processor.vector_store:
        #     raise ValueError("向量数据库未初始化，请先处理文档")       
        # logger.info(f"执行RAG流程，查询: {query}")
        search_results = self.text_processor.search_similar(query, k=k)
        answer = self.generate_answer(query, search_results)
        self.add_to_history("user", query)
        self.add_to_history("assistant", answer)
        result = {"query": query,"search_results": search_results,"answer": answer,"history": self.conversation_history.copy()}  
        logger.info("RAG流程执行完成")
        return result
    
    def add_to_history(self, role: str, content: str) -> None:
        """
        添加对话历史
        Args:
            role: 角色（user或assistant）
            content: 内容
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # 限制历史长度
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info("对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

# 创建全局RAG引擎实例
rag_engine = None

__all__ = ["RAGEngine", "rag_engine"]

# # 修正后的提示模板
prompt_template = """
基于以下上下文：
相关段落内容：
{context}

请回答：{query}
"""

# 更新后的上下文格式化方法
def format_context(documents):
    return "\n".join([f"内容来源：{doc.metadata['source']}\n{doc.page_content}" for doc in documents])