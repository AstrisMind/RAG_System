import os
import json
from dotenv import load_dotenv
from utils import logger, configure_standard_logging, ensure_directory
from document_processor import document_processor
from text_processor import text_processor
from rag_engine import RAGEngine
from agent_flow import RAGAgent
import sys

def setup_environment():
    if os.path.exists(".env"):
        load_dotenv()
        logger.info("环境变量加载成功")
    else:
        logger.warning(".env文件不存在，请根据.env.example创建")
    
    configure_standard_logging()
    
    ensure_directory("data")
    ensure_directory("vector_db")
    ensure_directory("logs")

def create_rag_system():
    rag_engine = RAGEngine(text_processor)
    agent = RAGAgent(
        document_processor=document_processor,
        text_processor=text_processor,
        rag_engine=rag_engine
    )
    return agent

def main():
    setup_environment()
    logger.info("=== RAG系统启动 ===")
    agent = create_rag_system()
    return agent

if __name__ == "__main__":
    agent = main()
    
    # if len(sys.argv) >= 2:
    #     # 处理所有文件参数
    #     file_paths = sys.argv[1:-1] if len(sys.argv)>2 else sys.argv[1:]
        
    #     # 处理文档
    #     for file_path in file_paths:
    #         if os.path.exists(file_path):
    #             agent.process_document(file_path)
    #             logger.success(f"成功处理文件: {file_path}")
    #         else:
    #             logger.error(f"文件不存在: {file_path}")
    #             sys.exit(1)
        
    #     # 执行问答
    #     if len(sys.argv) >=2:
    #         query = sys.argv[-1] if len(sys.argv)>1 else ""
    #         if query:
    #             result = agent.run(query)
    #             print("\n=== 问答结果 ===")
    #             print(result["answer"])
    #             # print("\n=== 参考来源 ===")
    #             # for doc in result["documents"]:
    #             #     print(f"- {doc.metadata['source']} 第{doc.metadata['page']}页")
    
    # sys.exit(0)

__all__ = ["setup_environment", "create_rag_system", "main"]