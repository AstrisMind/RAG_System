from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from utils import logger
import os
import time
from langsmith.client import Client

class AgentState(Dict[str, Any]):
    query: str
    search_results: List[Dict[str, Any]]
    answer: str
    history: List[Dict[str, str]]
    intermediate_steps: List[Dict[str, Any]]
    start_time: float

class RAGAgent:
    def __init__(self, document_processor, text_processor, rag_engine):
        self.document_processor = document_processor
        self.text_processor = text_processor
        self.rag_engine = rag_engine
        self.rag_engine.text_processor = text_processor
        
        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
            self.langsmith_client = Client(
                api_key=os.getenv("LANGSMITH_API_KEY")
            )
            logger.info("LangSmith客户端初始化成功")
        else:
            self.langsmith_client = None
            logger.info("LangSmith未启用，跳过初始化")

        # api_key = os.getenv("LANGSMITH_API_KEY")
        # if not api_key:
        #     raise ValueError("请设置LANGSMITH_API_KEY环境变量")
        
        # self.langsmith_client = Client(
        #     api_key=api_key
        # )
        # logger.info("LangSmith客户端初始化成功")
        
    
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=MemorySaver())
        
        logger.info("RAG Agent初始化完成")
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve_documents)
        graph.add_node("generate", self._generate_answer)
        graph.add_node("log_result", self._log_result)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "log_result")
        graph.add_edge("log_result", END)
        
        logger.info("LangGraph执行流程构建完成")
        return graph
    
    def _retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        logger.info(f"[节点: retrieve] 开始检索文档，查询: {query[:50]}...")
        search_results = self.text_processor.search_similar(query, k=5)
        intermediate_steps = state.get("intermediate_steps", [])
        intermediate_steps.append({
            "node": "retrieve",
            "results": search_results,
            "timestamp": time.time()
        })
        return {
            **state,
            "search_results": search_results,
            "intermediate_steps": intermediate_steps
        }
    
    def _generate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        search_results = state.get("search_results", [])
        logger.info(f"[节点: generate] 开始生成回答")
        answer = self.rag_engine.generate_answer(query, search_results)
        self.rag_engine.add_to_history("user", query)
        self.rag_engine.add_to_history("assistant", answer)
        intermediate_steps = state.get("intermediate_steps", [])
        intermediate_steps.append({
            "node": "generate",
            "answer": answer,
            "timestamp": time.time()
        })
        return {
            **state,
            "answer": answer,
            "history": self.rag_engine.get_history(),
            "intermediate_steps": intermediate_steps
        }
    
    def _log_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        search_results = state.get("search_results", [])
        start_time = state.get("start_time", time.time())
        response_time = time.time() - start_time
        logger.info(f"[节点: log_result] 完成处理，响应时间: {response_time:.2f}秒")
    
        if self.langsmith_client:
            try:
                similarity_scores = [r.get("score", 0) for r in search_results]
                avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                
                run = self.langsmith_client.create_run(
                    name="rag_query",
                    run_type="chain",
                    inputs={
                        "query": query,
                        "search_results_count": len(search_results)
                    },
                    outputs={
                        "answer": answer,
                        "response_time": response_time,
                        "avg_similarity_score": avg_similarity
                    },
                    tags=["rag", "demo"]
                )
                logger.info(f"记录到LangSmith，run_id: {run.id}")
            except Exception as e:
                logger.warning(f"记录到LangSmith失败: {str(e)}")

        return {
            **state,
            "metrics": {
                "response_time": response_time,
                "similarity_scores": [r.get("score", 0) for r in search_results],
                "intermediate_steps_count": len(state.get("intermediate_steps", []))
            }
        }
    
    def run(self, query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Agent开始处理查询: {query[:50]}...")
        initial_state = {
            "query": query,
            "search_results": [],
            "answer": "",
            "history": self.rag_engine.get_history(),
            "intermediate_steps": [],
            "start_time": time.time()
        }
        result = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "rag_thread_01"}}
        )
        logger.info("Agent处理完成")
        return result
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        document = self.document_processor.process_file(file_path)
        chunks = self.text_processor.process_document(document)
        return {
            "document": document,
            "chunks_count": len(chunks)
        }

__all__ = ["RAGAgent", "AgentState"]