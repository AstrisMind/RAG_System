from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from utils import logger
import numpy as np
import os
import httpx
import asyncio
import aiohttp
from sentence_transformers import SentenceTransformer

class BgeEmbeddings(Embeddings):
    def __init__(self, model_path: str = r"models\BAAI\bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_path)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding
        # except Exception as e:
        #     logger.error(f"获取嵌入向量时出错: {str(e)}")
        #     raise

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        self.embeddings = BgeEmbeddings()
        self.vector_store: Optional[FAISS] = None
        logger.info(f"文本处理器初始化完成，chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")
    
    def split_text(self, text: str, document_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        logger.info(f"开始分割文本，原文本长度: {len(text)}")
        
        chunks = self.text_splitter.split_text(text)
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_info = {
                "text": chunk,
                "chunk_id": i,
                "chunk_length": len(chunk)
            }
            
            if document_info:
                chunk_info.update({
                    "file_name": document_info.get("file_name"),
                    "file_path": document_info.get("file_path"),
                    "file_type": document_info.get("file_type")
                })
            
            chunks_with_metadata.append(chunk_info)
        logger.info(f"文本分割完成，生成 {len(chunks_with_metadata)} 个段落")
        return chunks_with_metadata
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> FAISS:
        logger.info(f"开始创建向量数据库，段落数量: {len(chunks)}") 
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{
            k: v for k, v in chunk.items() if k != "text"
        } for chunk in chunks]
        try:
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            logger.info("向量数据库创建成功")
            return self.vector_store
        except Exception as e:
            logger.error(f"创建向量数据库时出错: {str(e)}")
            raise
    
    def search_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.vector_store:
            raise ValueError("向量数据库未初始化")  
        logger.info(f"开始向量搜索，查询: {query[:50]}..., k={k}")
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        formatted_results = []
        for doc, score in results:
            result = {
                "text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            }
            formatted_results.append(result)
        logger.info(f"向量搜索完成，找到 {len(formatted_results)} 个相似结果")
        return formatted_results
    
    def save_vector_store(self, file_path: str) -> None:
        if not self.vector_store:
            raise ValueError("向量数据库未初始化")
        logger.info(f"保存向量数据库到: {file_path}")
        self.vector_store.save_local(file_path)

    def load_vector_store(self, file_path: str) -> FAISS:
        logger.info(f"从 {file_path} 加载向量数据库")
        self.vector_store = FAISS.load_local(
            file_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("向量数据库加载成功")
        return self.vector_store
    
    def process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = self.split_text(document["content"], document)
        if self.vector_store:
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [{
                k: v for k, v in chunk.items() if k != "text"
            } for chunk in chunks]
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        else:
            self.create_vector_store(chunks)
        return chunks

text_processor = TextProcessor()

__all__ = ["TextProcessor", "text_processor"]