import os
import io
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document
import mistune
from utils import logger, validate_file, get_file_extension

class DocumentProcessor:
    def __init__(self):
        # 初始化Markdown解析器
        self.md_parser = mistune.create_markdown()
        logger.info("文档处理器初始化完成")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        if not validate_file(file_path):
            raise FileNotFoundError(f"无效的文件路径: {file_path}")
        
        file_extension = get_file_extension(file_path)
        file_name = os.path.basename(file_path)
        
        logger.info(f"开始处理文件: {file_name}, 类型: {file_extension}")
        try:
            if file_extension == ".pdf":
                content = self._process_pdf(file_path)
            elif file_extension == ".docx":
                content = self._process_docx(file_path)
            elif file_extension == ".md":
                content = self._process_markdown(file_path)
            elif file_extension == ".txt":
                content = self._process_txt(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_extension}")
            logger.info(f"文件处理完成: {file_name}, 内容长度: {len(content)}")
            return {
                "file_name": file_name,
                "file_path": file_path,
                "file_type": file_extension,
                "content": content
            }
        except Exception as e:
            logger.error(f"处理文件 {file_name} 时出错: {str(e)}")
            raise
    
    def _process_pdf(self, file_path: str) -> str:
        """处理PDF文件"""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)
                text = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text.append(page.extract_text() or "")
                return "\n".join(text)
        except Exception as e:
            logger.error(f"解析PDF文件时出错: {str(e)}")
            raise
    
    def _process_docx(self, file_path: str) -> str:
        """处理Word文档"""
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            return "\n".join(text)
        except Exception as e:
            logger.error(f"解析Word文档时出错: {str(e)}")
            raise
    
    def _process_markdown(self, file_path: str) -> str:
        """处理Markdown文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as file:
                content = file.read()
            return content
        except Exception as e:
            logger.error(f"解析Markdown文件时出错: {str(e)}")
            raise
    
    def _process_txt(self, file_path: str) -> str:
        """处理TXT文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as file:
                return file.read()
        except Exception as e:
            logger.error(f"解析TXT文件时出错: {str(e)}")
            raise
    
    def process_files_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """批量处理多个文件"""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"跳过文件 {file_path}, 错误: {str(e)}")
        return results

document_processor = DocumentProcessor()

__all__ = ["DocumentProcessor", "document_processor"]