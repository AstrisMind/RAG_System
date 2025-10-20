# RAG系统项目

## 项目介绍

本项目基于LangChain和LangGraph构建的RAG系统项目。该系统支持多种文档格式的上传与解析、文本分段与向量化、向量检索与智能回答，以及多轮对话功能。系统还集成了LangSmith可观测性，便于监控和调试。

## 核心功能

### 1. 文档上传与解析
- 支持PDF、Word (.docx)、Markdown (.md)、TXT文件上传
- 自动提取文本内容，保留文档结构信息

### 2. 文本分段与向量化
- 智能分段算法，支持自定义分段大小和重叠度
- 使用BGE Embeddings模型进行向量化
- 基于FAISS构建高效向量检索索引

### 3. 向量检索与回答生成
- 基于语义相似度的高效检索
- 集成LLM（Qwen3）生成准确回答
- 回答中自动引用原文段落来源

### 4. 多轮对话能力
- 保留完整对话上下文
- 支持连续提问和补充提问
- 基于历史对话生成连贯回答

### 5. Agent调用可观测性
- 集成LangSmith平台，记录完整执行轨迹
- 自动上报输入、输出、响应时间、相似度分数等信息
- 提供可视化执行流程

## 技术栈

- **Python 3.11+**：核心编程语言
- **LangChain**：构建RAG系统的核心框架
- **LangGraph**：构建Agent执行流程图
- **LangSmith**：提供可观测性和追踪功能
- **BGE Embeddings**：用于文本嵌入
- **Qwen API**：用于生成回答
- **FAISS**：高效向量数据库
- **pypdf/python-docx/mistune**：文档解析库
- **loguru**：结构化日志管理
- **uv**：依赖管理工具

## 项目结构

```
RAG_System/
├── main.py              # 主程序入口
├── demo.py              # 演示脚本
├── utils.py             # 工具函数和日志配置
├── document_processor.py  # 文档处理模块
├── text_processor.py    # 文本分段和向量化模块
├── rag_engine.py        # RAG引擎核心模块
├── agent_flow.py        # LangGraph Agent流程模块
├── pyproject.toml       # 项目配置和依赖
└── data/                # 示例文档目录
```

## 安装与配置

### 1. 克隆项目

```bash
git clone 
cd RAG_System
```

### 2. 安装依赖

使用uv安装项目依赖：

```bash
# 安装uv
pip install uv

# 安装项目依赖
uv sync
```

### 3. 配置环境变量

编辑`.env`文件,配置API_KEY

```
Qwen_API_KEY=your_qwen_api_key
LangSmith_API_KEY=your_langsmith_api_key
```

### 快速开始

```bash
python demo.py /   python main.py "path/to/your/document.pdf" "your question"
```
