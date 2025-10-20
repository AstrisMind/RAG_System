from main import create_rag_system
import time

agent = create_rag_system()
agent.process_document(r"D:\cyy\RAG_System\data\txt_test_sample.txt")
session_id = f"session_{int(time.time())}"
config = {"configurable": {"thread_id": session_id}}

while True:
    user_input = input("\n User: ").strip()
    if user_input.lower() in ['quit', '退出']:
        break
        
    start_time = time.time()
    result = agent.run(user_input, config)
    response_time = time.time() - start_time

    answer = result['answer']
    has_source_in_answer = "（来源：" in answer
    
    search_results = result.get('search_results', [])
    
    print(f"\n Robot: {answer}")
    
    if has_source_in_answer and search_results:
        print(f"\n 引用原文段落: ")
        for i, result_item in enumerate(search_results, 1):
            text = result_item.get('text', '')
            score = result_item.get('score', 0)
            metadata = result_item.get('metadata', {})
            file_name = metadata.get('file_name')
            
            print(f"\n({i}) 来源: {file_name} (相似度: {score:.4f})")
            print(f"原文: {text[:200]}..." if len(text) > 200 else f"原文: {text}")
    
    

            
