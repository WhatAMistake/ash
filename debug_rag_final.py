import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path.cwd() / "src"))

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from writer_rag import WriterRAG

    print("--- СТАТУС RAG (ВЕКТОРНЫЙ РЕЖИМ) ---")
    
    # Инициализируем RAG
    # При первом запуске это может занять время (индексация)
    rag = WriterRAG()
    
    print(f"1. Библиотеки: OK")
    print(f"2. Эмбеддер: {'Загружен' if rag.embedder else 'ОШИБКА'}")
    
    # Проверяем коллекции ChromaDB
    print(f"3. Коллекции в ChromaDB:")
    if not rag.collections:
        print("   - Коллекции не инициализированы!")
    else:
        for name, col in rag.collections.items():
            count = col.count()
            print(f"   - {name}: {count} документов")

    if any(col.count() > 0 for col in rag.collections.values()):
        print("\n--- ТЕСТ СЕМАНТИЧЕСКОГО ПОИСКА ---")
        query = "как писать диалоги"
        results = rag.search_similar(query, n_results=2)
        
        for i, res in enumerate(results):
            print(f"\nРезультат #{i+1}:")
            print(f"Источник: {res.source}")
            print(f"Релевантность (cosine): {res.relevance:.4f}")
            print(f"Текст: {res.content[:150]}...")
    else:
        print("\nВНИМАНИЕ: База пуста. Возможно, идет процесс индексации или не найдены JSON-файлы.")

except Exception as e:
    print(f"\nОШИБКА: {e}")
    import traceback
    traceback.print_exc()