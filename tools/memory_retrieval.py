# main.py
import sys
from fetch_web import FetchWebTool  # your fetcher function
from vector_memory import VectorMemory
from graph_memory import GraphMemory

# Initialize memory layers
vector_mem = VectorMemory()
graph_mem = GraphMemory()

print("=== Research Agent CLI ===")
print("Type your query, or 'quit' to exit.")

while True:
    query = input("\nQuery> ").strip()
    if query.lower() in ["quit", "exit"]:
        print("Exiting Research Agent. Bye!")
        sys.exit(0)

    # --- Step 1: Fetch new documents ---
    print(f"\nFetching new documents for query: '{query}'...")
    new_docs = FetchWebTool().fetch_query(query, n_results=5)  # Should return list of dicts: {'url', 'chunk_id', 'text'}
    #print(new_docs)
    if not new_docs:
        print("No new documents found. Using existing memory.")

    # Step 2: Update vector memory and get chunks
    for doc in new_docs:
        chunks = vector_mem.add_document(doc['url'], doc['text'])
        
    # Step 3: Update graph memory using same chunks
        for chunk_id, chunk_text in chunks:
            graph_mem.add_chunk(doc['url'], chunk_id=chunk_id, text=chunk_text)
      
    # --- Step 4: Retrieve top-k relevant chunks ---
    print("\nRetrieving relevant chunks from vector memory...")
    top_chunks = vector_mem.search(query)
    # NEW STEP: also push these chunks into graph memory so entities are indexed
    for idx, chunk_data in enumerate(top_chunks):
        graph_mem.add_chunk(
            url=chunk_data["url"],
            chunk_id=idx,              # temporary id
            text=chunk_data["chunk"]   # actual chunk content
        )

    # --- Step 5: Retrieve related entities from graph memory ---
    print("\nRetrieving related entities from graph memory...")
    related_entities = []
    for chunk in top_chunks:
        # Attempt to find a main entity in chunk (simplest: first word as demo)
        main_entity = chunk['chunk'].split()[0]  # optional: improve with actual NER
        ents = graph_mem.query_entities(main_entity)
        related_entities.extend(ents)

    if not related_entities:
        print("No related entities found.")
    else:
        print("\nRelated entities:")
        for e in related_entities:
            print(f"{e['source']} --{e['relation']}--> {e['target']} | meta: {e['meta']}")

    # --- Step 6: Optional: Summarize (using small LLM or placeholder) ---
    print("\nSummary placeholder:")
    print(f"Query: '{query}' - Retrieved {len(top_chunks)} chunks and {len(related_entities)} entity connections.")
    print("You can feed these to an LLM to generate a detailed report.")