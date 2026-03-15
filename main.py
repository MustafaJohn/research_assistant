
from orchestration.graph import build_graph

graph = build_graph()

print("=== Multi-Agent Research System ===")

while True:
    q = input("\nEnter your Research Area> ").strip()
    if q.lower() in {"quit", "exit"}:
        break

    result = graph.invoke({
        "query": q,
        "fetched_docs": [],
        "vector_results": [],
        "graph_results": [],
        "final_context": "",
        "next_step": ""
    })

    print("\nFINAL ANSWER:\n")
    print(result["final_context"])