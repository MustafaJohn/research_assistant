"""
main.py — CLI entry point for local testing.
Note: the /api/fetch + /api/summarize split is API-only.
CLI runs the full LangGraph pipeline in one shot.
"""

from orchestration.graph import build_graph

graph = build_graph()
print("=== Multi-Agent Research System ===")

while True:
    q = input("\nEnter your Research Area> ").strip()
    if q.lower() in {"quit", "exit"}:
        break

    result = graph.invoke({
        "query":             q,
        "sort_by":           "relevance",
        "fetched_docs":      [],
        "vector_results":    [],
        "graph_results":     [],
        "final_context":     "",
        "next_step":         "",
        "analysis_decision": "",
        "sources":           [],
        "max_results":       10,
        "logs":              [],
    })

    print("\nFINAL ANSWER:\n")
    print(result["final_context"])

    if result.get("sources"):
        print("\nSOURCES:")
        for s in result["sources"]:
            oa = " [OPEN ACCESS]" if s.get("is_open_access") else ""
            print(f"  - {s['title']} ({s.get('authors','')}, {s.get('year','')}){oa}")
            print(f"    {s['url']}")
