from orchestration.state import ResearchState

def supervisor_agent(state: ResearchState) -> ResearchState:

    # INITIAL ENTRY — bootstrap flow
    if not state.get("next_step"):
        state["next_step"] = "research"
        print("[supervisor] Bootstrapping → research")
        return state

    # After analysis, supervisor decides
    if state.get("analysis_decision") == "ready":
        state["next_step"] = "summarize"
        print("[supervisor] Analysis ready → summarize")

    elif state.get("analysis_decision") == "need_more_info":
        state["next_step"] = "research"
        print("[supervisor] Need more info → research")
        
    else:
        state["next_step"] = "end"
        print("[supervisor] Ending")
    return state

