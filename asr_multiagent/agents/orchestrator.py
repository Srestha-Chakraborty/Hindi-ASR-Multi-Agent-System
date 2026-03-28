import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from asr_multiagent.agents import q1_asr_agent, q2_cleanup_agent, q3_spelling_agent, q4_lattice_agent
from asr_multiagent.agents.production_agents import build_production_graph, run_production_pipeline


class SupervisorState(TypedDict, total=False):
    active_agent: str
    task_queue: List[str]
    results: Dict[str, Any]
    user_config: Dict[str, Any]


def _run_q1(state: SupervisorState) -> SupervisorState:
    cfg = state["user_config"]
    result = q1_asr_agent.build_graph().invoke({"manifest_path": cfg["manifest_path"], "demo_mode": cfg.get("demo_mode", True)})
    state["results"]["q1"] = result
    return state


def _run_q2(state: SupervisorState) -> SupervisorState:
    cfg = state["user_config"]
    state["results"]["q2"] = q2_cleanup_agent.build_graph().invoke({"manifest_path": cfg["manifest_path"]})
    return state


def _run_q3(state: SupervisorState) -> SupervisorState:
    cfg = state["user_config"]
    state["results"]["q3"] = q3_spelling_agent.build_graph().invoke({"manifest_path": cfg["manifest_path"]})
    return state


def _run_q4(state: SupervisorState) -> SupervisorState:
    state["results"]["q4"] = q4_lattice_agent.build_graph().invoke({})
    return state


def _route_from_start(state: SupervisorState):
    selected = state["user_config"]["selected_questions"]
    if state["user_config"].get("parallel_mode", False):
        if "q1" in selected:
            return "q1"
        return "parallel_rest"
    if "q1" in selected:
        return "q1"
    if "q2" in selected:
        return "q2"
    if "q3" in selected:
        return "q3"
    if "q4" in selected:
        return "q4"
    return "finish"


def _chain(state: SupervisorState):
    selected = state["user_config"]["selected_questions"]
    if state["user_config"].get("parallel_mode", False) and state.get("active_agent") == "q1":
        return "parallel_rest"
    done = state.get("active_agent", "")
    order = [x for x in ["q1", "q2", "q3", "q4"] if x in selected]
    if done in order:
        idx = order.index(done)
        if idx + 1 < len(order):
            return order[idx + 1]
    return "finish"


def _mark(agent_name: str):
    def inner(state: SupervisorState) -> SupervisorState:
        state["active_agent"] = agent_name
        return state
    return inner


def _run_parallel_rest(state: SupervisorState) -> SupervisorState:
    cfg = state["user_config"]
    selected = [q for q in cfg["selected_questions"] if q in {"q2", "q3", "q4"}]

    def _runner(q: str):
        if q == "q2":
            return "q2", q2_cleanup_agent.build_graph().invoke({"manifest_path": cfg["manifest_path"]})
        if q == "q3":
            return "q3", q3_spelling_agent.build_graph().invoke({"manifest_path": cfg["manifest_path"]})
        return "q4", q4_lattice_agent.build_graph().invoke({})

    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as ex:
        for key, val in ex.map(_runner, selected):
            state["results"][key] = val
    return state


def build_graph():
    graph = StateGraph(SupervisorState)
    graph.add_node("q1", _run_q1)
    graph.add_node("q2", _run_q2)
    graph.add_node("q3", _run_q3)
    graph.add_node("q4", _run_q4)
    graph.add_node("mark_q1", _mark("q1"))
    graph.add_node("mark_q2", _mark("q2"))
    graph.add_node("mark_q3", _mark("q3"))
    graph.add_node("mark_q4", _mark("q4"))
    graph.add_node("parallel_rest", _run_parallel_rest)
    graph.add_node("finish", lambda s: s)

    graph.add_conditional_edges(START, _route_from_start, {"q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4", "parallel_rest": "parallel_rest", "finish": "finish"})
    graph.add_edge("q1", "mark_q1")
    graph.add_edge("q2", "mark_q2")
    graph.add_edge("q3", "mark_q3")
    graph.add_edge("q4", "mark_q4")
    graph.add_conditional_edges("mark_q1", _chain, {"q2": "q2", "q3": "q3", "q4": "q4", "parallel_rest": "parallel_rest", "finish": "finish"})
    graph.add_conditional_edges("mark_q2", _chain, {"q3": "q3", "q4": "q4", "finish": "finish"})
    graph.add_conditional_edges("mark_q3", _chain, {"q4": "q4", "finish": "finish"})
    graph.add_conditional_edges("mark_q4", _chain, {"finish": "finish"})
    graph.add_edge("parallel_rest", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def run_suite(selected_questions: List[str], manifest_path: str, demo_mode: bool = True, parallel_mode: bool = True) -> Dict[str, Any]:
    state = {"active_agent": "", "task_queue": selected_questions, "results": {}, "user_config": {"selected_questions": selected_questions, "manifest_path": manifest_path, "demo_mode": demo_mode, "parallel_mode": parallel_mode}}
    result = build_graph().invoke(state)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/final_report.json", "w", encoding="utf-8") as f:
        json.dump(result["results"], f, ensure_ascii=False, indent=2, default=str)
    return result["results"]


__all__ = [
    "build_graph",
    "run_suite",
    "build_production_graph",
    "run_production_pipeline",
]
