from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from asr_multiagent.tools.lattice_builder import (
    build_lattice_for_utterance,
    comparison_rows,
    leave_one_out_lattice_score,
    lattice_wer_for_model,
    standard_wer,
)


class Q4State(TypedDict, total=False):
    model_transcriptions: List[dict]
    human_reference: List[str]
    lattice: List[dict]
    lattice_wer_results: Dict[str, float]
    standard_wer_results: Dict[str, float]
    comparison_table: List[dict]


def load_transcriptions_node(state: Q4State) -> Q4State:
    refs = [f"यह वाक्य संख्या {i} है" for i in range(1, 21)]
    models = []
    for m in range(5):
        hyps = []
        for i, r in enumerate(refs):
            if i % (m + 2) == 0:
                hyps.append(r.replace("संख्या", "नंबर"))
            elif i % 7 == 0:
                hyps.append(r.replace("है", ""))
            else:
                hyps.append(r)
        models.append({"model_name": f"model_{m+1}", "transcripts": hyps})
    state["human_reference"] = refs
    state["model_transcriptions"] = models
    return state


def build_lattice_node(state: Q4State) -> Q4State:
    lattice = []
    for i, ref in enumerate(state["human_reference"]):
        hyps = [m["transcripts"][i] for m in state["model_transcriptions"]]
        lattice.append(build_lattice_for_utterance(ref, hyps).to_serializable())
    state["lattice"] = lattice
    return state


def compute_lattice_wer_node(state: Q4State) -> Q4State:
    standard, lattice_scores = {}, {}
    refs = state["human_reference"]
    utterance_model_hypotheses = [
        {
            model["model_name"]: model["transcripts"][utterance_idx]
            for model in state["model_transcriptions"]
        }
        for utterance_idx in range(len(refs))
    ]
    for m in state["model_transcriptions"]:
        name, hyps = m["model_name"], m["transcripts"]
        standard[name] = standard_wer(refs, hyps)
        lw = []
        for i, h in enumerate(hyps):
            lw.append(leave_one_out_lattice_score(refs[i], name, utterance_model_hypotheses[i]))
        lattice_scores[name] = sum(lw) / max(1, len(lw))
    state["standard_wer_results"] = standard
    state["lattice_wer_results"] = lattice_scores
    state["comparison_table"] = comparison_rows(standard, lattice_scores)
    return state


def build_graph():
    graph = StateGraph(Q4State)
    graph.add_node("load_transcriptions", load_transcriptions_node)
    graph.add_node("build_lattice", build_lattice_node)
    graph.add_node("compute_lattice_wer", compute_lattice_wer_node)
    graph.add_edge(START, "load_transcriptions")
    graph.add_edge("load_transcriptions", "build_lattice")
    graph.add_edge("build_lattice", "compute_lattice_wer")
    graph.add_edge("compute_lattice_wer", END)
    return graph.compile()
