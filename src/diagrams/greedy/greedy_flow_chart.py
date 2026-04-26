"""
Flowchart for MaxThroughputGreedyV2 using Graphviz.

V2 is SF-aware with a multi-objective scoring function:
  score = SF_priority × w_sf + buffer_fill × w_buf
          + duty_cycle × w_duty − distance_normalised × w_dist

Phase 1 — if in-range sensors have data and SF ≤ adaptive threshold: collect best
Phase 2 — navigate to globally highest-scored sensor (SF + buffer + duty − distance)

Author: ATILADE GABRIEL OKE
"""

import graphviz
from pathlib import Path


def create_max_throughput_flowchart():
    CURRENT_DIR = Path(__file__).resolve().parent.parent.parent.parent
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "greedy_flow_charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    dot = graphviz.Digraph(
        name="MaxThroughputGreedyV2",
        comment="MaxThroughputGreedyV2 Algorithm Flowchart",
        format="png",
    )
    dot.attr(rankdir="TB")
    dot.attr(
        "graph",
        bgcolor="white",
        splines="line",
        nodesep="0.5",
        ranksep="0.7",
        margin="0.5",
    )
    dot.attr("node", fontname="Arial", fontsize="10", margin="0.3,0.2")
    dot.attr("edge", fontname="Arial", fontsize="9", color="black")

    start_end = dict(shape="ellipse", fillcolor="#90EE90", style="filled", penwidth="2")
    process   = dict(shape="box",     fillcolor="#87CEEB", style="filled")
    decision  = dict(shape="diamond", fillcolor="#FFD700", style="filled", penwidth="2")
    collect   = dict(shape="box",     fillcolor="#FF6B9D", style="filled",
                     fontcolor="white", fontweight="bold")
    score_box = dict(shape="box",     fillcolor="#FFB6C1", style="filled", penwidth="2")

    # Nodes
    dot.node("START",   "START\nselect_action()", **start_end)
    dot.node("GET_POS", "Get UAV state\nposition, battery%, steps_left", **process)
    dot.node("SF_THR",  "Compute adaptive SF threshold\n(relaxes when battery < 30%\nor steps_left < 150)", **process)

    dot.node(
        "CHECK_IMMED",
        "In-range sensors with\ndata_buffer > 0\nAND SF ≤ sf_threshold?",
        **decision,
    )

    dot.node(
        "BEST_IMMED",
        "Score in-range candidates:\n"
        "key = (SF_priority, data_buffer, −distance)\n"
        "Select max → target_sensor",
        **score_box,
    )
    dot.node("COLLECT_IMMED", "Return ACTION_COLLECT", **collect)

    dot.node(
        "GLOBAL_SCORE",
        "Score ALL sensors with data_buffer > 0:\n"
        "score = SF_priority × w_sf\n"
        "      + (buffer/max_buffer) × w_buf\n"
        "      + duty_cycle_prob × w_duty\n"
        "      − (distance/grid_size) × w_dist\n"
        "(weights adapt to battery/time urgency)",
        **score_box,
    )
    dot.node("BEST_GLOBAL", "Select sensor with max score\n→ target_sensor", **process)

    dot.node("FOUND", "target_sensor\nfound?", **decision)
    dot.node("NO_TARGET",  "Return ACTION_COLLECT\n(no data left)", **collect)

    dot.node("AT_TARGET", "UAV within 0.5 units\nof target?", **decision)
    dot.node("COLLECT_AT", "Return ACTION_COLLECT", **collect)
    dot.node(
        "MOVE_DIR",
        "Determine direction:\n|dx| > |dy| → LEFT/RIGHT\n|dy| ≥ |dx| → UP/DOWN",
        **process,
    )
    dot.node("RETURN_MOVE", "Return movement action\n(UP / DOWN / LEFT / RIGHT)", **process)
    dot.node("END", "END\nAction returned", **start_end)

    # Edges
    dot.edge("START",       "GET_POS")
    dot.edge("GET_POS",     "SF_THR")
    dot.edge("SF_THR",      "CHECK_IMMED")

    dot.edge("CHECK_IMMED", "BEST_IMMED",    label="YES", color="green",   fontcolor="green",  penwidth="2")
    dot.edge("BEST_IMMED",  "COLLECT_IMMED", color="green")
    dot.edge("COLLECT_IMMED","END",          color="green")

    dot.edge("CHECK_IMMED", "GLOBAL_SCORE",  label="NO",  color="red",     fontcolor="red",    penwidth="2")
    dot.edge("GLOBAL_SCORE","BEST_GLOBAL",   color="red")
    dot.edge("BEST_GLOBAL", "FOUND",         color="red")

    dot.edge("FOUND",       "NO_TARGET",     label="NO",  color="red",     fontcolor="red",    penwidth="2")
    dot.edge("NO_TARGET",   "END",           color="red")

    dot.edge("FOUND",       "AT_TARGET",     label="YES", color="green",   fontcolor="green",  penwidth="2")
    dot.edge("AT_TARGET",   "COLLECT_AT",    label="YES", color="green",   fontcolor="green",  penwidth="2")
    dot.edge("COLLECT_AT",  "END",           color="green")
    dot.edge("AT_TARGET",   "MOVE_DIR",      label="NO",  color="orange",  fontcolor="orange", penwidth="2")
    dot.edge("MOVE_DIR",    "RETURN_MOVE",   color="orange")
    dot.edge("RETURN_MOVE", "END",           color="orange")

    output_file = output_dir / "MaxThroughputGreedyV2"
    dot.render(filename="MaxThroughputGreedyV2", directory=str(output_dir), cleanup=True)

    png = output_file.with_suffix(".png")
    if png.exists():
        print(f"Flowchart saved: {png}  ({png.stat().st_size / 1024:.1f} KB)")
    else:
        print("Warning: PNG not created")
    return str(png)


if __name__ == "__main__":
    print("Generating MaxThroughputGreedyV2 flowchart...")
    create_max_throughput_flowchart()
