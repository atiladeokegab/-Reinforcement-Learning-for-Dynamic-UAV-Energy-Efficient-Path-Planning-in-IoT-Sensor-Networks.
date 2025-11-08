"""
Flowchart for MaxThroughputGreedy Algorithm using Graphviz
 SF-AWARE: Prioritize sensors with BEST data rates (LOWEST SF)

Key Insight:
- SF7 = 684 B/s (best, needs close positioning)
- SF9 = 220 B/s (medium, medium distance)
- SF11 = 55 B/s (poor, far distance)
- SF12 = 31 B/s (worst, very far)

Strategy:
1. Collect from SF9 or better sensors if in range
2. Otherwise, move toward sensor with LOWEST SF (highest data rate)
3. Forces position optimization for data rate!

Author: ATILADE GABRIEL OKE
Date: November 2025
"""

import graphviz
from pathlib import Path



def create_max_throughput_flowchart():
    """Create flowchart for MaxThroughputGreedy algorithm using Graphviz."""

    # Create output directory
    CURRENT_DIR = Path(__file__).parent.parent.parent.parent
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "greedy_flow_charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Graphviz digraph (don't use directory parameter)
    dot = graphviz.Digraph(
        name='MaxThroughputGreedy',
        comment='MaxThroughputGreedy Algorithm Flowchart (SF-AWARE)',
        format='png'
    )

    # Set graph attributes for professional appearance
    dot.attr(rankdir='TB')  # Top to Bottom
    dot.attr('graph',
             bgcolor='white',
             splines='line',  # Changed from curved to line for edge labels
             nodesep='0.5',
             ranksep='0.7',
             margin='0.5')

    dot.attr('node',
             fontname='Arial',
             fontsize='10',
             margin='0.3,0.2')

    dot.attr('edge',
             fontname='Arial',
             fontsize='9',
             color='black')

    # Define node styles
    start_end_attrs = dict(
        shape='ellipse',
        fillcolor='#90EE90',  # Light green
        style='filled',
        fontcolor='black',
        penwidth='2'
    )

    process_attrs = dict(
        shape='box',
        fillcolor='#87CEEB',  # Light blue
        style='filled',
        fontcolor='black'
    )

    decision_attrs = dict(
        shape='diamond',
        fillcolor='#FFD700',  # Gold/Yellow
        style='filled',
        fontcolor='black',
        penwidth='2'
    )

    collect_attrs = dict(
        shape='box',
        fillcolor='#FF6B9D',  # Pink/Red
        style='filled',
        fontcolor='white',
        fontweight='bold'
    )

    sf_check_attrs = dict(
        shape='box',
        fillcolor='#FFB6C1',  # Light pink (SF-aware)
        style='filled',
        fontcolor='black',
        penwidth='2'
    )

    # ========================
    # CREATE NODES
    # ========================

    # Start
    dot.node('START',
             'START\nselect_action()\n✅ SF-AWARE MODE',
             **start_end_attrs)

    # Get UAV position
    dot.node('GET_POS',
             'Get UAV Position\nself.env.uav.position',
             **process_attrs)

    # Main decision: Check for good SF in range
    dot.node('CHECK_GOOD_SF',
             'Are sensors in range\nwith data_buffer > 0\nAND SF ≤ 9?\n(GOOD SF)',
             **decision_attrs)

    # YES path: Filter good SF sensors
    dot.node('FILTER_GOOD_SF',
             'Filter collectible sensors:\ndata_buffer > 0\nAND in_range\nAND spreading_factor ≤ 9',
             **sf_check_attrs)

    # Find best SF among good sensors
    dot.node('FIND_BEST_SF',
             'Find BEST SF sensor\nSort by: (-spreading_factor, data_buffer)\nPrioritize: Lowest SF + Highest buffer',
             **sf_check_attrs)

    # Set target and collect
    dot.node('SET_TARGET_GOOD',
             'Set: self.target_sensor = best_sensor',
             **process_attrs)

    dot.node('COLLECT_GOOD',
             'Return ACTION_COLLECT',
             **collect_attrs)

    # NO path: Find globally lowest SF
    dot.node('FIND_GLOBAL',
             'Search ALL sensors\nwith data_buffer > 0',
             **process_attrs)

    # Sort by SF globally
    dot.node('SORT_GLOBALLY',
             'Sort by:\n1️⃣ LOWEST spreading_factor\n2️⃣ HIGHEST data_buffer',
             **sf_check_attrs)

    # Select best sensor globally
    dot.node('GET_BEST_GLOBAL',
             'Select: min(sensors)\nby (spreading_factor, -data_buffer)',
             **sf_check_attrs)

    # Check if sensor found
    dot.node('CHECK_FOUND',
             'Sensor\nfound?',
             **decision_attrs)

    # If no sensor found
    dot.node('NO_SENSOR',
             'Return ACTION_COLLECT',
             **collect_attrs)

    # If sensor found, move toward it
    dot.node('MOVE_TOWARD',
             'Call: self._move_toward(target_sensor.position)\nOptimize position for SF coverage',
             **process_attrs)

    # Calculate target distance
    dot.node('CALC_DISTANCE',
             'Calculate: dx, dy\nto target position',
             **process_attrs)

    # Check if at target
    dot.node('AT_TARGET',
             'At target?\n(dx ≤ 0.5 & dy ≤ 0.5)',
             **decision_attrs)

    # If at target, collect
    dot.node('AT_TARGET_COLLECT',
             'Return ACTION_COLLECT',
             **collect_attrs)

    # If not at target, determine direction
    dot.node('DETERMINE_DIR',
             'Determine movement direction:\ncompare |dx| vs |dy|\nreturn: UP/DOWN/LEFT/RIGHT',
             **process_attrs)

    # Return movement action
    dot.node('RETURN_ACTION',
             'Return movement action\n(ACTION_UP/DOWN/LEFT/RIGHT)',
             **process_attrs)

    # End
    dot.node('END',
             'END\nAction selected',
             **start_end_attrs)

    # ========================
    # CREATE EDGES
    # ========================

    # Main flow
    dot.edge('START', 'GET_POS', label='', color='black')
    dot.edge('GET_POS', 'CHECK_GOOD_SF', label='', color='black')

    # YES path: Good SF sensors in range
    dot.edge('CHECK_GOOD_SF', 'FILTER_GOOD_SF',
             label='YES\n(Good SF available)',
             fontcolor='green',
             color='green',
             penwidth='2')

    dot.edge('FILTER_GOOD_SF', 'FIND_BEST_SF', label='', color='green')
    dot.edge('FIND_BEST_SF', 'SET_TARGET_GOOD', label='', color='green')
    dot.edge('SET_TARGET_GOOD', 'COLLECT_GOOD', label='', color='green')
    dot.edge('COLLECT_GOOD', 'END', label='', color='green')

    # NO path: No good SF sensors
    dot.edge('CHECK_GOOD_SF', 'FIND_GLOBAL',
             label='NO\n(Search globally)',
             fontcolor='red',
             color='red',
             penwidth='2')

    dot.edge('FIND_GLOBAL', 'SORT_GLOBALLY', label='', color='red')
    dot.edge('SORT_GLOBALLY', 'GET_BEST_GLOBAL', label='', color='red')
    dot.edge('GET_BEST_GLOBAL', 'CHECK_FOUND', label='', color='red')

    # Sensor not found
    dot.edge('CHECK_FOUND', 'NO_SENSOR',
             label='NO',
             fontcolor='red',
             color='red',
             penwidth='2')
    dot.edge('NO_SENSOR', 'END', label='', color='red')

    # Sensor found: Move toward
    dot.edge('CHECK_FOUND', 'MOVE_TOWARD',
             label='YES',
             fontcolor='green',
             color='green',
             penwidth='2')

    dot.edge('MOVE_TOWARD', 'CALC_DISTANCE', label='', color='green')
    dot.edge('CALC_DISTANCE', 'AT_TARGET', label='', color='green')

    # At target
    dot.edge('AT_TARGET', 'AT_TARGET_COLLECT',
             label='YES',
             fontcolor='green',
             color='green',
             penwidth='2')
    dot.edge('AT_TARGET_COLLECT', 'END', label='', color='green')

    # Not at target: Determine direction
    dot.edge('AT_TARGET', 'DETERMINE_DIR',
             label='NO',
             fontcolor='red',
             color='red',
             penwidth='2')

    dot.edge('DETERMINE_DIR', 'RETURN_ACTION', label='', color='red')
    dot.edge('RETURN_ACTION', 'END', label='', color='red')

    # Render the flowchart
    try:
        # Create full path for output
        output_file = output_dir / "MaxThroughputGreedy"

        # Render to file
        dot.render(filename='MaxThroughputGreedy', directory=str(output_dir), cleanup=True)

        png_file = output_file.with_suffix('.png')
        if png_file.exists():
            file_size = png_file.stat().st_size
            print(f"✅ Flowchart created successfully!")
            print(f"📊 File: {png_file}")
            print(f"📈 Size: {file_size / 1024:.1f} KB")
            print(f"\n✨ Flowchart ready to view!")
            return str(png_file)
        else:
            print("❌ PNG file was not created")
            return None

    except Exception as e:
        print(f"❌ Error creating flowchart: {e}")
        raise


if __name__ == "__main__":
    print("=" * 80)
    print("MaxThroughputGreedy Algorithm - Flowchart Generator")
    print("✅ SF-AWARE (Spreading Factor Aware)")
    print("=" * 80)
    print()
    print("Creating flowchart for MaxThroughputGreedy algorithm...")
    print()

    create_max_throughput_flowchart()

    print()
    print("=" * 80)
    print("Done! Check output/flowcharts/ for your flowchart.")
    print("=" * 80)