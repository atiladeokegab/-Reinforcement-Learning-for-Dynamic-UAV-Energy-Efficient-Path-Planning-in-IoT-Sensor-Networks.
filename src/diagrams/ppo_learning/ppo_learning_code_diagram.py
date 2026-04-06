"""
Greedy Baselines - Code Diagram
(Shows the class structure of NearestSensorGreedy and MaxThroughputGreedy
 from greedy_agents.py, and how they interface with UAVEnvironment.)
"""
from pathlib import Path
import logging
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating Greedy Baselines Code Diagram...")

    output_dir = CURRENT_DIR / "asset" / "diagrams" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "greedy_baselines_code_diagram"

    graph_attr = {"splines": "ortho", "pad": "0.5", "nodesep": "0.7", "ranksep": "1.2"}

    try:
        with Diagram(
            "Code Diagram - Greedy Baseline Agents (greedy_agents.py)",
            direction="TB",
            graph_attr=graph_attr,
            show=False,
            filename=str(output_file),
            outformat="png",
        ):
            with Cluster("src/agents/dqn/dqn_evaluation_results/"):
                greedy_file = Python(
                    "greedy_agents.py\n[Module]\n"
                    "Defines two heuristic baselines\n"
                    "for benchmarking against DQN"
                )

            with Cluster("NearestSensorGreedy"):
                nearest_agent = Python(
                    "NearestSensorGreedy\n"
                    "+ env: UAVEnvironment\n"
                    "+ target_sensor\n"
                    "——\n"
                    "+ select_action()\n"
                    "  1. Check sensors in range\n"
                    "     with data_buffer > 0\n"
                    "  2. If YES → ACTION_COLLECT (4)\n"
                    "  3. If NO → _move_toward(\n"
                    "     nearest sensor by Euclidean dist)\n"
                    "  4. _move_toward: compare |dx| vs |dy|\n"
                    "     → UP/DOWN/LEFT/RIGHT"
                )

            with Cluster("MaxThroughputGreedy (SF-Aware)"):
                max_tp_agent = Python(
                    "MaxThroughputGreedy\n"
                    "+ env: UAVEnvironment\n"
                    "+ target_sensor\n"
                    "——\n"
                    "+ select_action()\n"
                    "  1. Check sensors in range:\n"
                    "     data_buffer>0 AND SF≤9\n"
                    "  2. If good SF in range →\n"
                    "     collect from lowest SF\n"
                    "     (highest data rate)\n"
                    "  3. If not → find global\n"
                    "     lowest-SF sensor,\n"
                    "     move toward it\n"
                    "  4. SF priority: SF7>SF9>SF11>SF12"
                )

            with Cluster("Data Rate Reference (LORA_DATA_RATES)"):
                data_rates = Python(
                    "SF7  → 5470/8 = 683.8 B/s\n"
                    "SF8  → 3125/8 = 390.6 B/s\n"
                    "SF9  → 1760/8 = 220.0 B/s\n"
                    "SF10 →  980/8 = 122.5 B/s\n"
                    "SF11 →  440/8 =  55.0 B/s\n"
                    "SF12 →  250/8 =  31.3 B/s"
                )

            with Cluster("src/environment/"):
                env_class = Python(
                    "UAVEnvironment\n"
                    "+ uav.position\n"
                    "+ sensors[i].data_buffer\n"
                    "+ sensors[i].spreading_factor\n"
                    "+ sensors[i].is_in_range(pos)\n"
                    "——\n"
                    "Actions:\n"
                    "0=North, 1=South\n"
                    "2=East,  3=West\n"
                    "4=Hover/Collect"
                )

            with Cluster("compare_agents.py"):
                compare = Python(
                    "compare_agents.py\n"
                    "Instantiates both greedy agents\n"
                    "Runs same seeded episodes\n"
                    "Records: bytes, fairness, reward\n"
                    "Produces comparison PNGs"
                )

            # Relationships
            greedy_file >> Edge(label="defines", style="solid") >> nearest_agent
            greedy_file >> Edge(label="defines", style="solid") >> max_tp_agent

            nearest_agent >> Edge(label="reads env state") >> env_class
            max_tp_agent >> Edge(label="reads env state") >> env_class
            max_tp_agent >> Edge(label="uses SF→rate mapping") >> data_rates

            compare >> Edge(label="instantiates & runs", style="dashed") >> nearest_agent
            compare >> Edge(label="instantiates & runs", style="dashed") >> max_tp_agent
            compare >> Edge(label="steps") >> env_class

            logger.info("✓ Diagram components created")

        png_file = output_file.with_suffix(".png")
        if png_file.exists():
            logger.info(f"✓ Diagram saved: {png_file} ({png_file.stat().st_size:,} bytes)")
        else:
            logger.warning("⚠ Output file not found")

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    logger.info("Greedy baselines code diagram complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Running greedy baselines code diagram script...")
    create_diagram()
    logger.info("Script finished!")
