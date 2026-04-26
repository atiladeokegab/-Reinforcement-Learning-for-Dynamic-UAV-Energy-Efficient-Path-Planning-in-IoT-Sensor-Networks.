"""
DQN Ablation Study - Component Diagram
(Covers the 4 ablation variants A1–A4 and the control model.)
"""
from pathlib import Path
import logging
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating DQN Ablation Study Component diagram...")

    output_dir = CURRENT_DIR / "asset" / "diagrams" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dqn_ablation_component_diagram"

    graph_attr = {"splines": "spline", "pad": "0.5", "nodesep": "0.8", "ranksep": "1.0"}

    try:
        with Diagram(
            "Component Diagram - DQN Ablation Study (A1–A4)",
            direction="TB",
            graph_attr=graph_attr,
            show=False,
            filename=str(output_file),
            outformat="png",
        ):
            with Cluster("Baseline: Full DQN (dqn_final.zip)"):
                baseline = Python(
                    "Full DQN\n"
                    "Capture effect ON\n"
                    "EMA-ADR (λ=0.1)\n"
                    "AoI observation ON\n"
                    "Domain randomisation ON"
                )

            with Cluster("A1 - No Capture Effect"):
                a1_env = Python("UAVEnvironment\ncapture_effect=False\nAll same-SF collisions → loss")
                a1_dqn = Python("DQN Agent\n(same policy weights)")
                a1_env >> Edge(label="modified channel") >> a1_dqn

            with Cluster("A2 - Instant ADR"):
                a2_env = Python("UAVEnvironment\nadr_lambda=1.0\n(no EMA smoothing)")
                a2_sensor = Python("IoTSensor\nInstant SF update\non each RSSI sample")
                a2_env >> Edge(label="volatile SF") >> a2_sensor

            with Cluster("A3 - No AoI Observation"):
                a3_env = Python("UAVEnvironment\nurgency_features=zeros\nAoI hidden from agent")
                a3_dqn = Python("DQN Agent\nBlind to sensor urgency\n/ time-since-visit")
                a3_env >> Edge(label="masked obs") >> a3_dqn

            with Cluster("A4 - No Domain Randomisation (Control)"):
                a4_train = Python(
                    "train_ablation_a4.py\n"
                    "Fixed: 500×500, N=20\n"
                    "No curriculum\n"
                    "→ models/dqn_no_dr/"
                )
                a4_model = Python("dqn_no_dr model\n(fixed-env specialist)")
                a4_train >> Edge(label="produces") >> a4_model

            with Cluster("Ablation Runner (ablation_study.py)"):
                runner = Python(
                    "ablation_study.py\n"
                    "Loads each variant\n"
                    "Runs N episodes\n"
                    "Records: reward,\n"
                    "coverage, fairness,\n"
                    "bytes collected"
                )
                results = Python(
                    "fig_ablation_bars.png\n"
                    "fig_ablation_fairness.png\n"
                    "(in ablation_results/)"
                )
                runner >> Edge(label="generates") >> results

            # Connections showing what ablation_study.py tests
            baseline >> Edge(label="compared against", style="dashed") >> runner
            a1_dqn >> Edge(label="A1 variant", style="dashed") >> runner
            a2_sensor >> Edge(label="A2 variant", style="dashed") >> runner
            a3_dqn >> Edge(label="A3 variant", style="dashed") >> runner
            a4_model >> Edge(label="A4 control", style="dashed") >> runner

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

    logger.info("DQN ablation component diagram complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Running DQN ablation component diagram script...")
    create_diagram()
    logger.info("Script finished!")
