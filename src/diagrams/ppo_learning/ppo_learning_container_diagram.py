"""
DQN Evaluation Pipeline - Container Diagram
(Shows how evaluate_dqn.py, compare_agents.py, fairness_sweep.py,
 and ablation_study.py relate to each other and to the trained model.)
"""
from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import (
    Person,
    Container,
    Database,
    System,
    SystemBoundary,
    Relationship,
)

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating DQN Evaluation Pipeline Container diagram...")

    output_dir = CURRENT_DIR / "asset" / "diagrams" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dqn_evaluation_pipeline_container"

    graph_attr = {"splines": "spline"}

    try:
        with Diagram(
            "Container Diagram - DQN Evaluation Pipeline",
            direction="TB",
            graph_attr=graph_attr,
            show=False,
            filename=str(output_file),
            outformat="png",
        ):
            researcher = Person(
                name="Researcher",
                description="Runs evaluation scripts, reviews PNG outputs, "
                "writes results into dissertation",
            )

            visualization = System(
                name="Figures (PNG/PDF)",
                description="dqn_evaluation_results/ subdirectories: "
                "baseline_results/, ablation_results/, "
                "sweep_fairness_results/, multi_seed_results/, etc.",
                external=True,
            )

            with SystemBoundary("DQN Evaluation Pipeline"):
                model_store = Database(
                    name="Trained Models",
                    technology=".zip (SB3 format)",
                    description="models/dqn_domain_rand/dqn_final.zip — "
                    "main model (domain rand + curriculum).\n"
                    "models/dqn_no_dr/ — ablation A4 control.",
                )

                evaluate_single = Container(
                    name="evaluate_dqn.py",
                    technology="Python, Stable-Baselines3",
                    description="Single-seed evaluation of dqn_final.zip "
                    "on one (grid, sensor) configuration. "
                    "Reports reward, bytes, fairness, coverage.",
                )

                compare_script = Container(
                    name="compare_agents.py",
                    technology="Python",
                    description="Loads DQN model + NearestSensor + MaxThroughput. "
                    "Runs each on same episodes. "
                    "Outputs: final_comparison_graph.png, radar_chart.png, "
                    "per_sensor_bar.png, pareto_scatter.png, etc.",
                )

                fairness_script = Container(
                    name="fairness_sweep.py",
                    technology="Python",
                    description="Runs DQN and both greedy agents across all "
                    "16 conditions (4 grids × 4 sensor counts). "
                    "Plots Jain's fairness index summary matrix.",
                )

                ablation_script = Container(
                    name="ablation_study.py",
                    technology="Python",
                    description="Evaluates A1 (no capture), A2 (instant ADR), "
                    "A3 (no AoI), A4 (no DR) vs full DQN. "
                    "Outputs: fig_ablation_bars.png, fig_ablation_fairness.png.",
                )

                multi_seed = Container(
                    name="Multi-seed Evaluation",
                    technology="Python",
                    description="Runs dqn_final.zip with multiple random seeds "
                    "across sensor counts (10/20/30/40) and "
                    "grid sizes (100/300/500/1000). "
                    "Produces shaded reward/coverage curves.",
                )

                greedy_agents = Container(
                    name="greedy_agents.py",
                    technology="Python",
                    description="NearestSensorGreedy: moves to closest sensor with data.\n"
                    "MaxThroughputGreedy: SF-aware, targets lowest SF "
                    "(highest data rate = SF7 > SF9 > SF11 > SF12).",
                )

                env_eval = Container(
                    name="UAVEnvironment (eval mode)",
                    technology="Python, Gymnasium",
                    description="Same environment used in training. "
                    "Seeded for reproducibility. "
                    "2100 timesteps per episode.",
                )

            # Relationships
            researcher >> Relationship("Runs scripts, reviews outputs") >> evaluate_single
            researcher >> Relationship("Runs scripts, reviews outputs") >> compare_script
            researcher >> Relationship("Runs scripts, reviews outputs") >> fairness_script
            researcher >> Relationship("Runs scripts, reviews outputs") >> ablation_script

            evaluate_single >> Relationship("Loads model") >> model_store
            compare_script >> Relationship("Loads model") >> model_store
            fairness_script >> Relationship("Loads model") >> model_store
            ablation_script >> Relationship("Loads A4 control model") >> model_store
            multi_seed >> Relationship("Loads model") >> model_store

            compare_script >> Relationship("Uses baselines") >> greedy_agents
            fairness_script >> Relationship("Uses baselines") >> greedy_agents

            evaluate_single >> Relationship("Steps through") >> env_eval
            compare_script >> Relationship("Steps through") >> env_eval
            fairness_script >> Relationship("Steps through") >> env_eval
            ablation_script >> Relationship("Steps through") >> env_eval
            multi_seed >> Relationship("Steps through") >> env_eval

            evaluate_single >> Relationship("Writes PNGs") >> visualization
            compare_script >> Relationship("Writes PNGs") >> visualization
            fairness_script >> Relationship("Writes PNGs") >> visualization
            ablation_script >> Relationship("Writes PNGs") >> visualization
            multi_seed >> Relationship("Writes PNGs") >> visualization

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

    logger.info("DQN evaluation pipeline container diagram complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Running DQN evaluation pipeline container diagram script...")
    create_diagram()
    logger.info("Script finished!")
