from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import Person, System, SystemBoundary, Relationship

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV DQN Simulation Context diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dqn_system_context"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {
        "splines": "spline",
    }

    try:
        with Diagram(
            "System Context - UAV DQN Simulation (Stable-Baselines3)",
            direction="TB",
            graph_attr=graph_attr,
            show=False,
            filename=str(output_file),
            outformat="png",
        ):
            # External Actors
            researcher = Person(
                name="Researcher",
                description="Configures simulation parameters, DQN hyperparameters "
                "(learning_rate, gamma, batch_size, buffer_size), and analyzes results",
            )

            # External Systems (for data export/visualization)
            visualization_tools = System(
                name="Visualization & Analysis Tools",
                description="Matplotlib, Seaborn, Pandas for plotting trajectories, "
                "rewards, fairness curves, ablation bars",
                external=True,
            )

            results_storage = System(
                name="Results Repository",
                description="models/dqn_domain_rand/dqn_final.zip (SB3 format), "
                "dqn_evaluation_results/ PNG figures",
                external=True,
            )

            # Main System
            with SystemBoundary("UAV DQN Training & Evaluation System"):
                simulation_system = System(
                    name="DQN UAV Simulation",
                    description="Trains SB3 DQN (MlpPolicy [512,512,256]) across "
                    "16 domain-randomised conditions (4 grids × 4 sensor counts) "
                    "with curriculum learning. 4 parallel envs, 2M timesteps.",
                )

            # Relationships
            (
                researcher
                >> Relationship(
                    "Configures: grid sizes (100–1000), sensor counts (10–40), "
                    "DQN hyperparameters, curriculum thresholds"
                )
                >> simulation_system
            )
            (
                researcher
                >> Relationship(
                    "Monitors: episode rewards, Jain's fairness index, coverage, convergence"
                )
                >> simulation_system
            )

            (
                simulation_system
                >> Relationship(
                    "Exports: dqn_final.zip (trained model), checkpoint .zip files"
                )
                >> results_storage
            )
            (
                simulation_system
                >> Relationship(
                    "Sends: trajectory data, reward curves, ablation results, "
                    "fairness sweep PNGs"
                )
                >> visualization_tools
            )

            (
                researcher
                >> Relationship(
                    "Analyzes: DQN vs greedy baselines, ablation study (A1–A4), "
                    "cross-layout generalization"
                )
                >> visualization_tools
            )
            (
                researcher
                >> Relationship("Reviews: saved .zip models, evaluation PNGs")
                >> results_storage
            )

            logger.info("✓ Diagram components created")

        # Verify file was created
        png_file = output_file.with_suffix(".png")
        if png_file.exists():
            file_size = png_file.stat().st_size
            logger.info(f"✓ Diagram saved: {png_file}")
            logger.info(f"  File size: {file_size:,} bytes")
        else:
            logger.warning("⚠ Output file not found")

    except Exception as e:
        logger.error(f"✗ Error creating diagram: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise

    logger.info("DQN context diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Running DQN simulation context diagram script...")
    create_diagram()
    logger.info("Script finished!")
