from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import Person, System, SystemBoundary, Relationship

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV Q-Learning Simulation Context diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "q_learning_system_context"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {
        "splines": "spline",
    }

    try:
        with Diagram(
                "System Context - UAV Q-Learning Simulation (Simulation-Based)",
                direction="TB",
                graph_attr=graph_attr,
                show=False,
                filename=str(output_file),
                outformat="png"
        ):
            # External Actors
            researcher = Person(
                name="Researcher",
                description="Configures simulation parameters, Q-Learning hyperparameters, and analyzes results"
            )

            # External Systems (for data export/visualization)
            visualization_tools = System(
                name="Visualization & Analysis Tools",
                description="Matplotlib, Pandas for plotting trajectories, rewards, Q-values",
                external=True
            )

            results_storage = System(
                name="Results Repository",
                description="File system or cloud storage for simulation logs and trained models",
                external=True
            )

            # Main System
            with SystemBoundary("UAV Q-Learning Simulation System"):
                simulation_system = System(
                    name="Q-Learning UAV Simulation",
                    description="Simulated environment for training Q-Learning agent to optimize IoT data collection paths"
                )

            # Relationships
            researcher >> Relationship(
                "Configures: grid size, sensor positions, Q-Learning params (alpha, gamma, epsilon)") >> simulation_system
            researcher >> Relationship("Monitors: training progress, episode rewards, convergence") >> simulation_system

            simulation_system >> Relationship("Exports: Q-table, training metrics, episode logs") >> results_storage
            simulation_system >> Relationship(
                "Sends: trajectory data, reward curves, coverage maps") >> visualization_tools

            researcher >> Relationship(
                "Analyzes: performance metrics, learning curves, policy effectiveness") >> visualization_tools
            researcher >> Relationship("Reviews: saved models, experiment logs") >> results_storage

            logger.info("✓ Diagram components created")

        # Verify file was created
        png_file = output_file.with_suffix('.png')
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

    logger.info("Context diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Running Q-Learning simulation context diagram script...")
    create_diagram()
    logger.info("Script finished!")