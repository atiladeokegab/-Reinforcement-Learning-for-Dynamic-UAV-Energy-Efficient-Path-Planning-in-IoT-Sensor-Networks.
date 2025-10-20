from pathlib import Path
import logging
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.framework import React
from diagrams.programming.language import Python
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.custom import Custom

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV Q-Learning Component diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "q_learning_component_diagram"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {
        "splines": "spline",
        "pad": "0.5",
        "nodesep": "0.8",
        "ranksep": "1.0"
    }

    try:
        with Diagram(
                "Component Diagram - Q-Learning Simulation Components",
                direction="TB",
                graph_attr=graph_attr,
                show=False,
                filename=str(output_file),
                outformat="png"
        ):

            with Cluster("Q-Learning Agent Component"):
                q_table = Python("Q-Table\n(NumPy Array)")
                policy = Python("Policy Manager\n(Epsilon-Greedy)")
                action_selector = Python("Action Selector")
                value_updater = Python("Q-Value Updater\n(Bellman Equation)")

                policy >> Edge(label="selects action") >> action_selector
                action_selector >> Edge(label="updates") >> value_updater
                value_updater >> Edge(label="modifies") >> q_table

            with Cluster("Environment Component"):
                state_manager = Python("State Manager")
                grid_world = Python("Grid World\n(2D/3D Space)")
                transition_model = Python("Transition Model")

                state_manager >> Edge(label="manages") >> grid_world
                grid_world >> Edge(label="provides dynamics") >> transition_model

            with Cluster("Simulated UAV Component"):
                uav_physics = Python("UAV Physics Engine")
                position_tracker = Python("Position Tracker")
                battery_model = Python("Battery Model")
                movement_controller = Python("Movement Controller")

                movement_controller >> Edge(label="updates") >> position_tracker
                movement_controller >> Edge(label="consumes") >> battery_model
                uav_physics >> Edge(label="validates") >> movement_controller

            with Cluster("Simulated IoT Network Component"):
                sensor_generator = Python("Sensor Generator")
                coverage_tracker = Python("Coverage Tracker")
                data_simulator = Python("Data Simulator")
                lora_range_model = Python("LoRa Range Model")

                sensor_generator >> Edge(label="creates") >> data_simulator
                position_tracker >> Edge(label="distance calc") >> lora_range_model
                lora_range_model >> Edge(label="determines coverage") >> coverage_tracker

            with Cluster("Reward Component"):
                reward_function = Python("Reward Function")
                coverage_evaluator = Python("Coverage Evaluator")
                efficiency_calculator = Python("Efficiency Calculator")

                coverage_tracker >> Edge(label="provides metrics") >> coverage_evaluator
                battery_model >> Edge(label="efficiency data") >> efficiency_calculator
                coverage_evaluator >> Edge(label="contributes") >> reward_function
                efficiency_calculator >> Edge(label="contributes") >> reward_function

            with Cluster("Training Controller Component"):
                episode_manager = Python("Episode Manager")
                convergence_checker = Python("Convergence Checker")
                hyperparameter_manager = Python("Hyperparameter Manager")

                hyperparameter_manager >> Edge(label="configures") >> policy
                episode_manager >> Edge(label="monitors") >> convergence_checker

            with Cluster("Data Management Component"):
                logger_comp = Python("Metrics Logger")
                results_exporter = Python("Results Exporter")
                checkpoint_manager = Python("Checkpoint Manager")

                q_table >> Edge(label="saves") >> checkpoint_manager
                logger_comp >> Edge(label="exports") >> results_exporter

            # Component interactions
            action_selector >> Edge(label="action", style="bold") >> movement_controller
            transition_model >> Edge(label="next state", style="bold") >> state_manager
            state_manager >> Edge(label="observes", style="bold") >> q_table
            reward_function >> Edge(label="reward signal", style="bold") >> value_updater
            episode_manager >> Edge(label="resets", style="bold") >> state_manager
            coverage_tracker >> Edge(label="metrics", style="bold") >> logger_comp

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

    logger.info("Component diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Running Q-Learning component diagram script...")
    create_diagram()
    logger.info("Script finished!")