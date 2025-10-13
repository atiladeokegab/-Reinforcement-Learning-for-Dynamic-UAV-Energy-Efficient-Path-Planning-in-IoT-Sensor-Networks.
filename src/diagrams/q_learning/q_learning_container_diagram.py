from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import Person, Container, Database, System, SystemBoundary, Relationship

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV Q-Learning Simulation Container diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "q_learning_container_diagram"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {
        "splines": "spline",
    }

    try:
        with Diagram(
                "Container Diagram - UAV Q-Learning Simulation (Simulation-Based)",
                direction="TB",
                graph_attr=graph_attr,
                show=False,
                filename=str(output_file),
                outformat="png"
        ):
            # External Actors
            researcher = Person(
                name="Researcher",
                description="Runs experiments and analyzes results"
            )

            # External Systems
            visualization = System(
                name="Visualization Tools",
                description="Matplotlib, Seaborn, Plotly",
                external=True
            )

            file_system = System(
                name="File System Storage",
                description="Logs, models, results",
                external=True
            )

            # Main System Containers
            with SystemBoundary("UAV Q-Learning Simulation System"):
                # Training Controller
                training_controller = Container(
                    name="Training Controller",
                    technology="Python",
                    description="Orchestrates training episodes, manages hyperparameters, tracks convergence"
                )

                # Q-Learning Agent Container
                q_learning_agent = Container(
                    name="Q-Learning Agent",
                    technology="Python, NumPy",
                    description="Implements Q-Learning algorithm: state-action selection, Q-table updates, epsilon-greedy policy"
                )

                # Environment Simulation Container
                environment = Container(
                    name="Simulated Environment",
                    technology="Python, Gymnasium/OpenAI Gym",
                    description="Simulates 2D/3D grid world with IoT sensors, UAV physics, communication ranges"
                )

                # Simulated UAV
                simulated_uav = Container(
                    name="Simulated UAV",
                    technology="Python",
                    description="Virtual UAV with position, battery level, movement constraints, LoRa range simulation"
                )

                # Simulated IoT Network
                simulated_iot = Container(
                    name="Simulated IoT Network",
                    technology="Python",
                    description="Virtual sensor nodes with positions, data generation, coverage tracking"
                )

                # Reward Calculator
                reward_calculator = Container(
                    name="Reward Calculator",
                    technology="Python",
                    description="Computes rewards based on data collected, coverage, battery usage, path efficiency"
                )

                # Metrics & Logger
                metrics_logger = Container(
                    name="Metrics & Logger",
                    technology="Python, Pandas",
                    description="Logs episode rewards, Q-values, trajectories, coverage statistics"
                )

                # Configuration Manager
                config_manager = Container(
                    name="Configuration Manager",
                    technology="Python, YAML/JSON",
                    description="Manages simulation parameters, Q-Learning hyperparameters, experiment configs"
                )

                # Results Database
                results_db = Database(
                    name="Results Storage",
                    technology="SQLite / CSV / HDF5",
                    description="Stores Q-tables, episode history, training metrics, experiment results"
                )

            # External Relationships
            researcher >> Relationship("Configures experiments [Config files]") >> config_manager
            researcher >> Relationship("Starts training runs") >> training_controller
            researcher >> Relationship("Views plots, analyzes results") >> visualization

            file_system >> Relationship("Loads saved Q-tables, checkpoints") >> results_db
            results_db >> Relationship("Saves models, logs") >> file_system

            visualization >> Relationship("Reads metrics for plotting") >> results_db

            # Internal Container Relationships
            config_manager >> Relationship("Provides hyperparameters (alpha, gamma, epsilon)") >> q_learning_agent
            config_manager >> Relationship("Provides environment config (grid, sensors)") >> environment

            training_controller >> Relationship("Initializes episodes") >> environment
            training_controller >> Relationship("Controls training loop") >> q_learning_agent
            training_controller >> Relationship("Monitors convergence") >> metrics_logger

            q_learning_agent >> Relationship("Reads/updates Q-values") >> results_db
            q_learning_agent >> Relationship("Observes state") >> environment
            q_learning_agent >> Relationship("Selects action") >> simulated_uav

            environment >> Relationship("Contains UAV state") >> simulated_uav
            environment >> Relationship("Contains sensor network") >> simulated_iot
            environment >> Relationship("Requests reward") >> reward_calculator

            simulated_uav >> Relationship("Executes movement") >> environment
            simulated_uav >> Relationship("Checks sensor range") >> simulated_iot

            simulated_iot >> Relationship("Provides data availability") >> simulated_uav
            simulated_iot >> Relationship("Reports coverage status") >> reward_calculator

            reward_calculator >> Relationship("Returns reward value") >> q_learning_agent
            reward_calculator >> Relationship("Evaluates coverage, efficiency") >> simulated_iot

            metrics_logger >> Relationship("Logs episode data") >> results_db
            metrics_logger >> Relationship("Tracks Q-value changes") >> q_learning_agent
            metrics_logger >> Relationship("Records trajectories") >> simulated_uav

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

    logger.info("Container diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Running Q-Learning simulation container diagram script...")
    create_diagram()
    logger.info("Script finished!")