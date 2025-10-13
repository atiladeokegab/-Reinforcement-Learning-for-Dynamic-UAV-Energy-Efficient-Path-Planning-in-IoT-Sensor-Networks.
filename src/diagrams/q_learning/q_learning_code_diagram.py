from pathlib import Path
import logging
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV Q-Learning Code Structure diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "q_learning_code_diagram"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {
        "splines": "ortho",
        "pad": "0.5",
        "nodesep": "0.6",
        "ranksep": "1.2"
    }

    try:
        with Diagram(
                "Code Diagram - Q-Learning Class Structure",
                direction="TB",
                graph_attr=graph_attr,
                show=False,
                filename=str(output_file),
                outformat="png"
        ):

            with Cluster("agents/"):
                q_agent = Python(
                    "QLearningAgent\n+ q_table: np.ndarray\n+ alpha: float\n+ gamma: float\n+ epsilon: float\n--\n+ select_action(state)\n+ update_q_value()\n+ get_best_action(state)\n+ save_model()\n+ load_model()")

            with Cluster("environment/"):
                env = Python(
                    "UAVEnvironment\n+ grid_size: tuple\n+ sensors: List[Sensor]\n+ uav: UAV\n--\n+ reset()\n+ step(action)\n+ get_state()\n+ is_done()\n+ render()")

                uav_class = Python(
                    "UAV\n+ position: tuple\n+ battery: float\n+ max_battery: float\n+ speed: float\n--\n+ move(direction)\n+ update_battery()\n+ get_position()\n+ is_alive()")

                sensor_class = Python(
                    "Sensor\n+ id: int\n+ position: tuple\n+ data_collected: bool\n+ range: float\n--\n+ is_in_range(uav_pos)\n+ collect_data()\n+ reset()")

            with Cluster("rewards/"):
                reward_func = Python(
                    "RewardFunction\n+ coverage_weight: float\n+ efficiency_weight: float\n--\n+ calculate(state, action)\n+ coverage_reward()\n+ battery_penalty()\n+ collision_penalty()")

            with Cluster("training/"):
                trainer = Python(
                    "Trainer\n+ agent: QLearningAgent\n+ env: UAVEnvironment\n+ episodes: int\n--\n+ train()\n+ run_episode()\n+ evaluate()\n+ save_checkpoint()")

                metrics = Python(
                    "MetricsLogger\n+ episode_rewards: List\n+ q_values: List\n+ coverage: List\n--\n+ log_episode()\n+ log_step()\n+ export_to_csv()\n+ plot_results()")

            with Cluster("config/"):
                config = Python(
                    "Config\n+ grid_size: tuple\n+ num_sensors: int\n+ learning_rate: float\n+ discount_factor: float\n--\n+ load_from_yaml()\n+ validate()\n+ to_dict()")

            with Cluster("utils/"):
                visualizer = Python(
                    "Visualizer\n--\n+ plot_trajectory()\n+ plot_rewards()\n+ plot_q_values()\n+ animate_episode()")

                data_manager = Python(
                    "DataManager\n--\n+ save_results()\n+ load_results()\n+ export_trajectories()\n+ create_checkpoint()")

            # Relationships
            trainer >> Edge(label="uses", style="dashed") >> q_agent
            trainer >> Edge(label="interacts with", style="dashed") >> env
            trainer >> Edge(label="logs to", style="dashed") >> metrics
            trainer >> Edge(label="loads", style="dashed") >> config

            q_agent >> Edge(label="observes", style="bold") >> env
            q_agent >> Edge(label="receives reward from", style="bold") >> reward_func

            env >> Edge(label="contains", style="solid") >> uav_class
            env >> Edge(label="contains", style="solid") >> sensor_class
            env >> Edge(label="uses", style="dashed") >> reward_func

            reward_func >> Edge(label="evaluates", style="dashed") >> sensor_class

            metrics >> Edge(label="saves via", style="dashed") >> data_manager
            metrics >> Edge(label="plots via", style="dashed") >> visualizer

            config >> Edge(label="configures", style="dashed") >> env
            config >> Edge(label="configures", style="dashed") >> q_agent

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

    logger.info("Code diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Running Q-Learning code diagram script...")
    create_diagram()
    logger.info("Script finished!")