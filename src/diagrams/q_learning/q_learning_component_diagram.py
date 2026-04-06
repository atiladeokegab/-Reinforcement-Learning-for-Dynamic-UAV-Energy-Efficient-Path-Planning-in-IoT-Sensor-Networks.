from pathlib import Path
import logging
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV DQN Component diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dqn_component_diagram"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {"splines": "spline", "pad": "0.5", "nodesep": "0.8", "ranksep": "1.0"}

    try:
        with Diagram(
            "Component Diagram - DQN UAV Simulation (Stable-Baselines3)",
            direction="TB",
            graph_attr=graph_attr,
            show=False,
            filename=str(output_file),
            outformat="png",
        ):
            with Cluster("DQN Agent Component (Stable-Baselines3)"):
                mlp_policy = Python("MlpPolicy\n[512, 512, 256]\n(PyTorch)")
                exploration = Python("Exploration\n(Epsilon-Greedy)")
                action_selector = Python("Action Selector\n(5 discrete actions:\nN/S/E/W/Hover)")
                dqn_updater = Python("DQN Update\n(Huber Loss + Adam)\nTarget Network")
                replay_buffer = Python("Replay Buffer\n(Experience Replay)")

                exploration >> Edge(label="selects action") >> action_selector
                action_selector >> Edge(label="stores transition") >> replay_buffer
                replay_buffer >> Edge(label="mini-batch") >> dqn_updater
                dqn_updater >> Edge(label="updates weights") >> mlp_policy

            with Cluster("Environment Component (Gymnasium)"):
                state_manager = Python("State Manager\n(UAVEnvironment)")
                grid_world = Python("Grid World\n(100–1000 unit grid\n~10m per unit)")
                domain_rand = Python("Domain Randomiser\n(4 grids × 4 sensor counts\n= 16 conditions)")
                curriculum = Python("Curriculum Scheduler\n(Stage 0→1→2\nover 2M timesteps)")

                domain_rand >> Edge(label="samples") >> grid_world
                curriculum >> Edge(label="unlocks stages") >> domain_rand
                state_manager >> Edge(label="manages") >> grid_world

            with Cluster("Simulated UAV Component"):
                uav_model = Python("UAV (uav.py)\nTB60: 274 Wh, 100m alt\nspeed=10 m/s")
                position_tracker = Python("Position Tracker\n(x, y) in grid")
                battery_model = Python("Battery Model\nmove=500W, hover=700W")

                uav_model >> Edge(label="updates") >> position_tracker
                uav_model >> Edge(label="consumes") >> battery_model

            with Cluster("Simulated IoT Network Component"):
                sensor_model = Python("IoTSensor (iot_sensors.py)\nSF7–SF12, 1000-byte buffer\n1% duty cycle (EU)")
                adr_model = Python("EMA-ADR (λ=0.1)\nRSSI → SF update")
                path_loss = Python("Two-Ray Path Loss\n+ Gaussian Shadowing\nN(0, 4 dB)")

                sensor_model >> Edge(label="uses") >> adr_model
                adr_model >> Edge(label="computes") >> path_loss

            with Cluster("Reward Component"):
                reward_function = Python("RewardFunction\n(reward_function.py)")
                data_reward = Python("Data Reward\n+100/byte collected\n+5000 new sensor")
                fairness_reward = Python("Fairness / Urgency\n+1000 urgency reduced\n-500 starvation")
                penalty = Python("Penalties\n-2 revisit, -50 boundary\n-2000 unvisited end")

                reward_function >> Edge(label="computes") >> data_reward
                reward_function >> Edge(label="computes") >> fairness_reward
                reward_function >> Edge(label="applies") >> penalty

            with Cluster("Parallel Training Component"):
                vec_env = Python("DummyVecEnv\n(4 parallel envs)")
                checkpoint = Python("CheckpointCallback\nSaves .zip files")
                eval_cb = Python("EvalCallback\n(500×500, N=20)")

                vec_env >> Edge(label="feeds") >> replay_buffer
                dqn_updater >> Edge(label="triggers") >> checkpoint

            # Cross-component interactions
            action_selector >> Edge(label="action (0–4)", style="bold") >> uav_model
            state_manager >> Edge(label="obs vector", style="bold") >> mlp_policy
            reward_function >> Edge(label="reward signal", style="bold") >> dqn_updater
            battery_model >> Edge(label="battery %") >> state_manager
            path_loss >> Edge(label="RSSI → SF") >> sensor_model
            sensor_model >> Edge(label="AoI / urgency obs") >> state_manager

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

    logger.info("DQN component diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Running DQN component diagram script...")
    create_diagram()
    logger.info("Script finished!")
