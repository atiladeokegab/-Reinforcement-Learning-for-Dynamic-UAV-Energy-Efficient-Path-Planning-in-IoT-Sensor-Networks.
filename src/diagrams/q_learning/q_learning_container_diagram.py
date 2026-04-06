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
    logger.info("Creating UAV DQN Container diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dqn_container_diagram"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {
        "splines": "spline",
    }

    try:
        with Diagram(
            "Container Diagram - UAV DQN Training & Evaluation System",
            direction="TB",
            graph_attr=graph_attr,
            show=False,
            filename=str(output_file),
            outformat="png",
        ):
            # External Actors
            researcher = Person(
                name="Researcher / Student",
                description="Runs dqn.py training, evaluate_dqn.py, "
                "ablation_study.py, compare_agents.py, fairness_sweep.py",
            )

            # External Systems
            visualization = System(
                name="Visualization Tools",
                description="Matplotlib, Seaborn — produces PNG figures "
                "in dqn_evaluation_results/",
                external=True,
            )

            gpu_system = System(
                name="GPU Compute",
                description="CUDA 12.1, PyTorch 2.5.1+cu121 — "
                "accelerates DQN policy network training",
                external=True,
            )

            # Main System Containers
            with SystemBoundary("UAV DQN Training & Evaluation System"):
                # Training entry point
                training_script = Container(
                    name="Training Script (dqn.py)",
                    technology="Python, Stable-Baselines3",
                    description="Domain randomisation + curriculum learning. "
                    "4 parallel envs (DummyVecEnv). "
                    "2M timesteps. Output: dqn_final.zip",
                )

                # DQN Agent
                dqn_agent = Container(
                    name="DQN Agent (SB3)",
                    technology="Python, PyTorch, Stable-Baselines3",
                    description="MlpPolicy [512, 512, 256], Huber loss, "
                    "Adam optimiser, target network soft-update, "
                    "epsilon-greedy exploration",
                )

                # Environment
                environment = Container(
                    name="UAVEnvironment (uav_env.py)",
                    technology="Python, Gymnasium",
                    description="2D grid (100–1000 units, ~10m/unit), "
                    "5 discrete actions (N/S/E/W/Hover), "
                    "2100 timesteps per episode (~7 min flight). "
                    "Zero-padded obs for up to 50 sensors.",
                )

                # UAV model
                uav_container = Container(
                    name="UAV (uav.py)",
                    technology="Python",
                    description="DJI TB60-inspired: 274 Wh, 100m altitude, "
                    "10 m/s speed, 500W move / 700W hover power",
                )

                # IoT sensors
                iot_container = Container(
                    name="IoT Sensors (iot_sensors.py)",
                    technology="Python",
                    description="LoRa SF7–SF12, 1000-byte buffer, "
                    "1% EU duty cycle, EMA-ADR (λ=0.1), "
                    "Two-Ray path loss + N(0, 4dB) shadowing",
                )

                # Reward
                reward_container = Container(
                    name="RewardFunction (reward_function.py)",
                    technology="Python",
                    description="+100/byte, +5000 new sensor, +200 multi-sensor, "
                    "+1000 urgency reduction, -2 revisit, "
                    "-50 boundary, -500 starvation, -2000 unvisited",
                )

                # Evaluation scripts
                evaluation = Container(
                    name="Evaluation Suite",
                    technology="Python",
                    description="evaluate_dqn.py (single-seed), "
                    "ablation_study.py (A1–A4), "
                    "compare_agents.py (DQN vs greedy baselines), "
                    "fairness_sweep.py (multi-condition Jain's index)",
                )

                # Greedy baselines
                greedy_baselines = Container(
                    name="Greedy Baselines (greedy_agents.py)",
                    technology="Python",
                    description="NearestSensorGreedy: moves to nearest sensor "
                    "with data. MaxThroughputGreedy: SF-aware, "
                    "prioritises lowest SF (highest data rate).",
                )

                # Ablation control model
                ablation_control = Container(
                    name="Ablation Control (train_ablation_a4.py)",
                    technology="Python, Stable-Baselines3",
                    description="Fixed env: 500×500, N=20, no domain randomisation. "
                    "Output: models/dqn_no_dr/",
                )

                # Model storage
                model_storage = Database(
                    name="Model Storage",
                    technology=".zip (SB3 format)",
                    description="models/dqn_domain_rand/dqn_final.zip — "
                    "main trained model. "
                    "models/dqn_no_dr/ — ablation A4 control model.",
                )

            # External Relationships
            researcher >> Relationship("Runs training") >> training_script
            researcher >> Relationship("Runs evaluation & ablation scripts") >> evaluation
            researcher >> Relationship("Views result PNGs") >> visualization

            gpu_system >> Relationship("Accelerates policy network updates") >> dqn_agent

            # Internal relationships
            training_script >> Relationship("Creates and wraps (DummyVecEnv × 4)") >> environment
            training_script >> Relationship("Configures & trains") >> dqn_agent
            training_script >> Relationship("Saves checkpoints + final model") >> model_storage

            dqn_agent >> Relationship("Observes state vector") >> environment
            dqn_agent >> Relationship("Selects action (0–4)") >> uav_container
            dqn_agent >> Relationship("Receives reward + next state") >> reward_container

            environment >> Relationship("Contains") >> uav_container
            environment >> Relationship("Contains (10–40 sensors)") >> iot_container
            environment >> Relationship("Calls") >> reward_container

            uav_container >> Relationship("Moves, hovers, collects") >> iot_container

            evaluation >> Relationship("Loads model") >> model_storage
            evaluation >> Relationship("Benchmarks against") >> greedy_baselines
            evaluation >> Relationship("Exports PNGs") >> visualization

            ablation_control >> Relationship("Trains no-DR model") >> model_storage

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

    logger.info("DQN container diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Running DQN container diagram script...")
    create_diagram()
    logger.info("Script finished!")
