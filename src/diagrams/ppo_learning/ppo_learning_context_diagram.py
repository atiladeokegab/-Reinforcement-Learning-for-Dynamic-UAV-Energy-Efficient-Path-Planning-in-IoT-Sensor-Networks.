"""
DQN Training Pipeline - Context Diagram
(This file is in the ppo_learning folder by historical naming only;
the project uses DQN, not PPO.)
"""
from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import Person, System, SystemBoundary, Relationship

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating DQN Training Pipeline Context diagram...")

    output_dir = CURRENT_DIR / "asset" / "diagrams" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dqn_training_pipeline_context"

    graph_attr = {"splines": "spline"}

    try:
        with Diagram(
            "DQN Training Pipeline - Context Diagram",
            direction="LR",
            graph_attr=graph_attr,
            show=False,
            filename=str(output_file),
            outformat="png",
        ):
            researcher = Person(
                name="Researcher",
                description="Runs dqn.py, monitors TensorBoard, "
                "reviews checkpoint .zip files",
            )

            gpu = System(
                name="GPU (CUDA 12.1)",
                description="RTX-class GPU, PyTorch 2.5.1+cu121. "
                "Accelerates MlpPolicy forward/backward passes.",
                external=True,
            )

            checkpoint_store = System(
                name="Checkpoint Storage",
                description="models/dqn_domain_rand/\n"
                "*.zip checkpoints + dqn_final.zip",
                external=True,
            )

            with SystemBoundary("DQN Training Pipeline (dqn.py)"):
                curriculum = System(
                    name="Curriculum Scheduler",
                    description="Stage 0: grids 100–300, N=10–20 (0–150k steps)\n"
                    "Stage 1: + grid 500, N=30 (150k–400k steps)\n"
                    "Stage 2: full dist. incl. 1000×1000, N=40 (400k+ steps)",
                )

                domain_rand = System(
                    name="Domain Randomiser",
                    description="Samples (grid_size, num_sensors) each episode "
                    "from the active curriculum stage. 16 total conditions.",
                )

                vec_envs = System(
                    name="DummyVecEnv (4 workers)",
                    description="4 independent UAVEnvironment instances, "
                    "each with independently randomised conditions. "
                    "Observations zero-padded to MAX_SENSORS=50.",
                )

                dqn_sb3 = System(
                    name="SB3 DQN Agent",
                    description="MlpPolicy [512, 512, 256], Huber loss, "
                    "Adam optimiser, epsilon-greedy, "
                    "replay buffer 100k, batch 256, γ=0.99",
                )

            researcher >> Relationship("Launches training, sets hyperparameters") >> curriculum
            curriculum >> Relationship("Unlocks harder conditions") >> domain_rand
            domain_rand >> Relationship("Configures episode env") >> vec_envs
            vec_envs >> Relationship("(obs, reward, done)") >> dqn_sb3
            dqn_sb3 >> Relationship("action (0–4)") >> vec_envs
            dqn_sb3 >> Relationship("Saves .zip every N steps") >> checkpoint_store
            gpu >> Relationship("Accelerates forward/backward pass") >> dqn_sb3
            researcher >> Relationship("Reviews saved models") >> checkpoint_store

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

    logger.info("DQN training pipeline context diagram complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Running DQN training pipeline context diagram script...")
    create_diagram()
    logger.info("Script finished!")
