from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import (
    Person, Container, Database, System, SystemBoundary, Relationship,
)

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def create_diagram():
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "q_learning_container_diagram"

    graph_attr = {"splines": "spline"}

    with Diagram(
        "Container Diagram — UAV DQN Training & Evaluation System",
        direction="TB",
        graph_attr=graph_attr,
        show=False,
        filename=str(output_file),
        outformat="png",
    ):
        researcher = Person(
            name="Researcher / Student",
            description="Runs dqn.py training, ablation_study.py, compare_agents.py, "
            "sweep_eval.py, sf_sweep_analysis.py, relational_vs_tsp_sweep.py.",
        )

        viz = System(
            name="Visualisation Tools",
            description="Matplotlib + Seaborn + Pandas — produces PNG/PDF figures "
            "in dqn_evaluation_results/baseline_results/.",
            external=True,
        )

        gpu = System(
            name="GPU Compute",
            description="CUDA 12.1, PyTorch 2.5.1+cu121 — "
            "accelerates DQN Q-network and optional attention extractor.",
            external=True,
        )

        with SystemBoundary("UAV DQN Training & Evaluation System"):
            training = Container(
                name="Training Script (dqn.py)",
                technology="Python, Stable-Baselines3",
                description="Domain randomisation (4 grids × 4 sensor counts). "
                "Competence-based curriculum (5 stages, greedy benchmark gate). "
                "4 parallel workers via DummyVecEnv + VecFrameStack(k=4). "
                "3M timesteps. Output: models/dqn_v3/dqn_final.zip.",
            )

            dqn = Container(
                name="DQN Agent (SB3)",
                technology="Python, PyTorch, Stable-Baselines3",
                description="MlpPolicy [512, 512, 256] ReLU, Huber loss, Adam (lr=3e-4). "
                "Replay buffer 150k, batch 256, γ=0.99. "
                "Optional: UAVAttentionExtractor (gnn_extractor.py) — "
                "cross-attention over 50 padded sensor slots.",
            )

            env = Container(
                name="UAVEnvironment (uav_env.py + DomainRandEnv)",
                technology="Python, Gymnasium",
                description="2D grid (100–500 units, ~10 m/unit). "
                "5 discrete actions: N/S/E/W/Hover. "
                "2100 timesteps per episode (~7 min flight). "
                "Zero-padded observation for up to 50 sensors. "
                "Navigation fix: rejection-sampled distant start.",
            )

            uav = Container(
                name="UAV (uav.py)",
                technology="Python",
                description="DJI TB60-inspired: 274 Wh, 100 m altitude, 10 m/s speed. "
                "Move: 500 W (0.139 Wh/step). Hover: 700 W (0.194 Wh/step). "
                "Hover costs MORE than movement (rotary-wing aerodynamics).",
            )

            sensors = Container(
                name="IoT Sensors (iot_sensors.py)",
                technology="Python",
                description="LoRa SF7–SF12, 1000-byte buffer, "
                "10% effective tx probability (duty cycle × link success). "
                "EMA-ADR (λ=0.1), Two-Ray path loss + N(0, 4 dB) shadowing. "
                "Capture effect: 6 dB threshold resolves same-SF collisions.",
            )

            reward = Container(
                name="RewardFunction (reward_function.py)",
                technology="Python",
                description="+100/byte × urgency, +5000 new sensor, "
                "+1000 urgency reduction. "
                "−1000 × buffer variance (starvation), −1000 per sensor CR<20%, "
                "−2 revisit, −50 boundary, −5000 unvisited at end.",
            )

            eval_suite = Container(
                name="Evaluation Suite",
                technology="Python",
                description="compare_agents.py: DQN vs NearestSensor vs MaxThroughput "
                "vs Relational RL vs TSP Oracle. "
                "ablation_study.py: A1–A4 conditions, n=20 seeds, Welch's t-test. "
                "sweep_eval.py / sf_sweep_analysis.py: SF distribution & fairness sweep. "
                "relational_vs_tsp_sweep.py: scalability benchmark.",
            )

            relational = Container(
                name="Relational RL Policy (experiments/relational_policy/)",
                technology="Python, RLlib / SB3",
                description="Permutation-invariant sensor attention. "
                "Comparison baseline for DQN generalisation analysis. "
                "Checkpoints: models/relational_rl/.",
            )

            model_db = Database(
                name="Model Storage",
                technology=".zip (SB3 format)",
                description="models/dqn_v3/dqn_final.zip — main trained model. "
                "models/dqn_v3/best_model/ — EvalCallback best. "
                "models/relational_rl/ — relational RL checkpoints.",
            )

        # External
        researcher >> Relationship("Runs training (3M timesteps)") >> training
        researcher >> Relationship("Runs eval, ablation, sweep scripts") >> eval_suite
        researcher >> Relationship("Views result PNGs") >> viz
        gpu >> Relationship("Accelerates Q-network updates") >> dqn

        # Training flow
        training >> Relationship("Creates & wraps (DummyVecEnv × 4 + FrameStack)") >> env
        training >> Relationship("Configures & trains") >> dqn
        training >> Relationship("Saves checkpoints + final model") >> model_db

        # DQN ↔ env
        dqn >> Relationship("Observes 612-dim stacked state") >> env
        dqn >> Relationship("Selects action (0–4)") >> uav
        dqn >> Relationship("Receives reward signal") >> reward

        # Env internals
        env >> Relationship("Contains") >> uav
        env >> Relationship("Contains 10–40 sensors") >> sensors
        env >> Relationship("Calls per step") >> reward
        uav >> Relationship("Moves / hovers / collects") >> sensors

        # Eval
        eval_suite >> Relationship("Loads model") >> model_db
        eval_suite >> Relationship("Benchmarks DQN against") >> relational
        eval_suite >> Relationship("Exports figures") >> viz
        relational >> Relationship("Loads checkpoint") >> model_db

    logger.info("Container diagram saved: %s.png", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    create_diagram()
