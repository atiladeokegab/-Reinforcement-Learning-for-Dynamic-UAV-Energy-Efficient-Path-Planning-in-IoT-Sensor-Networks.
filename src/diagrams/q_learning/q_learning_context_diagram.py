from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import Person, System, SystemBoundary, Relationship

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def create_diagram():
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "q_learning_system_context"

    graph_attr = {"splines": "spline"}

    with Diagram(
        "System Context — UAV DQN IoT Data Collection",
        direction="TB",
        graph_attr=graph_attr,
        show=False,
        filename=str(output_file),
        outformat="png",
    ):
        researcher = Person(
            name="Researcher",
            description="Configures DQN hyperparameters, curriculum stages, "
            "and domain-randomisation conditions. Runs training and evaluation scripts.",
        )

        gpu = System(
            name="GPU Compute",
            description="CUDA 12.1, PyTorch 2.5.1+cu121. "
            "Accelerates DQN policy-network updates on RTX-class GPU.",
            external=True,
        )

        viz = System(
            name="Visualisation & Analysis",
            description="Matplotlib + Seaborn — generates trajectory plots, "
            "fairness sweeps, ablation bar charts, and scalability figures in "
            "dqn_evaluation_results/baseline_results/.",
            external=True,
        )

        storage = System(
            name="Model & Results Storage",
            description="models/dqn_v3/dqn_final.zip (SB3 format), "
            "training_config.json, condition_summary.json, graduation_log.json. "
            "Evaluation PNGs in dqn_evaluation_results/baseline_results/.",
            external=True,
        )

        with SystemBoundary("UAV DQN Training & Evaluation System"):
            system = System(
                name="DQN UAV Simulation",
                description="SB3 DQN trained across 16 domain-randomised conditions "
                "(4 grid sizes × 4 sensor counts) with competence-based curriculum "
                "(5 stages, greedy benchmark gate). 4 parallel workers, 3M timesteps. "
                "Optional UAVAttentionExtractor (cross-attention over 50 sensor slots).",
            )

        researcher >> Relationship(
            "Sets: grid sizes 100–500 units, sensor counts 10–40, "
            "DQN hyperparams (lr=3e-4, γ=0.99, buffer=150k, batch=256)"
        ) >> system

        researcher >> Relationship(
            "Monitors: NDR, Jain's fairness index, B/Wh efficiency, "
            "curriculum stage, greedy benchmark gap"
        ) >> system

        system >> Relationship(
            "Exports: dqn_final.zip, checkpoint .zip files every 25k steps"
        ) >> storage

        system >> Relationship(
            "Sends: trajectory data, reward curves, ablation results (A1–A4), "
            "SF sweep analysis, sim-to-real robustness figures"
        ) >> viz

        gpu >> Relationship("Accelerates Q-network forward/backward passes") >> system

        researcher >> Relationship(
            "Analyses: DQN vs Relational RL vs TSP Oracle vs greedy baselines; "
            "ablation conditions A1–A4; cross-layout generalisation"
        ) >> viz

        researcher >> Relationship("Loads: saved .zip models for evaluation") >> storage

    logger.info("System context diagram saved: %s.png", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    create_diagram()
