from pathlib import Path
import logging
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def create_diagram():
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "q_learning_code_diagram"

    graph_attr = {"splines": "ortho", "pad": "0.5", "nodesep": "0.6", "ranksep": "1.2"}

    with Diagram(
        "Code / Module Diagram — DQN UAV Project",
        direction="TB",
        graph_attr=graph_attr,
        show=False,
        filename=str(output_file),
        outformat="png",
    ):
        with Cluster("src/agents/dqn/"):
            dqn_train = Python(
                "dqn.py  [Training]\n"
                "SB3 DQN — MlpPolicy [512,512,256]\n"
                "OR UAVAttentionExtractor (gnn_extractor.py)\n"
                "domain randomisation (16 conditions)\n"
                "competence-based curriculum (5 stages)\n"
                "greedy benchmark gate, demotion gate\n"
                "4× DummyVecEnv + VecFrameStack(k=4)\n"
                "3M timesteps → models/dqn_v3/dqn_final.zip"
            )
            gnn_ext = Python(
                "gnn_extractor.py\n[UAVAttentionExtractor]\n"
                "temporal UAV encoder (k=10 frames)\n"
                "sensor encoder: Linear(3→64)\n"
                "cross-attention (1 query, 50 keys)\n"
                "ghost + OOR masking\n"
                "fusion → features_dim=128"
            )

        with Cluster("src/agents/dqn/dqn_evaluation_results/"):
            compare = Python(
                "compare_agents.py\n"
                "DQN vs NearestSensor\n"
                "vs MaxThroughput (V2)\n"
                "vs Relational RL\n"
                "vs TSP Oracle\n"
                "n=20 seeds, Welch's t-test"
            )
            ablation = Python(
                "ablation_study.py\n"
                "A1: no capture effect\n"
                "A2: instant ADR (λ=1)\n"
                "A3: no AoI observation\n"
                "A4: no domain randomisation\n"
                "500×500, N=20, n=20 seeds"
            )
            sweep = Python(
                "sweep_eval.py\n"
                "sf_sweep_analysis.py\n"
                "fairness sweep (all 16 conditions)\n"
                "SF distribution / HHI / entropy"
            )
            relational_run = Python(
                "relational_rl_runner.py\n"
                "loads Relational RL checkpoint\n"
                "wraps env for permutation-inv. obs\n"
                "(used by compare, sweep, ablation)"
            )
            greedy = Python(
                "greedy_agents.py\n"
                "NearestSensorGreedy\n"
                "MaxThroughputGreedyV2\n"
                "(SF-aware, v2 tie-breaking)"
            )

        with Cluster("src/environment/"):
            env_cls = Python(
                "uav_env.py  [UAVEnvironment]\n"
                "+ grid_size: Tuple[int,int]\n"
                "+ sensors: List[IoTSensor]\n"
                "+ uav: UAV\n"
                "+ observation_space: Box(3+3N)\n"
                "+ action_space: Discrete(5)\n"
                "——\n"
                "+ reset() → padded obs\n"
                "+ step(action) → (obs,r,done,info)\n"
                "+ _get_observation()  # normalised\n"
                "+ _execute_collect_action()  # capture\n"
                "+ render()  # matplotlib"
            )
            uav_cls = Python(
                "uav.py  [UAV]\n"
                "+ position: np.ndarray\n"
                "+ battery: float  (Wh)\n"
                "+ max_battery=274.0\n"
                "+ power_move=500 W\n"
                "+ power_hover=700 W  (>move!)\n"
                "+ speed=10 m/s, altitude=100 m\n"
                "——\n"
                "+ move(direction, grid_size)\n"
                "+ hover(duration)\n"
                "+ is_alive() → bool\n"
                "+ reset()"
            )
            sensor_cls = Python(
                "iot_sensors.py  [IoTSensor]\n"
                "+ position: np.ndarray\n"
                "+ spreading_factor: int  (SF7–12)\n"
                "+ data_buffer: float  (bytes)\n"
                "+ max_buffer_size=1000.0\n"
                "+ adr_lambda=0.1  (EMA)\n"
                "+ shadowing_std_db=4.0  (N(0,4))\n"
                "+ duty_cycle=10%  (eff. tx prob)\n"
                "——\n"
                "+ get_success_probability(uav_pos)\n"
                "+ update_spreading_factor(uav_pos)\n"
                "+ collect_data(uav_pos, duration)\n"
                "+ step(time_step)"
            )

        with Cluster("src/rewards/"):
            reward_cls = Python(
                "reward_function.py  [RewardFunction]\n"
                "——\n"
                "+100/byte × sensor_urgency\n"
                "+5000 new sensor visited\n"
                "+1000 × urgency_reduced\n"
                "−2 revisit (empty hover)\n"
                "−50 boundary hit\n"
                "−0.5 step,  −5 hover surcharge\n"
                "−1000 × buffer variance (starvation)\n"
                "−1000 per sensor (CR < 20%, terminal)\n"
                "−5000 × unvisited sensors (terminal)"
            )

        with Cluster("src/experiments/relational_policy/"):
            rel_mod = Python(
                "relational_module.py\n"
                "[RelationalModule]\n"
                "sensor embedding + self-attention\n"
                "permutation invariant policy"
            )
            rel_train = Python(
                "train_relational.py\n"
                "SB3/RLlib training loop\n"
                "→ models/relational_rl/"
            )

        # Training flow
        dqn_train >> Edge(label="uses (optional)", style="dashed") >> gnn_ext
        dqn_train >> Edge(label="wraps ×4") >> env_cls
        dqn_train >> Edge(label="benchmarks against") >> greedy

        # Eval flow
        compare >> Edge(label="evaluates") >> env_cls
        compare >> Edge(label="benchmarks") >> greedy
        compare >> Edge(label="uses") >> relational_run
        ablation >> Edge(label="A1–A4 variants") >> env_cls
        sweep >> Edge(label="sweeps conditions") >> env_cls
        sweep >> Edge(label="uses") >> relational_run
        relational_run >> Edge(label="loads") >> rel_mod

        # Env structure
        env_cls >> Edge(label="contains") >> uav_cls
        env_cls >> Edge(label="contains (10–40)") >> sensor_cls
        env_cls >> Edge(label="calls per step") >> reward_cls

        uav_cls >> Edge(label="moves toward") >> sensor_cls
        reward_cls >> Edge(label="reads urgency/buffer") >> sensor_cls
        reward_cls >> Edge(label="reads battery") >> uav_cls

        rel_train >> Edge(label="trains") >> rel_mod

    logger.info("Code diagram saved: %s.png", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    create_diagram()
