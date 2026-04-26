from pathlib import Path
import logging
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def create_diagram():
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "q_learning_component_diagram"

    graph_attr = {"splines": "spline", "pad": "0.5", "nodesep": "0.8", "ranksep": "1.0"}

    with Diagram(
        "Component Diagram — DQN UAV IoT Data Collection",
        direction="TB",
        graph_attr=graph_attr,
        show=False,
        filename=str(output_file),
        outformat="png",
    ):
        with Cluster("DQN Agent (Stable-Baselines3)"):
            policy = Python(
                "Policy Network\n"
                "MlpPolicy [512, 512, 256] ReLU\n"
                "OR UAVAttentionExtractor\n"
                "(cross-attention, 50 sensor slots,\n"
                " embed_dim=64, 4 heads)"
            )
            exploration = Python("Exploration\n(ε-greedy, ε: 1.0→0.03\nover 25% of timesteps)")
            action_sel = Python("Action Selector\n5 discrete: N / S / E / W / Hover")
            dqn_update = Python(
                "DQN Update\nHuber loss + Adam (lr=3e-4)\nTarget net (sync every 5k steps)\nBatch=256, γ=0.99"
            )
            replay = Python("Replay Buffer\n150k transitions\n(~0.7 GB, 612-dim obs)")

            exploration >> Edge(label="selects") >> action_sel
            action_sel >> Edge(label="stores transition") >> replay
            replay >> Edge(label="mini-batch (256)") >> dqn_update
            dqn_update >> Edge(label="updates weights") >> policy

        with Cluster("Domain-Randomised Environment (Gymnasium)"):
            state_mgr = Python("State Manager\n(UAVEnvironment + DomainRandEnv)")
            grid = Python("Grid World\n100–500 unit grids\n(~10 m per unit)")
            domain_rand = Python(
                "Domain Randomiser\n"
                "4 grids × 4 sensor counts = 16 conditions\n"
                "Layout: uniform random each episode"
            )
            curriculum = Python(
                "Competence-Based Curriculum\n"
                "5 stages (100→500 units)\n"
                "Greedy benchmark gate:\n"
                "NDR ≥ greedy+5%, Jain≥greedy+0.05"
            )
            frame_stack = Python("VecFrameStack\nk=4, 4 parallel workers\n(DummyVecEnv)")

            domain_rand >> Edge(label="samples grid") >> grid
            curriculum >> Edge(label="unlocks stages") >> domain_rand
            state_mgr >> Edge(label="manages") >> grid
            frame_stack >> Edge(label="stacks obs") >> state_mgr

        with Cluster("UAV Physics Model"):
            uav = Python("UAV (uav.py)\n100m altitude, 10 m/s\n274 Wh (DJI TB60-inspired)")
            pos = Python("Position (x, y)\nnormalised to [0,1]")
            battery = Python("Battery Model\nMove: 500 W → 0.139 Wh/step\nHover: 700 W → 0.194 Wh/step")

            uav >> Edge(label="updates") >> pos
            uav >> Edge(label="drains") >> battery

        with Cluster("LoRa IoT Sensor Network"):
            sensor = Python(
                "IoTSensor (iot_sensors.py)\n"
                "SF7–SF12, 1000-byte buffer\n"
                "10% effective tx probability\n"
                "(duty cycle × link success)"
            )
            adr = Python("EMA-ADR (λ=0.1)\nRSSI → SF mapping\n(4-threshold RSSI bands)")
            path_loss = Python(
                "Two-Ray Path Loss\n"
                "+ Gaussian Shadowing N(0, 4 dB)\n"
                "Capture Effect: 6 dB threshold"
            )

            sensor >> Edge(label="uses") >> adr
            adr >> Edge(label="computes") >> path_loss

        with Cluster("Reward Function"):
            rf = Python("RewardFunction\n(reward_function.py)")
            data_r = Python("+100/byte × urgency\n+5000 new sensor\n+1000 urgency reduced")
            fair_r = Python("Fairness\n−1000 × buffer variance\n−1000 per sensor CR<20%")
            penalty = Python("Penalties\n−2 revisit, −50 boundary\n−5000 unvisited at end\n−0.5 step, −5 hover")

            rf >> Edge(label="computes") >> data_r
            rf >> Edge(label="computes") >> fair_r
            rf >> Edge(label="applies") >> penalty

        with Cluster("Evaluation Suite"):
            compare = Python("compare_agents.py\nDQN vs NearestSensor\nvs MaxThroughput\nvs Relational RL\nvs TSP Oracle")
            ablation = Python("ablation_study.py\nA1: no capture effect\nA2: instant ADR (λ=1)\nA3: no AoI obs\nA4: no domain rand")
            sweep = Python("sweep_eval.py / sf_sweep_analysis.py\nfairness sweep, SF distribution\nacross all 16 conditions")

        # Cross-component
        action_sel >> Edge(label="action (0–4)", style="bold") >> uav
        state_mgr >> Edge(label="obs vector (612-dim stacked)", style="bold") >> policy
        rf >> Edge(label="reward signal", style="bold") >> dqn_update
        battery >> Edge(label="battery %") >> state_mgr
        path_loss >> Edge(label="RSSI → SF") >> sensor
        sensor >> Edge(label="buffer / urgency obs") >> state_mgr

        compare >> Edge(label="evaluates") >> state_mgr
        ablation >> Edge(label="A1–A4 variants") >> state_mgr

    logger.info("Component diagram saved: %s.png", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    create_diagram()
