from pathlib import Path
import logging
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV DQN Code Structure diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dqn_code_diagram"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {"splines": "ortho", "pad": "0.5", "nodesep": "0.6", "ranksep": "1.2"}

    try:
        with Diagram(
            "Code Diagram - DQN UAV Project Class/Module Structure",
            direction="TB",
            graph_attr=graph_attr,
            show=False,
            filename=str(output_file),
            outformat="png",
        ):
            with Cluster("src/agents/dqn/"):
                dqn_train = Python(
                    "dqn.py\n[Training Script]\n"
                    "SB3 DQN(MlpPolicy)\n"
                    "policy_kwargs=[512,512,256]\n"
                    "domain randomisation\n"
                    "curriculum: 3 stages\n"
                    "4× DummyVecEnv\n"
                    "2M timesteps\n"
                    "→ dqn_final.zip"
                )
                evaluate = Python(
                    "evaluate_dqn.py\n[Single-seed Eval]\n"
                    "loads dqn_final.zip\n"
                    "runs one test episode"
                )
                ablation_a4 = Python(
                    "train_ablation_a4.py\n[Ablation Control]\n"
                    "fixed 500×500, N=20\n"
                    "no domain randomisation\n"
                    "→ models/dqn_no_dr/"
                )

            with Cluster("src/agents/dqn/dqn_evaluation_results/"):
                ablation = Python(
                    "ablation_study.py\n"
                    "A1: no capture effect\n"
                    "A2: instant ADR (λ=1)\n"
                    "A3: no AoI observation\n"
                    "A4: no domain rand"
                )
                compare = Python(
                    "compare_agents.py\n"
                    "DQN vs NearestSensor\n"
                    "vs MaxThroughput"
                )
                fairness = Python(
                    "fairness_sweep.py\n"
                    "Jain's fairness index\n"
                    "across all 16 conditions"
                )
                greedy = Python(
                    "greedy_agents.py\n"
                    "NearestSensorGreedy\n"
                    "MaxThroughputGreedy\n"
                    "(SF-aware)"
                )

            with Cluster("src/environment/"):
                env = Python(
                    "uav_env.py\n[UAVEnvironment]\n"
                    "+ grid_size: Tuple[int,int]\n"
                    "+ sensors: List[IoTSensor]\n"
                    "+ uav: UAV\n"
                    "+ observation_space (Box)\n"
                    "+ action_space (Discrete 5)\n"
                    "——\n"
                    "+ reset() → obs\n"
                    "+ step(action) → (obs,r,done,info)\n"
                    "+ _get_observation()\n"
                    "+ render()"
                )

                uav_class = Python(
                    "uav.py\n[UAV]\n"
                    "+ position: np.ndarray\n"
                    "+ battery: float  # Wh\n"
                    "+ max_battery=274.0  # Wh\n"
                    "+ speed=10.0  # m/s\n"
                    "+ altitude=100.0  # m\n"
                    "+ power_move=500.0  # W\n"
                    "+ power_hover=700.0  # W\n"
                    "——\n"
                    "+ move(direction, grid_size)\n"
                    "+ hover(duration)\n"
                    "+ is_alive() → bool\n"
                    "+ reset()"
                )

                sensor_class = Python(
                    "iot_sensors.py\n[IoTSensor]\n"
                    "+ position: np.ndarray\n"
                    "+ spreading_factor: int  # SF7–12\n"
                    "+ data_buffer: float  # bytes\n"
                    "+ max_buffer_size=1000.0\n"
                    "+ adr_lambda=0.1  # EMA\n"
                    "+ shadowing_std_db=4.0  # N(0,4)\n"
                    "+ duty_cycle=10.0  # 1% EU\n"
                    "——\n"
                    "+ compute_rssi(uav_pos)\n"
                    "+ update_adr(rssi)\n"
                    "+ is_in_range(uav_pos)\n"
                    "+ collect_data()\n"
                    "+ generate_data()"
                )

            with Cluster("src/rewards/"):
                reward_func = Python(
                    "reward_function.py\n[RewardFunction]\n"
                    "——\n"
                    "+100 per byte collected\n"
                    "+5000 new sensor visited\n"
                    "+200 multi-sensor bonus\n"
                    "+1000 urgency reduction\n"
                    "-2 revisit penalty\n"
                    "-50 boundary hit\n"
                    "-500 starvation\n"
                    "-2000 unvisited at end"
                )

            # Relationships — training flow
            dqn_train >> Edge(label="wraps (×4)", style="dashed") >> env
            dqn_train >> Edge(label="trains SB3 DQN") >> evaluate
            dqn_train >> Edge(label="uses") >> ablation_a4

            evaluate >> Edge(label="loads model, runs") >> env
            ablation >> Edge(label="loads A1–A4 models, runs") >> env
            compare >> Edge(label="benchmarks") >> greedy
            compare >> Edge(label="evaluates") >> env
            fairness >> Edge(label="sweeps conditions") >> env

            # Environment structure
            env >> Edge(label="contains", style="solid") >> uav_class
            env >> Edge(label="contains (10–40)", style="solid") >> sensor_class
            env >> Edge(label="calls per step", style="dashed") >> reward_func

            uav_class >> Edge(label="moves toward", style="dashed") >> sensor_class
            reward_func >> Edge(label="reads buffer/urgency") >> sensor_class
            reward_func >> Edge(label="reads battery") >> uav_class

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

    logger.info("DQN code diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Running DQN code structure diagram script...")
    create_diagram()
    logger.info("Script finished!")
