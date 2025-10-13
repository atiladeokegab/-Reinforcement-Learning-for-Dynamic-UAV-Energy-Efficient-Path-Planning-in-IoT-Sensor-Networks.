from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import Person, Container, Database, System, SystemBoundary, Relationship

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV Q-Learning Container diagram...")

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
                "Container Diagram - UAV Q-Learning IoT Data Collection",
                direction="TB",
                graph_attr=graph_attr,
                show=False,
                filename=str(output_file),
                outformat="png"
        ):
            # External Actors
            mission_operator = Person(
                name="Mission Operator",
                description="Plans and monitors missions"
            )

            data_analyst = Person(
                name="Data Analyst",
                description="Analyzes results"
            )

            # External Systems
            iot_sensors = System(
                name="IoT Sensor Network",
                description="LoRa sensor nodes",
                external=True
            )

            lorawan_gateway = System(
                name="LoRaWAN Gateway",
                description="Network gateway",
                external=True
            )

            cloud_platform = System(
                name="Cloud Platform",
                description="Data storage and analytics",
                external=True
            )

            # Main System Containers
            with SystemBoundary("UAV Q-Learning Data Collection System"):
                # Flight Control Container
                flight_controller = Container(
                    name="Flight Controller",
                    technology="Python, DroneKit/MAVLink",
                    description="Controls UAV movement, position, and navigation"
                )

                # Q-Learning Agent Container
                q_learning_agent = Container(
                    name="Q-Learning Agent",
                    technology="Python, NumPy, Gymnasium",
                    description="Reinforcement learning agent that learns optimal data collection policy"
                )

                # Environment Simulation Container
                environment = Container(
                    name="Environment Simulator",
                    technology="Python, Gymnasium",
                    description="Simulates IoT network, UAV state space, and reward calculation"
                )

                # LoRa Communication Container
                lora_module = Container(
                    name="LoRa Communication Module",
                    technology="Python, PySerial, LoRa Radio",
                    description="Handles LoRa communication with IoT sensors"
                )

                # Data Collection Container
                data_collector = Container(
                    name="Data Collection Manager",
                    technology="Python",
                    description="Manages data reception, storage, and validation from sensors"
                )

                # Mission Planner Container
                mission_planner = Container(
                    name="Mission Planner",
                    technology="Python, Flask/FastAPI",
                    description="Web interface for mission configuration and monitoring"
                )

                # Local Database
                local_db = Database(
                    name="Local Database",
                    technology="SQLite/PostgreSQL",
                    description="Stores Q-table, mission logs, collected sensor data, trajectories"
                )

            # External Relationships
            mission_operator >> Relationship("Configures mission via web UI [HTTPS]") >> mission_planner
            mission_operator >> Relationship("Views dashboard") >> cloud_platform

            iot_sensors >> Relationship("Transmits data [LoRa 868/915MHz]") >> lora_module

            lorawan_gateway >> Relationship("Network relay [Optional]") >> lora_module

            data_analyst >> Relationship("Analyzes results") >> cloud_platform

            # Internal Container Relationships
            mission_planner >> Relationship("Sends mission config") >> q_learning_agent

            q_learning_agent >> Relationship("Reads/writes Q-values, policy") >> local_db
            q_learning_agent >> Relationship("Gets state, receives reward") >> environment
            q_learning_agent >> Relationship("Selects action (move direction)") >> flight_controller

            environment >> Relationship("Simulates sensor positions, calculates coverage") >> data_collector
            environment >> Relationship("Reads sensor data for reward calculation") >> local_db

            flight_controller >> Relationship("Executes movement commands") >> q_learning_agent
            flight_controller >> Relationship("Logs position, battery, telemetry") >> local_db

            lora_module >> Relationship("Forwards received data") >> data_collector

            data_collector >> Relationship("Stores collected sensor data") >> local_db
            data_collector >> Relationship("Updates coverage metrics") >> environment

            mission_planner >> Relationship("Reads mission status, metrics") >> local_db

            local_db >> Relationship("Uploads aggregated data, Q-table snapshots [HTTPS/MQTT]") >> cloud_platform

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

    logger.info("Running Q-Learning container diagram script...")
    create_diagram()
    logger.info("Script finished!")