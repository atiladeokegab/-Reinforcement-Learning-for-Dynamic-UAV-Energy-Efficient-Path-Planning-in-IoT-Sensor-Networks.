from pathlib import Path
import logging
from diagrams import Diagram
from diagrams.c4 import Person, System, SystemBoundary, Relationship

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.parent.parent.parent


def create_diagram():
    logger.info(f"CURRENT_DIR: {CURRENT_DIR}")
    logger.info("Creating UAV Q-Learning System Context diagram...")

    # Set output directory
    output_dir = CURRENT_DIR / "asset" / "diagrams" / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "q_learning_system_context"

    logger.debug(f"Output file: {output_file}.png")

    graph_attr = {
        "splines": "spline",
    }

    try:
        with Diagram(
                "System Context - UAV Q-Learning IoT Data Collection",
                direction="TB",
                graph_attr=graph_attr,
                show=False,
                filename=str(output_file),
                outformat="png"
        ):
            # External Actors
            mission_operator = Person(
                name="Mission Operator",
                description="Plans UAV missions and configures Q-Learning parameters"
            )

            data_analyst = Person(
                name="Data Analyst",
                description="Analyzes collected IoT data and system performance"
            )

            # External Systems
            iot_sensor_network = System(
                name="IoT Sensor Network",
                description="LoRa-enabled sensor nodes deployed in the field",
                external=True
            )

            lorawan_gateway = System(
                name="LoRaWAN Gateway",
                description="Network gateway for LoRa communication",
                external=True
            )

            cloud_storage = System(
                name="Cloud Storage & Analytics",
                description="Stores mission data, logs, and provides analytics dashboard",
                external=True
            )

            # Main System
            with SystemBoundary("UAV Q-Learning Data Collection System"):
                uav_system = System(
                    name="UAV Q-Learning Agent",
                    description="Autonomous UAV with Q-Learning algorithm for optimal IoT data collection"
                )

            # Relationships
            mission_operator >> Relationship("Configures mission parameters, Q-Learning settings") >> uav_system
            mission_operator >> Relationship("Monitors mission status, views dashboard") >> cloud_storage

            iot_sensor_network >> Relationship("Transmits sensor data [LoRa]") >> uav_system
            uav_system >> Relationship("Collects data from sensors in range") >> iot_sensor_network

            lorawan_gateway >> Relationship("Provides network connectivity") >> uav_system

            uav_system >> Relationship("Uploads trajectories, collected data, Q-values, rewards") >> cloud_storage

            data_analyst >> Relationship("Analyzes performance metrics, data coverage") >> cloud_storage

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

    logger.info("Context diagram creation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Running Q-Learning context diagram script...")
    create_diagram()
    logger.info("Script finished!")