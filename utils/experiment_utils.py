##
#
# Helper to broadcast which experiment is running.
#
##

# standard imports
import json
import os

# ROS imports
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

# topic the control nodes broadcast on and the logger records from
EXPERIMENT_INFO_TOPIC = "deploy_robot/experiment_info"


# latched QoS: a logger that subscribes AFTER the control node started still
# receives the last experiment-info message.
def experiment_info_qos():
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


# build the experiment-info dict
def build_experiment_info(config_name, config, policy=None):
    # strip directory + extension from the config name
    name = os.path.splitext(os.path.basename(config_name))[0]
    info = {"config": name}

    # verbatim copy of the config yaml -> the run stays reproducible even if the
    # yaml file is later edited. fall back to the parsed dict if the file is gone.
    root = os.getenv("DEPLOY_ROOT_DIR", "")
    cfg_file = os.path.join(root, "deploy", "configs", name + ".yaml")
    try:
        with open(cfg_file, "r") as f:
            info["config_yaml"] = f.read()
    except OSError:
        if config is not None:
            info["config_yaml"] = json.dumps(config, indent=2)

    # the policy's training-run id (from policy metadata, not present in the yaml)
    if policy is not None:
        run_path = getattr(policy, "metadata", {}).get("run_path")
        if run_path is not None:
            info["policy_trained_at"] = run_path if isinstance(run_path, str) else str(run_path)

    # drop missing keys
    return {k: v for k, v in info.items() if v is not None}


# create a latched publisher and broadcast the experiment info once
def publish_experiment_info(node, config_name, config, policy=None):
    info = build_experiment_info(config_name, config, policy)
    pub = node.create_publisher(String, EXPERIMENT_INFO_TOPIC, experiment_info_qos())
    msg = String()
    msg.data = json.dumps(info)
    pub.publish(msg)
    node.get_logger().info(f"Broadcasting experiment info: config='{info.get('config')}', fields={list(info.keys())}")
    return pub
