##
#
# Policy class for handling policy-related operations,
# such as loading, inferencing, and getting properties.
#
##

import json

import numpy as np
import onnx
import onnxruntime as ort


############################################################################
# HELPERS
############################################################################

# get the input and output dimensions of an onnx policy
def get_policy_io_size_onnx(policy):

    # input size from first input tensor
    input_shape = policy.graph.input[0].type.tensor_type.shape
    input_size = input_shape.dim[-1].dim_value

    # output size from first output tensor
    output_shape = policy.graph.output[0].type.tensor_type.shape
    output_size = output_shape.dim[-1].dim_value

    return input_size, output_size


# parse a CSV string into a list of floats
def parse_float_csv(s):
    return np.array([float(x) for x in s.split(",") if x.strip()], dtype=np.float32)

# parse a CSV string into a list of strings
def parse_str_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]

# load metadata embedded in an ONNX model
def load_policy_metadata(onnx_model):
    metadata = {}
    for prop in onnx_model.metadata_props:
        value = prop.value

        # JSON values (mjlab gait-library exports: library_clips, twist_metric_weights, ...)
        if value.lstrip().startswith(("[", "{")):
            try:
                metadata[prop.key] = json.loads(value)
                continue
            except ValueError:
                pass

        # try parsing as floats first
        try:
            parsed = parse_float_csv(value)
            if len(parsed) > 0:
                metadata[prop.key] = parsed
                continue
        except ValueError:
            pass

        # try parsing as string list (if it contains commas)
        if "," in value:
            metadata[prop.key] = parse_str_csv(value)
        else:
            metadata[prop.key] = value.strip()

    return metadata

# print each metadata key-value pair of a policy (if it has metadata)
def print_policy_metadata(policy):
    if hasattr(policy, 'metadata'):
        for key, value in policy.metadata.items():
            print()
            print(f"{key}: {value}")
    print()

# inference with an onnx policy
def policy_inference_onnx(session, input, **extra_inputs):

    # build input feed starting with the primary observation
    input_name = session.get_inputs()[0].name
    input_feed = {input_name: input.reshape(1, -1).astype(np.float32)}

    # fill in any additional required inputs (e.g., time_step)
    for inp in session.get_inputs()[1:]:
        if inp.name in extra_inputs:
            val = np.array(extra_inputs[inp.name], dtype=np.float32).reshape(1, -1)
        else:
            shape = [max(d, 1) for d in inp.shape]
            val = np.zeros(shape, dtype=np.float32)
        input_feed[inp.name] = val

    # forward pass
    action = session.run(None, input_feed)[0].squeeze()

    return action


############################################################################
# POLICY CLASS
############################################################################

class Policy:
    """
    Control policy class for ONNX policies (with embedded deployment metadata).
    """

    def __init__(self, policy_path):

        # load the policy
        self._load_policy(policy_path)

        # compute important properties of the policy
        self._get_policy_properties()


    # load an onnx policy given the path
    def _load_policy(self, policy_path):

        # metadata embedded in the policy
        self.metadata = {}

        # only onnx policies are supported
        if not policy_path.lower().endswith(".onnx"):
            raise ValueError("Unsupported policy format. Only .onnx policies are supported.")

        self.policy = onnx.load(policy_path)
        self._onnx_session = ort.InferenceSession(
            self.policy.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        self._policy_type = "onnx"

        # load embedded metadata if available
        if self.policy.metadata_props:
            self.metadata = load_policy_metadata(self.policy)


    # get important properties of the policy
    def _get_policy_properties(self):
        self.input_size, self.output_size = get_policy_io_size_onnx(self.policy)
        self.inputs = [{"name": inp.name, "shape": inp.shape} for inp in self._onnx_session.get_inputs()]
        self.outputs = [{"name": out.name, "shape": out.shape} for out in self._onnx_session.get_outputs()]
        self.input_sizes = [inp.shape[-1] for inp in self._onnx_session.get_inputs()]

        # recurrent state (rsl_rl RNN export: h_in/c_in -> h_out/c_out, GRU: h_in -> h_out).
        # Every "<x>_in" input with a matching "<x>_out" output is carried across inference
        # calls, so the network keeps its memory; reset() zeroes it (an episode start).
        in_names = {inp["name"]: inp["shape"] for inp in self.inputs[1:]}
        out_names = {out["name"] for out in self.outputs}
        self._state_io = {n: n[:-3] + "_out" for n in in_names
                          if n.endswith("_in") and n[:-3] + "_out" in out_names}
        self._state_shapes = {n: [d if isinstance(d, int) and d > 0 else 1 for d in in_names[n]]
                              for n in self._state_io}
        self._output_names = [out["name"] for out in self.outputs]
        self.is_recurrent = bool(self._state_io)
        self.reset()


    # zero the recurrent state (no-op for a feed-forward policy)
    def reset(self):
        self._state = {n: np.zeros(shape, dtype=np.float32) for n, shape in self._state_shapes.items()}


    # inference the policy given an input
    def inference(self, input, **extra_inputs):
        if not self.is_recurrent:
            return policy_inference_onnx(self._onnx_session, input, **extra_inputs)

        # recurrent: feed the carried state, keep the new state for the next call
        feed = {self.inputs[0]["name"]: input.reshape(1, -1).astype(np.float32), **self._state}
        results = dict(zip(self._output_names, self._onnx_session.run(None, feed)))
        for n_in, n_out in self._state_io.items():
            self._state[n_in] = results[n_out]
        return results[self._output_names[0]].squeeze()


    # gait period T in frames: from mjlab's per-clip library_clips (None if the clips mix
    # periods -- see utils/locomotion/gait_library.py), else metadata written by
    # policy/strip_motion_library.py, else the bundled joint_pos output's shape, else None
    @property
    def motion_period_frames(self):
        if "library_clips" in self.metadata:
            periods = {int(c[3]) for c in self.metadata["library_clips"]}
            return periods.pop() if len(periods) == 1 else None
        if "motion_period_frames" in self.metadata:
            return int(self.get_param("motion_period_frames")[0])
        for out in self.outputs:
            if out["name"] == "joint_pos" and len(out["shape"]) == 3:
                return int(out["shape"][1])
        return None


    # fetch a deployment parameter embedded in the policy metadata
    def get_param(self, key, default=None):
        if key in self.metadata:
            return np.asarray(self.metadata[key], dtype=np.float32)
        return None if default is None else np.asarray(default, dtype=np.float32)


############################################################################
# TEST
############################################################################

if __name__ == "__main__":

    import argparse
    import os
    ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")

    # policy name argument (with or without the .onnx extension)
    parser = argparse.ArgumentParser(description="Load a policy and print its properties.")
    parser.add_argument('policy', type=str, help='Policy file name inside the policy folder. Example: "g1_29dof_mimic_squat".')
    args = parser.parse_args()
    policy_name = args.policy if args.policy.endswith(".onnx") else args.policy + ".onnx"

    # load the policy
    policy_path = ROOT_DIR + "/policy/" + policy_name
    policy = Policy(policy_path)
    print(f"Policy loaded from [{policy_path}]")
    print(f"    Type: {policy._policy_type}")
    print(f"    Input size: {policy.input_size}")
    print(f"    Output size: {policy.output_size}")
    print(f"    Inputs: {policy.inputs}")
    print(f"    Outputs: {policy.outputs}")

    # print metadata if available
    print_policy_metadata(policy)

    # test inference with a zero input
    obs = np.zeros(policy.input_size, dtype=np.float32)
    action = policy.inference(obs)
    print(f"Test action shape: {action.shape}")
    print(f"Test action: {action}")
