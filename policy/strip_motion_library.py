##
#
# Strip the bundled reference-motion library out of mjlab ONNX policies.
#
# mjlab exports the training motion library (joint_pos, joint_vel, body_*_w for every
# clip) as extra outputs indexed by `time_step`. The `actions` output never reads them,
# so deployment only needs the MLP. This keeps just `actions`, prunes everything it does
# not depend on, records the clip length as `motion_period_frames` metadata, and checks
# that the stripped actions are bit-identical to the original before overwriting.
#
# Usage:
#   python policy/strip_motion_library.py                # every .onnx in policy/
#   python policy/strip_motion_library.py crawl_omni.onnx
#
##

# standard imports
import argparse
import os

# other imports
import numpy as np
import onnx
import onnxruntime as ort

# directory imports
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
POLICY_DIR = os.path.join(ROOT_DIR, "policy")

# the only output deployment uses
ACTION_OUTPUT = "actions"

# number of random observations used to check the stripped policy
NUM_CHECKS = 1000


# every tensor name the given output depends on
def ancestors(graph, output_name):
    producer = {o: n for n in graph.node for o in n.output}
    seen, stack = set(), [output_name]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if t in producer:
            stack.extend(i for i in producer[t].input if i)
    return seen


# frames per clip and number of clips from the bundled joint_pos output, if present
def motion_library_shape(model):
    frames = None
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        if out.name == "joint_pos" and len(dims) == 3:
            frames = dims[1]
    clips = None
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0].startswith("joint_pos"):
            for a in n.attribute:
                if len(a.t.dims) == 3:
                    clips = a.t.dims[0]
    for init in model.graph.initializer:
        if init.name.startswith("joint_pos") and len(init.dims) == 3:
            clips = init.dims[0]
    return frames, clips


# keep only the actions output and the nodes, initializers and inputs it needs
def strip(model):
    g = model.graph
    needed = ancestors(g, ACTION_OUTPUT)

    keep_nodes = [n for n in g.node if any(o in needed for o in n.output)]
    keep_inits = [i for i in g.initializer if i.name in needed]
    keep_inputs = [i for i in g.input if i.name in needed]
    keep_outputs = [o for o in g.output if o.name == ACTION_OUTPUT]
    keep_vinfo = [v for v in g.value_info if v.name in needed]

    del g.node[:]
    g.node.extend(keep_nodes)
    del g.initializer[:]
    g.initializer.extend(keep_inits)
    del g.input[:]
    g.input.extend(keep_inputs)
    del g.output[:]
    g.output.extend(keep_outputs)
    del g.value_info[:]
    g.value_info.extend(keep_vinfo)

    return model


# set (or overwrite) one metadata entry
def set_metadata(model, key, value):
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = str(value)
            return
    entry = model.metadata_props.add()
    entry.key, entry.value = key, str(value)


# run the actions output on the given feeds
def run_actions(model_bytes, feeds):
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    return [sess.run([ACTION_OUTPUT], {k: v for k, v in f.items() if k in names})[0] for f in feeds]


# strip one policy file in place; returns (old_bytes, new_bytes) or None if skipped
def process(path):
    model = onnx.load(path)
    outputs = [o.name for o in model.graph.output]

    if ACTION_OUTPUT not in outputs:
        print(f"[skip] {os.path.basename(path)}: no '{ACTION_OUTPUT}' output ({outputs})")
        return None
    if outputs == [ACTION_OUTPUT]:
        print(f"[skip] {os.path.basename(path)}: already actions-only")
        return None

    frames, clips = motion_library_shape(model)
    original_bytes = model.SerializeToString()

    # random observations (and time steps, which must not matter) for the check
    rng = np.random.default_rng(0)
    obs_input = model.graph.input[0]
    obs_dim = obs_input.type.tensor_type.shape.dim[-1].dim_value
    feeds = []
    for _ in range(NUM_CHECKS):
        feed = {obs_input.name: rng.standard_normal((1, obs_dim)).astype(np.float32)}
        for inp in model.graph.input[1:]:
            shape = [max(d.dim_value, 1) for d in inp.type.tensor_type.shape.dim]
            feed[inp.name] = rng.integers(0, max(frames or 1, 1), size=shape).astype(np.float32)
        feeds.append(feed)

    # strip, record the clip length, and validate the graph
    stripped = strip(model)
    if frames is not None:
        set_metadata(stripped, "motion_period_frames", frames)
    if clips is not None:
        set_metadata(stripped, "motion_num_clips", clips)
    onnx.checker.check_model(stripped)
    stripped_bytes = stripped.SerializeToString()

    # the stripped actions must be bit-identical to the original
    before = run_actions(original_bytes, feeds)
    after = run_actions(stripped_bytes, feeds)
    for a, b in zip(before, after):
        if not np.array_equal(a, b):
            raise RuntimeError(f"{path}: stripped actions differ (max {np.abs(a - b).max():.3e}); not overwritten")

    # write atomically over the original
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(stripped_bytes)
    os.replace(tmp_path, path)

    print(f"[ok]   {os.path.basename(path)}: {len(original_bytes) / 1e6:6.1f} MB -> "
          f"{len(stripped_bytes) / 1e6:5.2f} MB  (T={frames}, clips={clips}, "
          f"inputs={[i.name for i in stripped.graph.input]}, {NUM_CHECKS} checks bit-identical)")
    return len(original_bytes), len(stripped_bytes)


def main():
    parser = argparse.ArgumentParser(description="Strip bundled motion libraries from ONNX policies.")
    parser.add_argument("policies", nargs="*", help="policy filenames in policy/ (default: all .onnx)")
    args = parser.parse_args()

    names = args.policies or sorted(f for f in os.listdir(POLICY_DIR) if f.endswith(".onnx"))
    total_old = total_new = 0
    for name in names:
        result = process(os.path.join(POLICY_DIR, name))
        if result is not None:
            total_old += result[0]
            total_new += result[1]

    if total_old:
        print(f"\nTotal: {total_old / 1e6:.1f} MB -> {total_new / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
