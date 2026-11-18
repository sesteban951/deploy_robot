"""Frozen forward-only FP32 G1 rollout, on the JAX CUDA stream.

One fixed cooperative kernel configuration, no alternative evaluator or solver.
Compilation is startup-only. FFI libraries and CUDA modules outlive all captured
executables through the process-local registration owner.
"""
from functools import lru_cache
from pathlib import Path
import ctypes as ct
import hashlib
import shutil
import subprocess
import sysconfig
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SITE = Path(sysconfig.get_paths()['purelib'])
PARAM_FLOATS, FORWARD_WORK_PER_STEP = 660312, 1877
SOURCE_NAMES = ('rollout.cu', 'batch.cu', 'rollout_ffi.cc', 'projection.cuh',
                'projection_rounding.cuh', 'runtime.py')


def packed_lift_parameters(model):
    """Four frozen linear layers, three LN pairs, normalizer and epsilons."""
    network = model.lifting_network.learned_features.network
    linear = [layer for layer in network if isinstance(layer, torch.nn.Linear)]
    norm = [layer for layer in network if isinstance(layer, torch.nn.LayerNorm)]
    if (tuple(type(layer).__name__ for layer in network)
            != ('Linear', 'LayerNorm', 'ELU')*3 + ('Linear',)
            or [layer.in_features for layer in linear] != [85, 512, 512, 512]
            or [layer.out_features for layer in linear] != [512, 512, 512, 171]
            or any(layer.alpha != 1. for layer in network if isinstance(layer, torch.nn.ELU))):
        raise ValueError('Native GPU inference requires the frozen 85/512/171 LN/ELU architecture')
    tensors = [layer.weight.T for layer in linear] + [layer.bias for layer in linear]
    tensors += [value for layer in norm for value in (layer.weight, layer.bias)]
    tensors += [model.normalizer.state_mean, model.normalizer.state_scale]
    if any(value.dtype != torch.float32 for value in tensors):
        raise ValueError('Native GPU inference requires FP32 checkpoint parameters')
    flat = [np.ascontiguousarray(value.detach().cpu().numpy()).reshape(-1) for value in tensors]
    flat.append(np.asarray([layer.eps for layer in norm], dtype=np.float32))
    result = np.concatenate(flat)
    if result.shape != (PARAM_FLOATS,) or not np.isfinite(result).all():
        raise ValueError('Invalid frozen parameter buffer')
    return result


def _compile(architecture):
    compiler = shutil.which('g++')
    if compiler is None:
        raise RuntimeError('Native KCF GPU startup needs g++ with C++17 support')
    build = tempfile.TemporaryDirectory(prefix='kcf_native_rollout_')
    directory = Path(build.name)
    nvrtc = ct.CDLL(str(SITE/'nvidia/cuda_nvrtc/lib/libnvrtc.so.12'))
    program = ct.c_void_p()
    nvrtc.nvrtcCreateProgram.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p,
        ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p]
    nvrtc.nvrtcCompileProgram.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)]
    for name in ('nvrtcGetProgramLogSize', 'nvrtcGetPTXSize'):
        getattr(nvrtc, name).argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    for name in ('nvrtcGetProgramLog', 'nvrtcGetPTX'):
        getattr(nvrtc, name).argtypes = [ct.c_void_p, ct.c_void_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [ct.POINTER(ct.c_void_p)]
    status = nvrtc.nvrtcCreateProgram(ct.byref(program), (HERE/'rollout.cu').read_bytes(),
                                    b'rollout.cu', 0, None, None)
    if status:
        raise RuntimeError(f'Cannot create native CUDA program: NVRTC {status}')
    try:
        options = [b'--std=c++17', f'--gpu-architecture={architecture}'.encode(),
                   f'--include-path={HERE}'.encode()]
        options += [f'--include-path={SITE/name}'.encode() for name in (
            'nvidia/cuda_runtime/include', 'nvidia/cuda_nvcc/include', 'nvidia/cuda_cccl/include')]
        status = nvrtc.nvrtcCompileProgram(program, len(options), (ct.c_char_p*len(options))(*options))
        size = ct.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
        log = ct.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(program, log)
        if status:
            raise RuntimeError(f'Native CUDA compilation failed ({status}): {log.value.decode()}')
        if nvrtc.nvrtcGetPTXSize(program, ct.byref(size)):
            raise RuntimeError('Cannot size native PTX')
        ptx = ct.create_string_buffer(size.value)
        if nvrtc.nvrtcGetPTX(program, ptx):
            raise RuntimeError('Cannot retrieve native PTX')
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
    binary = directory/'rollout_ffi.so'
    try:
        subprocess.run([compiler, '-std=c++17', '-O3', '-shared', '-fPIC',
            '-Wno-attributes', '-Wno-return-type', f'-I{SITE/"jaxlib/include"}',
            f'-I{SITE/"nvidia/cuda_runtime/include"}', str(HERE/'rollout_ffi.cc'),
            '-lcuda', '-o', str(binary)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Native CUDA FFI build failed: {error.stderr}') from error
    return build, binary, ptx


@lru_cache(maxsize=1)
def _registration():
    # Force initialization before querying the device or loading its module.
    device = jax.devices('gpu')[0]
    jax.block_until_ready(jax.device_put(np.zeros(1, np.float32), device))
    driver = ct.CDLL('libcuda.so.1')
    if driver.cuInit(0):
        raise RuntimeError('Cannot initialize CUDA driver')
    cuda_device, major, minor = ct.c_int(), ct.c_int(), ct.c_int()
    if (driver.cuDeviceGet(ct.byref(cuda_device), 0)
            or driver.cuDeviceGetAttribute(ct.byref(major), 75, cuda_device)
            or driver.cuDeviceGetAttribute(ct.byref(minor), 76, cuda_device)):
        raise RuntimeError('Cannot query visible CUDA device architecture')
    build, binary, ptx = _compile(f'compute_{major.value}{minor.value}')
    library = ct.CDLL(str(binary))
    library.kcf_rollout_error_native.restype = ct.c_char_p
    library.kcf_rollout_initialize_native.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int)]
    info = (ct.c_int*10)()
    if library.kcf_rollout_initialize_native(ptx, info):
        raise RuntimeError(library.kcf_rollout_error_native().decode())
    if info[5] or info[7] or info[9] < 1:
        raise RuntimeError('Native GPU kernel spills; this GPU/compiler combination is not qualified')
    digest = hashlib.sha256()
    for name in SOURCE_NAMES:
        digest.update(name.encode())
        digest.update((HERE/name).read_bytes())
    digest.update(jax.__version__.encode())
    source_hash = digest.hexdigest()
    suffix = source_hash[:20]
    for direction in ('forward', 'batch'):
        symbol = f'kcf_rollout_{direction}_native'
        jax.ffi.register_ffi_target(symbol+'_'+suffix,
            jax.ffi.pycapsule(getattr(library, symbol)), platform='CUDA')
    library.ffi_suffix = '_'+suffix
    library.build_directory = build  # Keep artifacts alive with the library.
    library.build_info = dict(blocks=32, threads=512, shared_bytes=86848,
        parameter_floats=PARAM_FLOATS, forward_work_per_step=FORWARD_WORK_PER_STEP,
        cooperative=info[0], sms=info[1], shared_limit=info[2],
        forward_blocks_per_sm=info[3], forward_registers=info[4], forward_local_bytes=info[5],
        batch_registers=info[6], batch_local_bytes=info[7],
        batch_shared_bytes=info[8], batch_blocks_per_sm=info[9],
        source_sha256=source_hash, cuda_architecture=f'{major.value}.{minor.value}')
    return library


class NativeRollout:
    """Full physical-state recursion for inference; no derivative registration."""
    def __init__(self, model, array, device):
        if device.platform != 'gpu' or not np.isclose(model.dt, .02, rtol=0., atol=1e-12):
            raise ValueError('Native rollout requires CUDA and the frozen 20 ms model')
        self.dt, self.balance = float(model.dt), float(model.projection_balance)
        if not np.isfinite(self.balance) or self.balance <= 0:
            raise ValueError('Projection balance must be finite and positive')
        packed = packed_lift_parameters(model)
        self.packed_weights = array(packed)
        batch_weights = packed.copy()
        for offset, inputs, outputs in ((0, 85, 512), (43520, 512, 512),
                                        (305664, 512, 512), (567808, 512, 171)):
            batch_weights[offset:offset + inputs * outputs] = packed[
                offset:offset + inputs * outputs].reshape(inputs, outputs).T.ravel()
        self.batch_weights = array(batch_weights)
        self.registration = _registration()


    def _check(self, initial, matrices, weights):
        if (initial.shape != (85,) or len(matrices.shape) != 3
                or matrices.shape[1:] != (85, 256) or not 1 <= matrices.shape[0] <= 20
                or weights.shape != (PARAM_FLOATS,)
                or any(v.dtype != jnp.float32 for v in (initial, matrices, weights))):
            raise ValueError('Native FP32 rollout buffer shape/dtype mismatch')

    def forward(self, initial, matrices, weights):
        self._check(initial, matrices, weights)
        h = matrices.shape[0]
        shapes = (jax.ShapeDtypeStruct((h, 85), jnp.float32),
                  jax.ShapeDtypeStruct((h*FORWARD_WORK_PER_STEP,), jnp.float32))
        return jax.ffi.ffi_call('kcf_rollout_forward_native'+self.registration.ffi_suffix,
            shapes, vmap_method='sequential')(initial, matrices, weights,
                dt=np.float64(self.dt), balance=np.float64(self.balance))

    def __call__(self, initial, matrices):
        return self.forward(initial, matrices, self.packed_weights)[0]

    def batched(self, initials, matrices):
        """Forward-only independent trajectories; no gradient primitive."""
        if (matrices.ndim != 4 or matrices.shape[0] < 1
                or not 1 <= matrices.shape[1] <= 20 or matrices.shape[2:] != (85, 256)
                or initials.shape != (matrices.shape[0], 85)
                or any(v.dtype != jnp.float32 for v in (initials, matrices))):
            raise ValueError('Native population rollout shape/dtype mismatch')
        shape = jax.ShapeDtypeStruct(matrices.shape[:2] + (85,), jnp.float32)
        return jax.ffi.ffi_call('kcf_rollout_batch_native'+self.registration.ffi_suffix,
            shape, vmap_method='sequential')(initials, matrices, self.batch_weights,
                dt=np.float64(self.dt), balance=np.float64(self.balance))
