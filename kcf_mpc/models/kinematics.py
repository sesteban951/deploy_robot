"""Differentiable G1 geometry and weighted position/velocity projection."""

import torch

POSITION = (0, 1, 7)
VELOCITY = (4, 5, 40)
ROM10 = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10)


def unit_vector(value, fallback):
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    default = value.new_tensor(fallback).expand_as(value)
    return torch.where(norm > 1e-8, value / norm.clamp_min(1e-8), default)


def unit_geometry(state):
    """Keep the heading circle and gravity sphere in physical units."""
    return torch.cat((state[..., :2], unit_vector(state[..., 2:4], (0., 1.)),
                      state[..., 4:8], unit_vector(state[..., 8:11], (0., 0., -1.)),
                      state[..., 11:]), dim=-1)


def body_to_local_rotation(state):
    """Reconstruct full ZYX orientation from heading and projected down gravity.

    In the upright G1 envelope, g_B = -R_LB.T @ e_z. Heading is measured
    about local vertical. Roll and pitch are retained. At the Euler singularity
    use a bounded, orthonormal extension; recorded upright states avoid it.
    """
    heading = unit_vector(state[..., 2:4], (0., 1.))
    gravity = unit_vector(state[..., 8:11], (0., 0., -1.))
    sy, cy = heading.unbind(-1)
    sp = gravity[..., 0].clamp(-1. + 1e-7, 1. - 1e-7)
    cp = (1. - sp.square()).sqrt()
    sr, cr = unit_vector(-gravity[..., 1:3], (0., 1.)).unbind(-1)
    return torch.stack((
        torch.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr), -1),
        torch.stack((sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), -1),
        torch.stack((-sp, cp*sr, cp*cr), -1),
    ), -2)


def kinematic_residual(previous, following, dt):
    before = (body_to_local_rotation(previous) @ previous[..., list(VELOCITY), None]).squeeze(-1)
    after = (body_to_local_rotation(following) @ following[..., list(VELOCITY), None]).squeeze(-1)
    return following[..., list(POSITION)] - previous[..., list(POSITION)] - .5*dt*(before + after)


def project_state(previous, prediction, state_scale, *, dt=.02, balance=1.):
    """Closest scaled learned p/v pair satisfying the trapezoidal equation."""
    prediction = unit_geometry(prediction)
    rotation = body_to_local_rotation(prediction)
    a = .5 * dt * rotation
    b = previous[..., list(POSITION)] + .5*dt*(
        body_to_local_rotation(previous) @ previous[..., list(VELOCITY), None]
    ).squeeze(-1)
    wp = balance / state_scale[list(POSITION)].clamp_min(1e-4).square()
    wv = 1. / state_scale[list(VELOCITY)].clamp_min(1e-4).square()
    weighted_a = wp[:, None] * a
    matrix = a.transpose(-1, -2) @ weighted_a + torch.diag(wv)
    rhs = wv * prediction[..., list(VELOCITY)] + (
        a.transpose(-1, -2) @ (wp*(prediction[..., list(POSITION)] - b)).unsqueeze(-1)
    ).squeeze(-1)
    # solve_ex avoids the per-step CUDA/CPU synchronization of solve(). The
    # coefficient matrix is positive definite because all wv entries are > 0.
    velocity = torch.linalg.solve_ex(matrix, rhs.unsqueeze(-1)).result.squeeze(-1)
    position = b + (a @ velocity.unsqueeze(-1)).squeeze(-1)
    corrected = prediction.clone()
    corrected[..., list(POSITION)] = position
    corrected[..., list(VELOCITY)] = velocity
    return corrected
