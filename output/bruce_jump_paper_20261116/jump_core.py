"""Offline research components from arXiv:2304.00536v1, Sections IV and V.

There is deliberately no serial port, live shared-memory writer, launch hook or
motor enable interface here. A solved QP is not a validated jump trajectory.
All spatial quantities must use consistent frames; landing uses a level,
yaw-aligned frame (x forward, y left, z up), with feet ordered right then left.
"""
from dataclasses import dataclass
import math
import numpy as np
import osqp
from scipy import sparse


def finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite array of shape {shape}")
    return value


def positive(value, shape, name):
    value = finite(value, shape, name)
    if np.any(value <= 0):
        raise ValueError(f"{name}: must be positive")
    return value


def ballistic_requirements(com_rise_m, gravity=9.81):
    """CoM rise from takeoff, not foot clearance or full squat-to-apex change."""
    if not (math.isfinite(com_rise_m) and com_rise_m > 0 and
            math.isfinite(gravity) and gravity > 0):
        raise ValueError("Positive finite height and gravity required")
    vz = math.sqrt(2 * gravity * com_rise_m)
    return {"takeoff_vz_m_s": vz, "same_height_flight_s": 2 * vz / gravity}


@dataclass(frozen=True)
class LandingGains:
    # Length / linear-momentum units: s/kg.
    linear_x: float
    linear_y: float
    # Length / angular-momentum units: s/(kg*m). Signs require robot validation.
    angular_pitch: float
    angular_roll: float
    widen_y: float

    def __post_init__(self):
        values = np.array([self.linear_x, self.linear_y, self.angular_pitch,
                           self.angular_roll, self.widen_y])
        if not np.isfinite(values).all() or self.widen_y < 0:
            raise ValueError("Finite gains and nonnegative widening required")


def landing_targets(nominal, linear_momentum, angular_momentum, gains,
                    lower, upper, minimum_width):
    """Eqs. 18-22. Bounds are supplied per foot in the SAME reference frame.

    z stays on the optimized vertical reference. Bounds are only a rectangular
    screen; real leg IK, self collision and terrain clearance remain necessary.
    Fail closed if saturation destroys minimum foot separation.
    """
    feet = finite(nominal, (2, 3), "nominal").copy()
    lin = finite(linear_momentum, (3,), "linear_momentum")
    ang = finite(angular_momentum, (3,), "angular_momentum")
    lo = finite(lower, (2, 3), "lower")
    hi = finite(upper, (2, 3), "upper")
    if np.any(lo > hi) or not math.isfinite(minimum_width) or minimum_width <= 0:
        raise ValueError("Invalid bounds or minimum width")
    if np.any(feet[:, 2] < lo[:, 2]) or np.any(feet[:, 2] > hi[:, 2]):
        raise ValueError("Optimized vertical reference outside reach screen")
    feet[:, 0] += gains.linear_x * lin[0] + gains.angular_pitch * ang[1]
    feet[:, 1] += gains.linear_y * lin[1] + gains.angular_roll * ang[0]
    widening = gains.widen_y * abs(lin[1])
    feet[0, 1] -= widening  # right
    feet[1, 1] += widening  # left
    feet[:, :2] = np.clip(feet[:, :2], lo[:, :2], hi[:, :2])
    if feet[1, 1] - feet[0, 1] < minimum_width - 1e-12:
        raise ValueError("Landing bounds would cross or crowd the feet")
    return feet


def flight_pd(q, dq, q_ref, dq_ref, kp, kd, torque_limit,
              joint_torque_to_current, current_limit):
    """Eq. 23 flight branch: feedforward is zero; bound the TOTAL PD effort.

    Uniform scaling is a screening mechanism, not proof of adequate tracking.
    The transmission/current map must come from the actual robot version.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q must be a vector")
    n = len(q)
    q = finite(q, (n,), "q")
    dq = finite(dq, (n,), "dq")
    qr = finite(q_ref, (n,), "q_ref")
    dqr = finite(dq_ref, (n,), "dq_ref")
    kp = finite(kp, (n,), "kp")
    kd = finite(kd, (n,), "kd")
    if np.any(kp < 0) or np.any(kd < 0):
        raise ValueError("Negative PD gain")
    torque_limit = positive(torque_limit, (n,), "torque_limit")
    current_map = finite(joint_torque_to_current, (n, n), "current map")
    current_limit = positive(current_limit, (n,), "current_limit")
    torque = kp * (qr - q) + kd * (dqr - dq)
    ratio = max(1., float(np.max(np.abs(torque) / torque_limit)),
                float(np.max(np.abs(current_map @ torque) / current_limit)))
    return {"torque": torque / ratio, "current": current_map @ (torque / ratio),
            "scale": 1. / ratio}


@dataclass(frozen=True)
class AccelerationTask:
    """J*ddq + bias = target. Weight is a nonnegative diagonal vector."""
    jacobian: np.ndarray
    bias: np.ndarray
    target: np.ndarray
    weight: np.ndarray


class JumpQP:
    """Paper Eqs. 24-25, with explicit total-torque and motor-current bounds.

    Coordinates: the first six generalized velocities are floating base;
    remaining coordinates are leg joints. H, CG and every J use the same basis.
    Weighted tasks approximate priority, as in the paper; not lexicographic HQP.
    No last-good solution is returned on failure.
    """

    def __init__(self, nv=16, contact_count=4, acceleration_regularizer=1e-5,
                 force_regularizer=1e-5):
        if nv <= 6 or contact_count < 1:
            raise ValueError("Floating base and contacts required")
        positive([acceleration_regularizer, force_regularizer], (2,), "regularizers")
        self.nv, self.nc = nv, contact_count
        self.acc_reg, self.force_reg = acceleration_regularizer, force_regularizer
        self.solver = None
        self.pattern = None

    def solve(self, H, CG, Jc, active, mu, maximum_normal_force, tasks,
              torque_limit, joint_torque_to_current, current_limit, *,
              pd_torque=None, joint_acceleration_lower=None,
              joint_acceleration_upper=None):
        nv, nc, nj = self.nv, self.nc, self.nv - 6
        nf, nx = 3 * nc, nv + 3 * nc
        H = finite(H, (nv, nv), "H")
        if not np.allclose(H, H.T, atol=1e-8):
            raise ValueError("H must be symmetric")
        # Reject corrupt or inconsistent mass matrices before any optimization.
        np.linalg.cholesky(H)
        CG = finite(CG, (nv,), "CG")
        Jc = finite(Jc, (nc, 3, nv), "Jc")
        active = np.asarray(active)
        if active.shape != (nc,) or not np.isin(active, [0, 1]).all():
            raise ValueError("active must be a contact mask")
        if not math.isfinite(mu) or mu <= 0:
            raise ValueError("Invalid friction coefficient")
        fmax = positive(maximum_normal_force, (nc,), "maximum_normal_force")
        tau_max = positive(torque_limit, (nj,), "torque_limit")
        M = finite(joint_torque_to_current, (nj, nj), "current map")
        i_max = positive(current_limit, (nj,), "current_limit")
        pd = np.zeros(nj) if pd_torque is None else finite(pd_torque, (nj,), "pd_torque")
        P = np.diag(np.r_[np.full(nv, self.acc_reg), np.full(nf, self.force_reg)])
        cost = np.zeros(nx)
        for task in tasks:
            J = np.asarray(task.jacobian, dtype=float)
            if J.ndim != 2 or J.shape[1] != nv:
                raise ValueError("Task Jacobian shape")
            J = finite(J, J.shape, "task J")
            m = J.shape[0]
            b = finite(task.bias, (m,), "task bias")
            target = finite(task.target, (m,), "task target")
            weight = finite(task.weight, (m,), "task weight")
            if np.any(weight < 0):
                raise ValueError("Negative task weight")
            P[:nv, :nv] += J.T @ (weight[:, None] * J)
            cost[:nv] += J.T @ (weight * (b - target))
        stacked = Jc.reshape(nf, nv)
        dyn = np.c_[H[:6], -stacked[:, :6].T]
        rows, lower, upper = [dyn], [-CG[:6]], [-CG[:6]]
        for i in range(nc):
            # Inscribed pyramid |fx|+|fy| <= mu*fz stays within circular cone.
            C = np.zeros((5, nx))
            C[:4, nv + 3*i:nv + 3*i + 3] = [
                [1, 1, -mu], [1, -1, -mu],
                [-1, 1, -mu], [-1, -1, -mu]]
            C[4, nv + 3*i + 2] = 1.
            rows.append(C)
            lower.append(np.r_[np.full(4, -np.inf), 0.])
            upper.append(np.r_[np.zeros(4), fmax[i] * active[i]])
        # Include joint PD in constraints: bounding only tau_ff is insufficient.
        T = np.c_[H[6:], -stacked[:, 6:].T]
        offset = CG[6:] + pd
        rows.extend([T, M @ T])
        lower.extend([-tau_max - offset, -i_max - M @ offset])
        upper.extend([tau_max - offset, i_max - M @ offset])
        if (joint_acceleration_lower is None) != (joint_acceleration_upper is None):
            raise ValueError("Both joint acceleration bounds required")
        if joint_acceleration_lower is not None:
            alo = finite(joint_acceleration_lower, (nj,), "acceleration lower")
            ahi = finite(joint_acceleration_upper, (nj,), "acceleration upper")
            if np.any(alo > ahi):
                raise ValueError("Empty acceleration interval")
            S = np.zeros((nj, nx))
            S[:, 6:nv] = np.eye(nj)
            rows.append(S)
            lower.append(alo)
            upper.append(ahi)
        A = np.vstack(rows)
        lo, hi = np.concatenate(lower), np.concatenate(upper)
        # Fixed dense sparsity preserves zero-to-nonzero entries across updates.
        pr, pc = np.triu_indices(nx)
        Ps = sparse.csc_matrix((P[pr, pc], (pr, pc)), shape=(nx, nx))
        ar, ac = np.indices(A.shape)
        As = sparse.csc_matrix((A.ravel(), (ar.ravel(), ac.ravel())), shape=A.shape)
        pattern = (nx, A.shape[0])
        if self.solver is None or self.pattern != pattern:
            self.solver = osqp.OSQP()
            self.solver.setup(P=Ps, q=cost, A=As, l=lo, u=hi,
                              verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                              max_iter=4000, polish=False)
            self.pattern = pattern
        else:
            self.solver.update(Px=Ps.data, Ax=As.data, q=cost, l=lo, u=hi)
        solution = self.solver.solve()
        if solution.info.status != "solved" or solution.x is None:
            raise RuntimeError("Jump QP rejected: " + solution.info.status)
        x = solution.x
        if not np.isfinite(x).all():
            raise RuntimeError("Nonfinite QP solution")
        residual = max(float(np.max(lo - A @ x)), float(np.max(A @ x - hi)), 0.)
        if residual > 1e-3:
            raise RuntimeError(f"Constraint violation {residual}")
        tau_ff = T @ x + CG[6:]
        tau = tau_ff + pd
        return {"ddq": x[:nv], "forces": x[nv:].reshape(nc, 3),
                "torque_ff": tau_ff, "torque_total": tau,
                "current": M @ tau, "constraint_violation": residual,
                "solver_s": solution.info.solve_time,
                "status": solution.info.status}
