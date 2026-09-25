"""Shared feature definition for the SONIC drift predictor.

ONE definition, used by both sides:
  * offline  (training)  -- features_from_traj(), over a whole reference clip
  * online   (SQP)       -- features_and_jac(), at one frame, with d(phi)/d(qpos)

Train/solve skew is the quietest way this whole approach fails, so both paths
go through the same _assemble() and are checked against each other by
verify_consistency().

Feature layout (21), all in the reference pelvis-heading frame at frame i:
   0      pelvis height
   1:4    pelvis linear velocity        (backward difference)
   4      yaw rate                      (backward difference)
   5      roll rate                     (backward difference)
   6      pitch rate                    (backward difference)
   7      roll
   8      pitch
   9:12   left  foot position rel. pelvis
  12:15   left  foot velocity rel. pelvis   (backward difference)
  15:18   right foot position rel. pelvis
  18:21   right foot velocity rel. pelvis   (backward difference)

Every rate is a BACKWARD difference: inside the SQP, frame i is solved before
frame i+1 exists as a decision variable, so a central difference is not
computable there.

The heading-frame rotation R(-yaw) is DIFFERENTIATED, not frozen. Freezing it
was measured against finite differences at ~41% relative error on the foot-xy
Jacobian columns -- concentrated in the yaw direction the SQP actively pushes
on -- so the dR/dyaw term is carried explicitly.
"""
import numpy as np

FEATURE_DIM = 21
ERROR_DIM = 3          # (dx, dy, dyaw)
DT = 1.0 / 30.0


# ------------------------------------------------------------------ rotations
def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def roll_pitch_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def Rz_inv(yaw):
    """2x2 rotation taking world xy into the heading frame."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s], [s, c]])


def dRz_inv_dyaw(yaw):
    """d/dyaw of Rz_inv. Rz_inv = [[cos y, sin y], [-sin y, cos y]]."""
    return np.array([[-np.sin(yaw), np.cos(yaw)],
                     [-np.cos(yaw), -np.sin(yaw)]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def bdiff(a, dt=DT):
    d = np.empty_like(a)
    d[0] = (a[1] - a[0]) / dt
    d[1:] = (a[1:] - a[:-1]) / dt
    return d


# ------------------------------------------------------------------ assembly
def _assemble(height, v_h, yaw_rate, roll_rate, pitch_rate, roll, pitch,
              lf_rel, lf_vel, rf_rel, rf_vel):
    """The single source of truth for feature ordering.

    Works both per-frame (every arg a scalar or (3,)) and batched over a whole
    clip (every arg (T,) or (T,3)). Both callers must route through here or the
    two paths can silently drift apart.
    """
    parts = [height, v_h, yaw_rate, roll_rate, pitch_rate, roll, pitch,
             lf_rel, lf_vel, rf_rel, rf_vel]
    batched = any(np.ndim(p) == 2 for p in parts)
    out = []
    for p in parts:
        a = np.asarray(p)
        if batched and a.ndim == 1:
            a = a.reshape(-1, 1)      # scalar-per-frame -> column
        elif not batched:
            a = np.atleast_1d(a)
        out.append(a)
    return np.concatenate(out, axis=-1)


# ------------------------------------------------------------------ offline
def features_from_traj(root_pos, root_quat, lf_world, rf_world):
    """(T,21) features for a whole reference clip. lf/rf_world from FK."""
    yaw = yaw_of(root_quat)
    roll, pitch = roll_pitch_of(root_quat)

    v_w = bdiff(root_pos)
    v_h = np.stack([
        np.cos(-yaw) * v_w[:, 0] - np.sin(-yaw) * v_w[:, 1],
        np.sin(-yaw) * v_w[:, 0] + np.cos(-yaw) * v_w[:, 1],
        v_w[:, 2]], axis=1)

    def rel(f):
        d = f - root_pos
        p = np.stack([
            np.cos(-yaw) * d[:, 0] - np.sin(-yaw) * d[:, 1],
            np.sin(-yaw) * d[:, 0] + np.cos(-yaw) * d[:, 1],
            d[:, 2]], axis=1)
        return p, bdiff(p)

    lf_rel, lf_vel = rel(lf_world)
    rf_rel, rf_vel = rel(rf_world)
    return _assemble(root_pos[:, 2], v_h, bdiff(np.unwrap(yaw)),
                     bdiff(roll), bdiff(pitch), roll, pitch,
                     lf_rel, lf_vel, rf_rel, rf_vel)


# ------------------------------------------------------------------ online
def features_and_jac(qpos, lf_world, rf_world, J_lf, J_rf,
                     prev_root_pos, prev_lf_rel, prev_rf_rel,
                     prev_roll, prev_pitch, prev_yaw_unwrapped, dt=DT):
    """One frame: features (21,) and d(phi)/d(qpos) (21, nq).

    qpos        : full configuration at this iterate (root pos 0:3, quat 3:7, joints)
    lf/rf_world : world foot positions at this iterate
    J_lf, J_rf  : (3, nq) world foot Jacobians wrt qpos (from
                  _calc_manipulator_jacobians, already qpos-converted by T)
    prev_*      : the SOLVED previous frame's quantities (constants here)
    """
    # Width comes from the JACOBIAN, not from len(qpos): the retargeting model's
    # configuration carries the 7-DOF object pose after the robot's 36, so these
    # two differ (43 vs 36) and building the selection matrix from len(qpos)
    # silently mismatches. Columns 0:3 (root translation) and 3:7 (root quat) are
    # at the same offsets in both.
    nq = J_lf.shape[1]
    root = qpos[0:3]
    quat = qpos[3:7]
    yaw = yaw_of(quat)
    roll, pitch = roll_pitch_of(quat)
    R = Rz_inv(yaw)
    dR = dRz_inv_dyaw(yaw)

    # d(euler)/d(quat), needed before the rotation terms below
    dang = _euler_jac(quat)               # (3,4) rows = roll, pitch, yaw
    J_ang = np.zeros((3, nq))
    J_ang[:, 3:7] = dang
    dyaw_dq = J_ang[2, :]                 # (nq,)

    # selection matrix for the root translation
    P = np.zeros((3, nq))
    P[0, 0] = P[1, 1] = P[2, 2] = 1.0

    # --- pelvis linear velocity (backward difference) ---
    # The heading frame rotates with yaw, so d/dq carries a dR/dyaw term. Without
    # it the foot-xy columns are ~41% wrong in exactly the yaw direction the SQP
    # pushes on (measured against finite differences).
    v_w = (root - prev_root_pos) / dt
    v_h = np.concatenate([R @ v_w[:2], [v_w[2]]])
    Jv = np.zeros((3, nq))
    Jv[:2, :] = R @ P[:2, :] / dt + np.outer(dR @ v_w[:2], dyaw_dq)
    Jv[2, :] = P[2, :] / dt

    # --- feet relative to pelvis, heading frame ---
    def rel_and_jac(f_world, J_f):
        d = f_world - root
        p = np.concatenate([R @ d[:2], [d[2]]])
        Jd = J_f - P
        Jp = np.zeros((3, nq))
        Jp[:2, :] = R @ Jd[:2, :] + np.outer(dR @ d[:2], dyaw_dq)
        Jp[2, :] = Jd[2, :]
        return p, Jp

    lf_rel, J_lf_rel = rel_and_jac(lf_world, J_lf)
    rf_rel, J_rf_rel = rel_and_jac(rf_world, J_rf)

    lf_vel = (lf_rel - prev_lf_rel) / dt
    rf_vel = (rf_rel - prev_rf_rel) / dt

    yaw_rate = wrap(yaw - prev_yaw_unwrapped) / dt
    roll_rate = (roll - prev_roll) / dt
    pitch_rate = (pitch - prev_pitch) / dt

    phi = _assemble(root[2], v_h, yaw_rate, roll_rate, pitch_rate, roll, pitch,
                    lf_rel, lf_vel, rf_rel, rf_vel)

    J = np.zeros((FEATURE_DIM, nq))
    J[0, :] = P[2, :]                     # height
    J[1:4, :] = Jv                        # pelvis lin vel
    J[4, :] = J_ang[2, :] / dt            # yaw rate
    J[5, :] = J_ang[0, :] / dt            # roll rate
    J[6, :] = J_ang[1, :] / dt            # pitch rate
    J[7, :] = J_ang[0, :]                 # roll
    J[8, :] = J_ang[1, :]                 # pitch
    J[9:12, :] = J_lf_rel
    J[12:15, :] = J_lf_rel / dt           # lf vel (prev is constant)
    J[15:18, :] = J_rf_rel
    J[18:21, :] = J_rf_rel / dt
    return phi, J


def _euler_jac(q, eps=1e-6):
    """(3,4) d[roll,pitch,yaw]/d[quat]. Finite difference: this is 4 cheap
    evaluations of a closed form, and it cannot silently disagree with
    roll_pitch_of/yaw_of the way a hand-derived analytic form can."""
    out = np.zeros((3, 4))
    for k in range(4):
        qp = q.astype(float).copy()
        qm = q.astype(float).copy()
        qp[k] += eps
        qm[k] -= eps
        qp /= np.linalg.norm(qp)
        qm /= np.linalg.norm(qm)
        rp_, pp_ = roll_pitch_of(qp)
        rm_, pm_ = roll_pitch_of(qm)
        out[0, k] = wrap(rp_ - rm_) / (2 * eps)
        out[1, k] = (pp_ - pm_) / (2 * eps)
        out[2, k] = wrap(yaw_of(qp) - yaw_of(qm)) / (2 * eps)
    return out


# ------------------------------------------------------------------ error state
def error_state(exec_root_pos, exec_root_quat, ref_root_pos, ref_root_quat):
    """(T,3) accumulated error (dx, dy, dyaw) in the reference heading frame."""
    yaw = yaw_of(ref_root_quat)
    d = exec_root_pos - ref_root_pos
    dx = np.cos(-yaw) * d[:, 0] - np.sin(-yaw) * d[:, 1]
    dy = np.sin(-yaw) * d[:, 0] + np.cos(-yaw) * d[:, 1]
    dyaw = wrap(yaw_of(exec_root_quat) - yaw)
    return np.stack([dx, dy, dyaw], axis=1)
