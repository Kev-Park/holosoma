"""SQP objective term: penalise predicted SONIC root drift.

Adds  lambda * || e_i + (g0 + A @ dqa) * dt ||^2  at each frame, where g is the
learned drift-rate model and A = dg/dphi @ dphi/dqpos.

The term is an affine expression in dqa, so it stays a convex QP and needs no
change to the solver's structure -- it slots in beside the Laplacian, nominal
tracking, Q_diag and smoothness terms.

The model runs as plain numpy here, not torch: a 1-hidden-layer ELU net's
Jacobian is exact in closed form, and the solve loop calls this thousands of
times per clip, so autograd overhead is not worth paying.

State is carried frame to frame (prev_* and the accumulated error e), because
the features use BACKWARD differences and the error estimate is integrated
along the clip as the solver sweeps forward.
"""
import numpy as np

from . import drift_features as dfeat


class MLPNumpy:
    """1-hidden-layer ELU net with an exact analytic input-Jacobian."""

    def __init__(self, W1, b1, W2, b2, x_mu, x_sd, y_mu, y_sd):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.x_mu, self.x_sd = x_mu, x_sd
        self.y_mu, self.y_sd = y_mu, y_sd

    @staticmethod
    def from_torch(net, x_mu, x_sd, y_mu, y_sd):
        ls = [m for m in net if hasattr(m, "weight")]
        assert len(ls) == 2, "expected Linear-ELU-Linear"
        return MLPNumpy(ls[0].weight.detach().numpy(), ls[0].bias.detach().numpy(),
                        ls[1].weight.detach().numpy(), ls[1].bias.detach().numpy(),
                        x_mu, x_sd, y_mu, y_sd)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 x_mu=self.x_mu, x_sd=self.x_sd, y_mu=self.y_mu, y_sd=self.y_sd)

    @staticmethod
    def load(path):
        d = np.load(path)
        return MLPNumpy(d["W1"], d["b1"], d["W2"], d["b2"],
                        d["x_mu"], d["x_sd"], d["y_mu"], d["y_sd"])

    def forward_and_jac(self, x_raw):
        """-> g (dim_out,), dg/dx_raw (dim_out, dim_in), in raw units."""
        xs = (x_raw - self.x_mu) / self.x_sd
        z1 = self.W1 @ xs + self.b1
        h1 = np.where(z1 > 0, z1, np.expm1(np.minimum(z1, 0.0)))
        dh1 = np.where(z1 > 0, 1.0, np.exp(np.minimum(z1, 0.0)))
        out = self.W2 @ h1 + self.b2
        J = (self.W2 * dh1[None, :]) @ self.W1 / self.x_sd[None, :]
        return out * self.y_sd + self.y_mu, J * self.y_sd[:, None]


class DriftObjective:
    """Per-clip state + the cvxpy term for one frame.

    Usage inside the retargeter:
        obj = DriftObjective(model, lam, out_dim)
        obj.reset(q0, lf0, rf0)                       # before frame 0
        term = obj.cost_term(q, lf, rf, J_lf, J_rf, dqa, q_a_indices)
        ...  solve  ...
        obj.advance(q_solved, lf_solved, rf_solved)   # after each frame
    """

    def __init__(self, model, lam=1.0, e_idx=(1,), dt=dfeat.DT, max_e=1.0):
        """e_idx: which components of e = (dx, dy, dyaw) this model predicts.

        Default (1,) = dy ONLY. dx is deliberately excluded: measured clip-mean
        sign agreement for dx is 0.40, i.e. WORSE than chance, so including it
        would push the solver the wrong way on roughly 60% of clips.
        """
        self.model = model
        self.lam = float(lam)
        self.e_idx = np.asarray(e_idx, dtype=int)
        self.out_dim = len(self.e_idx)
        self.dt = dt
        self.max_e = max_e        # clamp: keep a diverging rollout from dominating
        self.prev = None
        self.e = np.zeros(3)

    def _pack(self, q, lf, rf):
        root, quat = q[0:3], q[3:7]
        yaw = dfeat.yaw_of(quat)
        R = dfeat.Rz_inv(yaw)
        roll, pitch = dfeat.roll_pitch_of(quat)

        def rel(f):
            d = f - root
            return np.concatenate([R @ d[:2], [d[2]]])

        return {"root": root.copy(), "lf": rel(lf), "rf": rel(rf),
                "roll": roll, "pitch": pitch, "yaw": yaw}

    def reset(self, q0, lf0, rf0):
        self.prev = self._pack(q0, lf0, rf0)
        self.e = np.zeros(3)

    def _features(self, q, lf, rf, J_lf, J_rf):
        p = self.prev
        return dfeat.features_and_jac(q, lf, rf, J_lf, J_rf,
                                      p["root"], p["lf"], p["rf"],
                                      p["roll"], p["pitch"], p["yaw"], dt=self.dt)

    def cost_term(self, q, lf, rf, J_lf, J_rf, dqa, q_a_indices, cp):
        """Convex quadratic in dqa, or None if the objective is inactive."""
        if self.prev is None or self.lam <= 0:
            return None
        phi, J_phi = self._features(q, lf, rf, J_lf, J_rf)
        x = np.concatenate([phi, self.e])
        g0, G = self.model.forward_and_jac(x)
        G_phi = G[:, :dfeat.FEATURE_DIM]          # e is constant within this frame
        A = G_phi @ J_phi                          # (out_dim, nq)
        A_a = A[:, q_a_indices] * self.dt
        c0 = self.e[self.e_idx] + g0 * self.dt
        return self.lam * cp.sum_squares(c0 + A_a @ dqa)

    def advance(self, q, lf, rf, J_lf, J_rf):
        """Integrate the error estimate and roll the backward-difference state."""
        if self.prev is not None:
            phi, _ = self._features(q, lf, rf, J_lf, J_rf)
            g0, _ = self.model.forward_and_jac(np.concatenate([phi, self.e]))
            self.e[self.e_idx] = np.clip(
                self.e[self.e_idx] + g0 * self.dt, -self.max_e, self.max_e)
        self.prev = self._pack(q, lf, rf)
        return self.e.copy()
