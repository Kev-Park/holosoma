"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Example: {"L_Toe": [(30, 60)], "R_Toe": [(10, 20), (80, 95)]}"""

    z_floor: float = 0.0
    """Floor height used by Z pinning constraints."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class CoMStabilityConfig:
    """Configuration for the center-of-mass static-stability barrier constraint.

    Enforces that the ground projection of the robot CoM stays inside the support polygon
    (convex hull of the planted feet's contact spheres), in the discrete-time control
    barrier function form

        h(q) = b - A x_com(q) >= 0,    A x <= b the polygon in halfspace form
        h(q + dq) >= (1 - gamma) h(q)  =>  (A J_com) dq <= gamma * h(q)

    which linearises to one linear inequality per polygon edge in the SQP variable dq.
    Following ConstrainedMimic (arXiv:2606.00374) App. B.1, the constraint is enforced hard
    but relaxed with a penalised slack when the QP would otherwise be infeasible.
    """

    enable: bool = False
    """Whether to enforce the CoM stability barrier."""

    gamma: float = 0.5
    """Discrete-time CBF rate in (0, 1]. 1.0 = enforce h >= 0 outright; smaller values let
    the CoM approach the boundary more gradually and recover from an initial violation."""

    margin: float = 0.02
    """Shrink the support polygon inward by this many metres before enforcing (safety buffer)."""

    slack_penalty: float = 1e4
    """Penalty on the relaxation slack. Large => near-hard. Set <=0 for a strictly hard constraint."""

    rest_only: bool = False
    """If True, only enforce from ``rest_start_frame`` onward (static-stability is the wrong
    criterion during locomotion, which is deliberately statically unstable)."""

    rest_start_frame: int = -1
    """First frame index at which to enforce when ``rest_only``. -1 disables enforcement."""

    ramp_frames: int = 10
    """Frames over which to ramp gamma in before ``rest_start_frame``, avoiding a seam."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    com_stability: CoMStabilityConfig = field(default_factory=CoMStabilityConfig)
    """Configuration for the CoM static-stability barrier constraint."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
