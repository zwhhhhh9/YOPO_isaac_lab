"""
PX4-style Attitude and Rate Controller (PyTorch Implementation)

Based on PX4 Autopilot:
- AttitudeControl: Nonlinear Quadrocopter Attitude Control (Brescianini et al., 2013)
- RateControl: PID angular rate control with feed forward

Author: Chengyong Lei
Affiliation: ZJU FAST Lab
Adapted from: PX4-Autopilot
Reference: https://github.com/PX4/PX4-Autopilot
"""

import torch
import math


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle tensors to [-pi, pi)."""
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert ZYX Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z]."""
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw

    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=1)


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to ZYX Euler angles (roll, pitch, yaw)."""
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(
        torch.abs(sin_pitch) >= 1.0,
        torch.sign(sin_pitch) * (torch.pi / 2.0),
        torch.asin(sin_pitch),
    )

    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    roll = wrap_to_pi(roll)
    yaw = wrap_to_pi(yaw)

    return torch.stack([roll, pitch, yaw], dim=1)


class QuadrotorDynamics:
    """
    Quadrotor dynamics model with motor thrust mapping and allocation matrix.
    Based on Crazyflie 2.0 parameters.
    """

    def __init__(self,
                 num_envs: int = 1,
                 mass: torch.Tensor = None,
                 inertia: torch.Tensor = None,
                 arm_l: torch.Tensor = None,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.num_envs = num_envs

        # Initialize mass
        if mass is None:
            self.mass_ = torch.full((num_envs,), 0.027, device=self.device)
        else:
            self.mass_ = mass.to(self.device)
            if self.mass_.dim() == 0:
                self.mass_ = self.mass_.expand(num_envs)

        # Initialize arm length
        if arm_l is None:
            self.arm_l_ = torch.full((num_envs,), 0.046, device=self.device)
        else:
            self.arm_l_ = arm_l.to(self.device)
            if self.arm_l_.dim() == 0:
                self.arm_l_ = self.arm_l_.expand(num_envs)

        # Initialize inertia
        default_inertia = torch.tensor([9.19e-6, 9.19e-6, 22.8e-6], device=self.device)
        if inertia is None:
            self.inertia_ = default_inertia.expand(num_envs, 3)
        else:
            if inertia.dim() == 1:
                self.inertia_ = inertia.to(self.device).expand(num_envs, 3)
            elif inertia.dim() == 2:
                self.inertia_ = inertia.to(self.device)
            else:
                raise ValueError(f"Inertia must be tensor of shape (3,) or (num_envs, 3), got {inertia.shape}")

        # Motor limits
        kv_rpm_per_v = 10000.0
        kt = 60.0 / (2 * math.pi * kv_rpm_per_v)
        self.motor_tau_inv_ = torch.full((num_envs,), 1.0 / kt, device=self.device)

        rpm_max, rpm_min = 24000.0, 1200.0
        self.motor_omega_max_ = torch.full((num_envs,), rpm_max * math.pi / 30, device=self.device)
        self.motor_omega_min_ = torch.full((num_envs,), rpm_min * math.pi / 30, device=self.device)

        # Thrust map coefficients: T = a*omega^2 + b*omega + c
        self.thrust_map_ = torch.tensor([9.96063125e-08, -2.55003087e-05, 5.84422691e-03],
                                        device=self.device).expand(num_envs, 3)

        # Drag-torque ratio (yaw moment per unit thrust)
        self.kappa_ = torch.full((num_envs,), 0.005964552, device=self.device)

        # Thrust limits
        self.thrust_min_ = torch.zeros(num_envs, device=self.device)
        a, b, c = self.thrust_map_[:, 0], self.thrust_map_[:, 1], self.thrust_map_[:, 2]
        w = self.motor_omega_max_
        self.thrust_max_ = a * (w ** 2) + b * w + c # 0.857 N

        # Maximum torque calculation parameter
        # This defines what fraction of thrust_max is available for torque generation
        # When hovering at 50% throttle, available thrust headroom = 0.5 * thrust_max per motor
        # Default: 0.375 (assumes hover at ~62.5% throttle, conservative for stability)
        self.thrust_ratio = 0.375  # Configurable via set_thrust_ratio()

        self.updateInertiaMatrix()

    def updateInertiaMatrix(self):
        """Build allocation matrix for X-configuration."""
        # t_BM: mapping matrix for X-configuration, shape (num_envs, 3, 4)
        base_mat = torch.tensor([[1, -1, -1, 1],
                                 [-1, -1, 1, 1],
                                 [0, 0, 0, 0]], dtype=torch.float, device=self.device)
        factor = self.arm_l_ * math.sqrt(0.5)
        self.t_BM_ = factor.view(-1, 1, 1) * base_mat

        # Construct the diagonal inertia matrix J_ and its inverse
        self.J_ = torch.diag_embed(self.inertia_)
        self.J_inv_ = torch.diag_embed(1.0 / self.inertia_)

    def set_thrust_ratio(self, ratio: float):
        """
        Set the thrust ratio available for torque generation.

        This defines what fraction of thrust_max can be used for differential thrust.

        Args:
            ratio: Thrust ratio for torque [0.0, 1.0]
                   - 0.25: Conservative (assumes hover at 75% throttle)
                   - 0.375: Default (assumes hover at ~62.5% throttle)
                   - 0.5: Balanced (assumes hover at 50% throttle)
                   - 1.0: Aggressive (full motor range available)

        Example:
            If hovering requires 50% throttle per motor:
            - Each motor has 0.5*thrust_max headroom
            - Set ratio = 0.5 for accurate max torque calculation
        """
        self.thrust_ratio = max(0.0, min(1.0, ratio))

    def get_max_torque(self):
        """
        Get maximum torque based on motor limits for X-configuration.

        For X-configuration quadrotor with thrust headroom constraints:
        - Roll/Pitch: Two diagonal motors increase thrust, two decrease
          Max differential = 2 × (thrust_ratio × thrust_max) × lever_arm
        - Yaw: Two motors increase, two decrease
          Max differential = 2 × (thrust_ratio × thrust_max) × kappa

        The thrust_ratio parameter accounts for the thrust headroom available.
        For example, if hovering at 50% throttle, only 50% of thrust_max is available
        for differential thrust generation.

        Returns:
            max_torque: (num_envs, 3) maximum torque [τ_roll, τ_pitch, τ_yaw] in Nm
        """
        # Available thrust per motor for torque generation
        thrust_available = self.thrust_max_ * self.thrust_ratio

        # Lever arm factor for X-configuration (45° motor placement)
        factor = self.arm_l_ * math.sqrt(0.5)

        # Roll/Pitch max torque: 2 motors increase, 2 decrease
        # τ_max = 2 × thrust_available × lever_arm
        max_torque_x = thrust_available * factor * 2
        max_torque_y = thrust_available * factor * 2

        # Yaw max torque: 2 motors increase, 2 decrease
        # τ_yaw_max = 2 × thrust_available × kappa
        max_torque_z = thrust_available * self.kappa_ * 2

        return torch.stack([max_torque_x, max_torque_y, max_torque_z], dim=1)

    def clampThrust(self, thrusts: torch.Tensor):
        """Clamp thrust to physical limits."""
        t_min = self.thrust_min_.unsqueeze(1)
        t_max = self.thrust_max_.unsqueeze(1)
        return torch.max(torch.min(thrusts, t_max), t_min)

    def motorOmegaToThrust(self, omega: torch.Tensor):
        """Convert motor angular velocity to thrust."""
        a = self.thrust_map_[:, 0].unsqueeze(1)
        b = self.thrust_map_[:, 1].unsqueeze(1)
        c = self.thrust_map_[:, 2].unsqueeze(1)
        return a * (omega ** 2) + b * omega + c

    def motorThrustToOmega(self, thrusts: torch.Tensor):
        """
        Convert thrust to motor angular velocity using quadratic formula.

        Solves: a*omega^2 + b*omega + c = thrust
        for omega using the quadratic formula.

        Args:
            thrusts: (num_envs, 4) motor thrust values [N]

        Returns:
            omega: (num_envs, 4) motor angular velocities [rad/s]
        """
        a = self.thrust_map_[:, 0].unsqueeze(1)
        b = self.thrust_map_[:, 1].unsqueeze(1)
        c = self.thrust_map_[:, 2].unsqueeze(1)

        # Solve quadratic: a*omega^2 + b*omega + (c - thrust) = 0
        # omega = (-b + sqrt(b^2 - 4*a*(c - thrust))) / (2*a)
        discriminant = b ** 2 - 4.0 * a * (c - thrusts)

        # For numerical stability, clamp negative discriminants to 0
        # Negative discriminant means thrust is below minimum achievable or numerical error
        # In this case, omega will be: -b / (2*a) which is the minimum omega
        discriminant_safe = torch.clamp(discriminant, min=0.0)
        omega = (-b + torch.sqrt(discriminant_safe)) / (2.0 * a)

        return omega

    def getAllocationMatrix(self):
        """
        Get control allocation matrix for X-configuration.
        Maps [total_thrust, roll_torque, pitch_torque, yaw_torque] -> [T1, T2, T3, T4]
        """
        num_envs = self.mass_.shape[0]
        ones_row = torch.ones(num_envs, 4, device=self.device)
        alloc = torch.stack([
            ones_row,
            self.t_BM_[:, 0, :],
            self.t_BM_[:, 1, :],
            self.kappa_.unsqueeze(1) * torch.tensor([1, -1, 1, -1], device=self.device).expand(num_envs, 4)
        ], dim=1)
        return alloc

    def computeControlAllocationScale(self, mix_pinv: torch.Tensor):
        """
        Compute control allocation scale following PX4 normalization.

        This computes the scaling factors to normalize the control allocation matrix,
        making PID parameters independent of vehicle size/inertia.

        Based on PX4: ControlAllocationPseudoInverse::updateControlAllocationMatrixScale()

        Args:
            mix_pinv: (num_envs, 4, 4) pseudo-inverse of allocation matrix

        Returns:
            control_allocation_scale: (num_envs, 4) scaling factors [thrust, roll, pitch, yaw]
        """
        num_envs = mix_pinv.shape[0]
        scale = torch.ones(num_envs, 4, device=self.device)

        # For Roll and Pitch: normalize by RMS of effectiveness per active motor
        # This ensures same normalized torque produces similar angular acceleration regardless of size

        # Count non-zero motors for roll (threshold: 1e-3)
        # Note: mix_pinv column order is [thrust, roll, pitch, yaw] matching alloc_matrix row order
        num_motors_roll = (torch.abs(mix_pinv[:, :, 1]) > 1e-3).sum(dim=1).float()
        # Compute RMS normalization: sqrt(||column||^2 / (N_motors / 2))
        roll_norm_squared = torch.sum(mix_pinv[:, :, 1] ** 2, dim=1)
        roll_scale = torch.sqrt(roll_norm_squared / (num_motors_roll / 2.0 + 1e-6))

        # Same for pitch
        num_motors_pitch = (torch.abs(mix_pinv[:, :, 2]) > 1e-3).sum(dim=1).float()
        pitch_norm_squared = torch.sum(mix_pinv[:, :, 2] ** 2, dim=1)
        pitch_scale = torch.sqrt(pitch_norm_squared / (num_motors_pitch / 2.0 + 1e-6))

        # Use max of roll and pitch scale for both axes (PX4 behavior)
        roll_pitch_scale = torch.maximum(roll_scale, pitch_scale)

        # Thrust (column 0): typically not normalized
        scale[:, 0] = 1.0

        # Roll and Pitch (columns 1, 2)
        scale[:, 1] = roll_pitch_scale  # Roll
        scale[:, 2] = roll_pitch_scale  # Pitch

        # Yaw (column 3): use maximum absolute value in yaw column
        yaw_scale = torch.max(torch.abs(mix_pinv[:, :, 3]), dim=1)[0]
        scale[:, 3] = yaw_scale

        return scale


class AttitudeControl:
    """
    Quaternion-based attitude controller.

    Based on PX4's AttitudeControl implementing the paper:
    "Nonlinear Quadrocopter Attitude Control" by Brescianini, Hehn, D'Andrea (2013)
    """

    def __init__(self,
                 num_envs: int = 1,
                 device: torch.device = torch.device("cpu"),
                 proportional_gain: torch.Tensor = None,
                 yaw_weight: float = 0.4,
                 rate_limit: torch.Tensor = None):
        """
        Initialize attitude controller.

        Args:
            num_envs: Number of parallel environments
            device: PyTorch device
            proportional_gain: (3,) tensor [Kp_roll, Kp_pitch, Kp_yaw]
            yaw_weight: Weight for yaw control [0,1], lower = deprioritize yaw
            rate_limit: (3,) tensor [max_roll_rate, max_pitch_rate, max_yaw_rate] in rad/s
        """
        self.num_envs = num_envs
        self.device = device

        # Default PX4 gains (typical for racing drones)
        if proportional_gain is None:
            proportional_gain = torch.tensor([6.5, 6.5, 2.8], device=device)

        if rate_limit is None:
            rate_limit = torch.tensor([3.0, 3.0, 1.5], device=device)  # rad/s

        self.set_proportional_gain(proportional_gain, yaw_weight)
        self.rate_limit = rate_limit.to(device)

        # Attitude setpoint (identity quaternion by default)
        self.attitude_setpoint_q = torch.zeros(num_envs, 4, device=device)
        self.attitude_setpoint_q[:, 0] = 1.0  # [w, x, y, z]
        self.yawspeed_setpoint = torch.zeros(num_envs, device=device)

    def set_proportional_gain(self, proportional_gain: torch.Tensor, yaw_weight: float):
        """Set P gains with yaw weight."""
        self.proportional_gain = proportional_gain.to(self.device)
        self.yaw_w = torch.clamp(torch.tensor(yaw_weight, device=self.device), 0.0, 1.0)

        # Compensate for yaw weight rescaling
        if self.yaw_w > 1e-4:
            self.proportional_gain[2] = self.proportional_gain[2] / self.yaw_w

    def set_attitude_setpoint(self, qd: torch.Tensor, yawspeed_setpoint: torch.Tensor = None):
        """
        Set attitude setpoint.

        Args:
            qd: (num_envs, 4) desired quaternion [w, x, y, z]
            yawspeed_setpoint: (num_envs,) yaw rate feed-forward in world frame
        """
        self.attitude_setpoint_q = qd / torch.norm(qd, dim=1, keepdim=True)
        if yawspeed_setpoint is not None:
            self.yawspeed_setpoint = yawspeed_setpoint

    def update(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute attitude control (main update loop).

        Args:
            q: (num_envs, 4) current attitude quaternion [w, x, y, z]

        Returns:
            rate_setpoint: (num_envs, 3) desired body rates [p, q, r] in rad/s
        """
        qd = self.attitude_setpoint_q

        # Step 1: Calculate reduced desired attitude (prioritize roll/pitch over yaw)
        e_z = self._quat_to_dcm_z(q)  # Current body z-axis in world frame
        e_z_d = self._quat_to_dcm_z(qd)  # Desired body z-axis in world frame
        qd_red = self._quat_from_two_vectors(e_z, e_z_d)

        # Check for singularity (opposite thrust directions)
        singularity = (torch.abs(qd_red[:, 1]) > (1.0 - 1e-5)) | (torch.abs(qd_red[:, 2]) > (1.0 - 1e-5))
        qd_red = torch.where(singularity.unsqueeze(1), qd, qd_red)

        # Transform to world frame reduced attitude
        qd_red = self._quat_mul(qd_red, q)

        # Step 2: Mix full and reduced desired attitude based on yaw weight
        q_mix = self._quat_mul(self._quat_inv(qd_red), qd)
        q_mix = self._quat_canonical(q_mix)

        # Constrain for numerical stability
        q_mix[:, 0] = torch.clamp(q_mix[:, 0], -1.0, 1.0)
        q_mix[:, 3] = torch.clamp(q_mix[:, 3], -1.0, 1.0)

        # Apply yaw weight
        yaw_w_acos = self.yaw_w * torch.acos(q_mix[:, 0])
        yaw_w_asin = self.yaw_w * torch.asin(q_mix[:, 3])
        q_yaw = torch.stack([
            torch.cos(yaw_w_acos),
            torch.zeros_like(yaw_w_acos),
            torch.zeros_like(yaw_w_acos),
            torch.sin(yaw_w_asin)
        ], dim=1)
        qd = self._quat_mul(qd_red, q_yaw)

        # Step 3: Calculate attitude error quaternion
        qe = self._quat_mul(self._quat_inv(q), qd)

        # Step 4: Convert to rotation vector (scaled axis-angle)
        # Use canonical form only for extraction, don't modify qe itself (matching PX4)
        eq = 2.0 * self._quat_canonical(qe)[:, 1:]  # [x, y, z] components

        # Step 5: Proportional control
        rate_setpoint = eq * self.proportional_gain.unsqueeze(0)

        # Step 6: Add yaw rate feed-forward (transform from world to body frame)
        if torch.any(torch.isfinite(self.yawspeed_setpoint)):
            z_body = self._quat_to_dcm_z(self._quat_inv(q))
            rate_setpoint += z_body * self.yawspeed_setpoint.unsqueeze(1)

        # Step 7: Rate limiting
        rate_setpoint = torch.clamp(rate_setpoint, -self.rate_limit, self.rate_limit)

        return rate_setpoint

    # ===== Quaternion utilities =====

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication [w, x, y, z]."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=1)

    def _quat_inv(self, q: torch.Tensor) -> torch.Tensor:
        """Quaternion inverse (conjugate for unit quaternions)."""
        return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)

    def _quat_canonical(self, q: torch.Tensor) -> torch.Tensor:
        """Ensure quaternion is in canonical form (w >= 0)."""
        return torch.where((q[:, 0:1] < 0).expand_as(q), -q, q)

    def _quat_to_dcm_z(self, q: torch.Tensor) -> torch.Tensor:
        """Extract z-axis (3rd column) of rotation matrix from quaternion."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        return torch.stack([
            2 * (w * y + x * z),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ], dim=1)

    def _quat_from_two_vectors(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Quaternion representing rotation from vector a to vector b.
        Shortest path rotation.
        """
        # Normalize vectors
        a = a / torch.norm(a, dim=1, keepdim=True)
        b = b / torch.norm(b, dim=1, keepdim=True)

        # Cross and dot products
        cr = torch.cross(a, b, dim=1)
        dt = torch.sum(a * b, dim=1)

        # Build quaternion [w, x, y, z]
        w = torch.sqrt((1.0 + dt) * 0.5)
        xyz = cr / (2.0 * w.unsqueeze(1))

        return torch.cat([w.unsqueeze(1), xyz], dim=1)


class RateControl:
    """
    PID angular rate controller with feed-forward and anti-windup.

    Based on PX4's RateControl with:
    - Proportional term
    - Integral term with anti-windup
    - Derivative term (using angular acceleration)
    - Feed-forward term
    """

    def __init__(self,
                 num_envs: int = 1,
                 device: torch.device = torch.device("cpu"),
                 gain_p: torch.Tensor = None,
                 gain_i: torch.Tensor = None,
                 gain_d: torch.Tensor = None,
                 gain_ff: torch.Tensor = None,
                 gain_k: torch.Tensor = None,
                 integrator_limit: torch.Tensor = None,
                 inertia: torch.Tensor = None):
        """
        Initialize rate controller with PHYSICAL TORQUE output.

        Args:
            num_envs: Number of parallel environments
            device: PyTorch device
            gain_p: (3,) P gains [roll, pitch, yaw] in [Nm/(rad/s)]
            gain_i: (3,) I gains in [Nm/rad]
            gain_d: (3,) D gains in [Nm/(rad/s^2)]
            gain_ff: (3,) Feed-forward gains
            gain_k: (3,) K gains for PX4 parameter compatibility (converts parallel to ideal form)
            integrator_limit: (3,) Maximum integral torque values [Nm]
            inertia: (3,) or (num_envs, 3) Inertia tensor [Ixx, Iyy, Izz] in kg*m^2

        Note:
            Output is PHYSICAL TORQUE in Nm, not normalized values.
            Gains should be tuned in physical units.
        """
        self.num_envs = num_envs
        self.device = device

        # Default inertia (Crazyflie 2.0)
        if inertia is None:
            inertia = torch.tensor([9.19e-6, 9.19e-6, 22.8e-6], device=device)

        if inertia.dim() == 1:
            self.inertia = inertia.to(device).expand(num_envs, 3)
        else:
            self.inertia = inertia.to(device)

        # Default PX4 K gains (typically 1.0 for direct parameter usage)
        if gain_k is None:
            gain_k = torch.tensor([1.0, 1.0, 1.0], device=device)

        gain_k = gain_k.to(device)

        # Default PX4 gains (typical for Crazyflie, converted to physical units)
        # These will be modulated by K gains and inertia
        if gain_p is None:
            gain_p = torch.tensor([0.15, 0.15, 0.18], device=device)
        if gain_i is None:
            gain_i = torch.tensor([0.3, 0.3, 0.3], device=device)
        if gain_d is None:
            gain_d = torch.tensor([0.003, 0.003, 0.0], device=device)
        if gain_ff is None:
            gain_ff = torch.tensor([0.0, 0.0, 0.0], device=device)
        if integrator_limit is None:
            integrator_limit = torch.tensor([0.3, 0.3, 0.3], device=device)

        # Apply K gain modulation (PX4 compatibility)
        # Keep gains in normalized space, NOT multiplied by inertia
        # DO NOT unsqueeze here - will be done in update() method
        self.gain_p = (gain_p * gain_k).to(device)  # shape: (3,)
        self.gain_i = (gain_i * gain_k).to(device)  # shape: (3,)
        self.gain_d = (gain_d * gain_k).to(device)  # shape: (3,)
        self.gain_ff = (gain_ff * gain_k).to(device)  # shape: (3,)
        self.lim_int = (integrator_limit * gain_k).to(device)  # shape: (3,)

        # Integral state (physical torque)
        self.rate_int = torch.zeros(num_envs, 3, device=device)

        # Saturation flags (from control allocator)
        self.saturation_positive = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)
        self.saturation_negative = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)

    def set_saturation_status(self, saturation_pos: torch.Tensor, saturation_neg: torch.Tensor):
        """Set saturation flags from control allocator."""
        self.saturation_positive = saturation_pos
        self.saturation_negative = saturation_neg

    def update(self,
               rate: torch.Tensor,
               rate_sp: torch.Tensor,
               angular_accel: torch.Tensor,
               dt: float,
               landed: torch.Tensor = None) -> torch.Tensor:
        """
        Rate control update - outputs PHYSICAL TORQUE.

        Args:
            rate: (num_envs, 3) current angular rates [p, q, r] in body frame [rad/s]
            rate_sp: (num_envs, 3) desired angular rates [rad/s]
            angular_accel: (num_envs, 3) angular acceleration [rad/s^2] (for D term)
            dt: time step [s]
            landed: (num_envs,) boolean, True if landed (disables integral)

        Returns:
            torque: (num_envs, 3) PHYSICAL torque command [Nm] (not normalized)
        """
        # Rate error
        rate_error = rate_sp - rate

        # PID control with feed-forward - ALL IN PHYSICAL TORQUE
        # torque [Nm] = P[Nm/(rad/s)] * error[rad/s] + I_accumulated[Nm]
        #               - D[Nm/(rad/s^2)] * accel[rad/s^2] + FF * setpoint
        torque = (self.gain_p.unsqueeze(0) * rate_error +
                  self.rate_int -
                  self.gain_d.unsqueeze(0) * angular_accel +
                  self.gain_ff.unsqueeze(0) * rate_sp)

        # Update integral (only if not landed)
        if landed is None:
            landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._update_integral(rate_error, dt, ~landed)

        return torque

    def _update_integral(self, rate_error: torch.Tensor, dt: float, update_mask: torch.Tensor):
        """Update integral with anti-windup.

        Following PX4 logic:
        1. Apply anti-windup limiting to rate_error
        2. Calculate i_factor using the LIMITED rate_error
        3. Integrate using the LIMITED rate_error
        """
        # Anti-windup: prevent integral growth when saturated
        # Directly modify rate_error following PX4's approach
        rate_error_modified = rate_error.clone()

        # Prevent further positive saturation
        rate_error_modified = torch.where(
            self.saturation_positive,
            torch.minimum(rate_error_modified, torch.zeros_like(rate_error_modified)),
            rate_error_modified
        )

        # Prevent further negative saturation
        rate_error_modified = torch.where(
            self.saturation_negative,
            torch.maximum(rate_error_modified, torch.zeros_like(rate_error_modified)),
            rate_error_modified
        )

        # I-factor: reduce I gain with large rate errors (prevents overshoot after flips)
        # Formula from PX4: i_factor = 1 - (error / 400deg)^2
        # Use the LIMITED rate_error (after anti-windup) matching PX4 logic
        i_factor = rate_error_modified / math.radians(400.0)
        i_factor = torch.maximum(torch.zeros_like(i_factor), 1.0 - i_factor * i_factor)

        # Integrate using LIMITED rate_error
        rate_i = self.rate_int + i_factor * self.gain_i.unsqueeze(0) * rate_error_modified * dt

        # Clamp and update (only where update_mask is True)
        rate_i = torch.clamp(rate_i, -self.lim_int.unsqueeze(0), self.lim_int.unsqueeze(0))
        self.rate_int = torch.where(update_mask.unsqueeze(1), rate_i, self.rate_int)

    def reset_integral(self, env_ids: torch.Tensor = None):
        """
        Reset integral term to zero.

        Args:
            env_ids: (N,) tensor of environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            self.rate_int.zero_()
        else:
            self.rate_int[env_ids] = 0.0


class PX4QuadrotorController:
    """
    PX4-style quadrotor controller with attitude and rate control modes.
    Supports both attitude control and direct rate control modes.
    Outputs motor speeds based on PX4 mixer.
    """

    def __init__(self,
                 num_envs: int = 1,
                 device: torch.device = torch.device("cpu"),
                 # Attitude control params
                 att_p_gain: torch.Tensor = None,
                 att_yaw_weight: float = 0.4,
                 att_rate_limit: torch.Tensor = None,
                 # Rate control params
                 rate_p_gain: torch.Tensor = None,
                 rate_i_gain: torch.Tensor = None,
                 rate_d_gain: torch.Tensor = None,
                 rate_k_gain: torch.Tensor = None,
                 rate_int_limit: torch.Tensor = None,
                 # Dynamics params
                 mass: torch.Tensor = None,
                 inertia: torch.Tensor = None,
                 arm_length: torch.Tensor = None):
        """
        Initialize PX4 controller.

        Args:
            num_envs: Number of parallel environments
            device: PyTorch device
            att_p_gain: Attitude P gains [roll, pitch, yaw]
            att_yaw_weight: Yaw priority weight [0,1]
            att_rate_limit: Max body rates [rad/s]
            rate_p_gain: Rate P gains
            rate_i_gain: Rate I gains
            rate_d_gain: Rate D gains
            rate_k_gain: Rate K gains (PX4 parallel-to-ideal form conversion)
            rate_int_limit: Rate integrator limits
            mass: Quadrotor mass (kg)
            inertia: Inertia tensor [Ixx, Iyy, Izz] (kg*m^2)
            arm_length: Motor arm length (m)
        """
        self.num_envs = num_envs
        self.device = device

        # Initialize dynamics model
        self.dynamics = QuadrotorDynamics(
            num_envs=num_envs,
            mass=mass,
            inertia=inertia,
            arm_l=arm_length,
            device=device
        )

        # Initialize sub-controllers
        self.attitude_control = AttitudeControl(
            num_envs=num_envs,
            device=device,
            proportional_gain=att_p_gain,
            yaw_weight=att_yaw_weight,
            rate_limit=att_rate_limit
        )

        self.rate_control = RateControl(
            num_envs=num_envs,
            device=device,
            gain_p=rate_p_gain,
            gain_i=rate_i_gain,
            gain_d=rate_d_gain,
            gain_k=rate_k_gain,
            gain_ff=torch.tensor([0.0, 0.0, 0.0], device=device),
            integrator_limit=rate_int_limit,
            inertia=inertia  # Pass inertia to rate controller
        )

        # Precompute allocation matrices
        self.alloc_matrix_ = self.dynamics.getAllocationMatrix()
        self.alloc_matrix_pinv_ = torch.linalg.pinv(self.alloc_matrix_)

        # Initialize state variables for angular acceleration computation
        self.prev_omega_body = torch.zeros(num_envs, 3, device=device)
        self.prev_angular_accel = torch.zeros(num_envs, 3, device=device)
        self.accel_filter_coef = 0.5

        # NO LONGER using normalized pseudo-inverse - use raw physical allocation
        print(f"[PX4 Controller] Using PHYSICAL TORQUE control (no normalization)")
        print(f"[PX4 Controller] Initialized with {num_envs} environments")

    def _normalizeControlAllocationMatrix(self, mix_pinv: torch.Tensor, scale: torch.Tensor):
        """
        Normalize the control allocation pseudo-inverse matrix.

        Based on PX4: ControlAllocationPseudoInverse::normalizeControlAllocationMatrix()

        This divides each column by its corresponding scale factor, making the
        allocation matrix independent of vehicle size/inertia.

        Args:
            mix_pinv: (num_envs, 4, 4) pseudo-inverse matrix
            scale: (num_envs, 4) scaling factors [thrust, roll, pitch, yaw]

        Returns:
            mix_pinv_normalized: (num_envs, 4, 4) normalized pseudo-inverse
        """
        mix_normalized = mix_pinv.clone()

        # Thrust column (column 0): typically not normalized in PX4
        # mix_normalized[:, :, 0] remains unchanged

        # Normalize roll and pitch columns (columns 1, 2)
        if torch.all(scale[:, 1] > 1e-6):
            mix_normalized[:, :, 1] = mix_normalized[:, :, 1] / scale[:, 1].unsqueeze(1)
        
        if torch.all(scale[:, 2] > 1e-6):
            mix_normalized[:, :, 2] = mix_normalized[:, :, 2] / scale[:, 2].unsqueeze(1)

        # Normalize yaw column (column 3)
        if torch.all(scale[:, 3] > 1e-6):
            mix_normalized[:, :, 3] = mix_normalized[:, :, 3] / scale[:, 3].unsqueeze(1)

        # Set small elements to 0 to avoid numerical issues
        mix_normalized[torch.abs(mix_normalized) < 1e-3] = 0.0

        return mix_normalized

    def set_accel_filter_coef(self, coef: float):
        """
        Set the low-pass filter coefficient for angular acceleration.

        Args:
            coef: Filter coefficient in range [0, 1]
                  0 = no filtering (use raw derivative)
                  1 = maximum smoothing (heavily filtered)
                  Typical values: 0.3-0.7 for moderate noise reduction

        Note:
            Higher values provide more smoothing but introduce phase lag.
            Lower values are more responsive but noisier.
        """
        self.accel_filter_coef = max(0.0, min(1.0, coef))
        print(f"[PX4 Controller] Angular acceleration filter coefficient set to: {self.accel_filter_coef}")

    def compute_control(self,
                       state: torch.Tensor,
                       cmd: torch.Tensor,
                       dt: float = 0.005,
                       mode: str = 'attitude') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Main control loop with two modes: attitude control or rate control.

        Args:
            state: (num_envs, 19+) tensor containing:
                   [0:3] position, [3:7] quaternion [w,x,y,z],
                   [7:10] velocity, [10:13] angular velocity (world frame),
                   [13:16] body torque, [16:19] linear acceleration
            cmd: Command tensor, format depends on mode:
                 - 'attitude': (num_envs, 4) [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd] normalized
                 - 'rate': (num_envs, 4) [roll_rate, pitch_rate, yaw_rate, thrust_cmd] in rad/s and normalized thrust
            dt: timestep
            mode: Control mode - 'attitude' or 'rate'

        Returns:
            force: (num_envs, 3) body frame force [Fx, Fy, Fz]
            torque: (num_envs, 3) body frame torque [Tx, Ty, Tz]
            motor_speeds: (num_envs, 4) motor angular velocities [ω1, ω2, ω3, ω4] in rad/s
            info: dict containing intermediate control values:
                  - 'omega_body': (num_envs, 3) current body rates [p, q, r]
                  - 'rate_sp': (num_envs, 3) desired body rates [p, q, r]
                  - 'rate_error': (num_envs, 3) rate tracking error
                  - 'angular_accel': (num_envs, 3) angular acceleration
                  - 'tau_des': (num_envs, 3) desired normalized torque
                  - 'motor_thrusts': (num_envs, 4) individual motor thrusts
                  - 'q_des': (num_envs, 4) desired quaternion (attitude mode only)
                  - 'thrust_cmd': (num_envs,) commanded thrust
        """
        batch = state.shape[0]

        # Parse state
        q_cur = state[:, 3:7]  # Current quaternion (body-to-world)
        omega_world = state[:, 10:13]  # Angular velocity in world frame

        # Convert angular velocity to body frame
        omega_body = self._quat_rotate_inverse(q_cur, omega_world)

        # Get thrust command (same for both modes)
        thrust_cmd = cmd[:, 3]  # Normalized [0, 1]
        F_des = thrust_cmd * self.dynamics.thrust_max_ * self.dynamics.thrust_ratio * 4.0

        # Get desired body rates based on control mode
        if mode == 'attitude':
            # Attitude control mode: convert attitude commands to rate setpoints
            roll_des = cmd[:, 0]
            pitch_des = cmd[:, 1]
            yaw_des = cmd[:, 2]

            q_des = self._quat_from_euler(roll_des, pitch_des, yaw_des)
            self.attitude_control.set_attitude_setpoint(q_des)
            rate_sp = self.attitude_control.update(q_cur)

        elif mode == 'rate':
            # Rate control mode: use rate commands directly
            rate_sp = cmd[:, 0:3].clone()  # [roll_rate, pitch_rate, yaw_rate] in rad/s

        else:
            raise ValueError(f"Unknown control mode: {mode}. Use 'attitude' or 'rate'.")

        # === Compute angular acceleration (derivative of angular velocity) ===
        # Use finite difference: α = (ω_current - ω_previous) / dt
        angular_accel_raw = (omega_body - self.prev_omega_body) / dt

        # Apply low-pass filter to reduce noise
        # First-order IIR filter: α_filtered = (1-α) * α_new + α * α_old
        # This reduces high-frequency noise in the derivative term
        angular_accel = ((1.0 - self.accel_filter_coef) * angular_accel_raw +
                        self.accel_filter_coef * self.prev_angular_accel)

        # Store current values for next iteration
        self.prev_omega_body = omega_body.clone()
        self.prev_angular_accel = angular_accel.clone()

        # === Rate control: rate_sp -> torque ===
        # tau_des is now PHYSICAL TORQUE in Nm
        tau_des = self.rate_control.update(
            rate=omega_body,
            rate_sp=rate_sp,
            angular_accel=angular_accel,
            dt=dt,
            landed=None
        )

        # === Control allocation with motor dynamics ===
        # Step 1: Compute baseline thrust
        T0 = F_des / 4.0

        # Step 2: Calculate differential thrust bounds for torque generation
        # T_motor_max here is the available headroom for differential thrust
        T_motor_max = self.dynamics.thrust_max_ * self.dynamics.thrust_ratio
        T_motor_min = self.dynamics.thrust_min_
        delta_T_max_pos = (T_motor_max - T0).clamp(min=0.0)
        delta_T_max_neg = (T0 - T_motor_min).clamp(min=0.0)

        # Step 3: Control allocation using PHYSICAL TORQUE (NO normalization)
        # Build control wrench: [thrust, roll_torque, pitch_torque, yaw_torque]
        # tau_des is PHYSICAL TORQUE in Nm from rate controller
        torque_only_wrench = torch.cat([
            torch.zeros_like(F_des).unsqueeze(1),  # thrust = 0 (differential only)
            tau_des  # PHYSICAL torque [Nm]
        ], dim=1)

        # Use RAW pseudo-inverse for control allocation (no normalization)
        delta_T = torch.bmm(self.alloc_matrix_pinv_, torque_only_wrench.unsqueeze(-1)).squeeze(-1)

        # Step 4: Scale differential thrust to respect differential thrust bounds
        # This scaling ensures differential thrust stays within the available headroom
        # defined by thrust_ratio (e.g., if hovering at 50% throttle, only 50% headroom available)
        # Add epsilon to avoid division by zero when thrust is near saturation
        eps = 1e-6
        pos_mask = delta_T > 0
        neg_mask = delta_T < 0

        # Safe division: clamp denominators to avoid division by very small numbers
        delta_T_max_pos_safe = torch.clamp(delta_T_max_pos.unsqueeze(1), min=eps)
        delta_T_max_neg_safe = torch.clamp(delta_T_max_neg.unsqueeze(1), min=eps)

        scale_factors_pos = torch.where(pos_mask, delta_T / delta_T_max_pos_safe,
                                        torch.zeros_like(delta_T))
        scale_factors_neg = torch.where(neg_mask, -delta_T / delta_T_max_neg_safe,
                                        torch.zeros_like(delta_T))

        max_scale_per_batch, _ = torch.max(torch.maximum(scale_factors_pos, scale_factors_neg), dim=1, keepdim=True)

        delta_T_scaling = torch.where(
            max_scale_per_batch > 1.0,
            1.0 / max_scale_per_batch,
            torch.ones_like(max_scale_per_batch)
        )
        delta_T = delta_T * delta_T_scaling

        # Step 5: Combine baseline and differential thrust
        motor_thrusts = T0.unsqueeze(1) + delta_T

        # Step 6: Enforce non-negative thrust and zero throttle handling
        motor_thrusts = torch.clamp(motor_thrusts, min=0.0)

        # Zero throttle → zero thrust
        zero_throttle = (cmd[:, 3] == 0.0).unsqueeze(1)
        motor_thrusts = torch.where(zero_throttle, torch.zeros_like(motor_thrusts), motor_thrusts)

        # Step 7: Limit per-motor maximum thrust to absolute physical limit
        # This second-stage scaling ensures no motor exceeds thrust_max, even when
        # baseline thrust T0 is high. The first scaling (Step 4) only ensures differential
        # thrust respects the headroom defined by thrust_ratio.
        thrust_max_expanded = T_motor_max.unsqueeze(1).expand_as(motor_thrusts)
        scale_factors = motor_thrusts / thrust_max_expanded
        max_scale_per_batch, _ = torch.max(scale_factors, dim=1, keepdim=True)

        scaling = torch.where(
            max_scale_per_batch > 1.0,
            1.0 / max_scale_per_batch,
            torch.ones_like(max_scale_per_batch)
        )
        motor_thrusts = motor_thrusts * scaling

        # Step 8: Convert motor thrusts to motor speeds (PX4 mixer output)
        motor_speeds = self.dynamics.motorThrustToOmega(motor_thrusts)

        # Step 9: Reconstruct resulting force & torque
        alloc = torch.bmm(self.alloc_matrix_, motor_thrusts.unsqueeze(-1)).squeeze(-1)

        force = torch.zeros((batch, 3), device=self.device)
        force[:, 2] = alloc[:, 0]  # Thrust in z-direction
        torque = alloc[:, 1:4]     # Torque [roll, pitch, yaw] in Nm

        # Step 10: Saturation detection using PHYSICAL TORQUE (NO normalization needed)
        # Both tau_des and torque are in Nm, can directly compare!
        unallocated_torque = tau_des - torque

        # Detect saturation using PX4's threshold
        # PX4 uses FLT_EPSILON (≈ 1.19e-7) but for physical torque we need a reasonable threshold
        # For Crazyflie: typical max torque ~1e-4 Nm, so use 1e-6 Nm as threshold
        saturation_threshold = 1e-6  # Nm
        sat_positive = unallocated_torque > saturation_threshold
        sat_negative = unallocated_torque < -saturation_threshold

        self.rate_control.set_saturation_status(sat_positive, sat_negative)

        # Build info dictionary with intermediate control values
        info = {
            'omega_body': omega_body,              # Current body rates [p, q, r]
            'rate_sp': rate_sp,                    # Desired body rates
            'rate_error': rate_sp - omega_body,    # Rate tracking error
            'angular_accel': angular_accel,        # Angular acceleration
            'tau_des': tau_des,                    # Desired normalized torque
            'motor_thrusts': motor_thrusts,        # Individual motor thrusts [N]
            'thrust_cmd': thrust_cmd,              # Commanded thrust [0, 1]
            'thrust': force[:, 2],                # Actual thrust force [N]
            'torque': torque                       # Actual torque [Nm]
        }

        # Add desired quaternion only in attitude mode
        if mode == 'attitude':
            info['q_des'] = q_des

        return force, torque, motor_speeds, info

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector from world frame to body frame."""
        # v_body = q^-1 * v * q
        q_inv = torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)
        v_quat = torch.cat([torch.zeros(v.shape[0], 1, device=v.device), v], dim=1)
        temp = self._quat_mul(q_inv, v_quat)
        result = self._quat_mul(temp, q)
        return result[:, 1:]

    def _quat_from_euler(self, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Convert ZYX Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z]."""
        return quat_from_euler_xyz(roll, pitch, yaw)

    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion batch to ZYX Euler angles [roll, pitch, yaw]."""
        return quat_to_euler_xyz(quat)

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=1)

    def reset(self, env_ids: torch.Tensor = None):
        """
        Reset controller states for specified environments.
        Clears PID integral, angular velocity history, and angular acceleration filter.

        Args:
            env_ids: (N,) tensor of environment indices to reset. If None, resets all environments.
                     Can be integer indices tensor.

        Example:
            # Reset specific environments
            controller.reset(torch.tensor([0, 2, 5]))

            # Reset all environments
            controller.reset()

            # Reset using boolean mask (convert to indices first)
            done_mask = torch.tensor([True, False, True, False])
            controller.reset(done_mask.nonzero(as_tuple=True)[0])
        """
        if env_ids is None:
            # Reset all environments
            self.rate_control.reset_integral()
            self.prev_omega_body.zero_()
            self.prev_angular_accel.zero_()
        else:
            # Reset specific environments
            self.rate_control.reset_integral(env_ids)
            self.prev_omega_body[env_ids] = 0.0
            self.prev_angular_accel[env_ids] = 0.0
