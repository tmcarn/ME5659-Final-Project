import mujoco
import mujoco.viewer
import time
import numpy as np

from scipy.linalg import solve_discrete_are

# Encoder resolution
TICKS_PER_REV = 1024
WHEEL_RADIUS = 0.075   # 7.5 cm diameter wheel


class PyClopsEnv():
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("./pyclops/scene.xml")
        self.data = mujoco.MjData(self.model)

        # Previous encoder state for velocity estimation
        self.prev_right_pos = 0.0
        self.prev_left_pos = 0.0
        self.prev_time = 0.0

        self.prev_pitch = 0.0

        # Complementary filter state for pitch
        self.pitch_estimate = 0.0
        self.alpha = 0.98  # gyro trust factor (0.95-0.99 typical)

    def get_imu(self):
        """Read simulated IMU data from MuJoCo sensors."""
        accel = self.data.sensor("imu_accel").data.copy()
        gyro = self.data.sensor("imu_gyro").data.copy()
        return accel, gyro

    def get_encoders(self):
        """Read simulated wheel encoders (position only) and derive velocity."""
        right_pos = self.data.sensor("right_wheel_pos").data[0]
        left_pos = self.data.sensor("left_wheel_pos").data[0]
        t = self.data.time

        # Derive velocity via finite difference
        dt = t - self.prev_time
        if dt > 0:
            right_vel = (right_pos - self.prev_right_pos) / dt
            left_vel = (left_pos - self.prev_left_pos) / dt
        else:
            right_vel = 0.0
            left_vel = 0.0

        # Update previous state
        self.prev_right_pos = right_pos
        self.prev_left_pos = left_pos
        self.prev_time = t

        return right_pos, left_pos, right_vel, left_vel

    def estimate_pitch(self, accel, gyro, dt):
        """Complementary filter: fuse accelerometer and gyro for pitch."""
        # # Accel-based pitch (only valid when not accelerating)
        # accel_pitch = np.arctan2(-accel[0], accel[2])

        # # Gyro integration
        # if dt > 0:
        #     gyro_pitch = self.pitch_estimate + gyro[1] * dt
        # else:
        #     gyro_pitch = self.pitch_estimate

        # # Blend: trust gyro short-term, accel long-term
        # self.pitch_estimate = self.alpha * gyro_pitch + (1 - self.alpha) * accel_pitch
        # return self.pitch_estimate

        """Get pitch angle directly from qpos (freejoint quaternion)."""
        # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        w, x, y, z = quat
 
        # Extract pitch (rotation about x-axis) from quaternion
        pitch = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
 
        # Pitch rate via finite difference
        dt = self.model.opt.timestep
        pitch_rate = (pitch - self.prev_pitch) / dt if dt > 0 else 0.0
        self.prev_pitch = pitch
 
        return pitch, pitch_rate


    def estimate_state(self):
        """Build state vector [x, theta, x_dot, theta_dot] from sensors."""
        accel, gyro = self.get_imu()
        right_pos, left_pos, right_vel, left_vel = self.get_encoders()

        dt = self.model.opt.timestep

        # Forward position & velocity (average of both wheels)
        x = WHEEL_RADIUS * (right_pos + left_pos) / 2.0
        dx = WHEEL_RADIUS * (right_vel + left_vel) / 2.0

        # Pitch from complementary filter
        theta, dtheta = self.estimate_pitch(accel, gyro, dt)

        # Pitch rate directly from gyro
        # theta_dot = gyro[1]

        return np.array([x, theta, dx, dtheta])

    def lqr_control(self, state, x_des, K):
        """Compute control input: u = -K @ (state - x_des)"""
        error = state - x_des
        u = -K @ error
        return u
    
class LQRController:
    def __init__(self, env):
        self.env = env
        self.nu = env.model.nu # Number of Actuated DOFs (2)
        self.nv = env.model.nv # Total DOFs (8) SE(3) + 2 wheels

        self.A, self.B = self.linearize_dynamics()
        self.K = self.compute_lqr_gain(Q=np.diag([20, 100, 10, 100]), R=np.diag([1, 1]))
        print("LQR Gain K:\n", self.K)

    def linearize_dynamics(self):
        # Allocate Jacobians
        # A is (2*nv x 2*nv), B is (2*nv x nu)
        A_full = np.zeros((2 * self.nv, 2 * self.nv))
        B_full = np.zeros((2 * self.nv, self.nu))

        eps = 1e-6
        flg_centered = 1
 
        mujoco.mjd_transitionFD(self.env.model, self.env.data, eps, flg_centered, A_full, B_full, None, None)


        # Index mapping for our 4-state LQR
        # In the 2*nv state vector: first nv entries = position, next nv = velocity
        idx_x = 0          # forward position (dof 0)
        idx_theta = 4      # pitch angle (dof 4, rotation about -axis)
        idx_x_dot = self.nv + 0
        idx_theta_dot = self.nv + 4
    
        state_idx = [idx_x, idx_theta, idx_x_dot, idx_theta_dot]

            # Extract the 4x4 and 4x2 sub-matrices
        A = A_full[np.ix_(state_idx, state_idx)]
        B = B_full[np.ix_(state_idx, range(self.nu))]

        return A, B
    
    def compute_lqr_gain(self, Q, R):
        """Solve the continuous-time Algebraic Riccati Equation for LQR gain."""
        P = solve_discrete_are(self.A, self.B, Q, R)
        self.K = np.linalg.inv(R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        return self.K
    
    def compute_control(self, state, x_des, K):
        error = state - x_des
        u = -K @ error
        return u


class PIDController:
    def __init__(self, kp, ki, kd, limit=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.integral = 0.0
        self.prev_error = 0.0
 
    def compute(self, error, dt):
        if dt <= 0:
            return 0.0
 
        self.integral += error * dt

        # Anti-windup clamp
        self.integral = np.clip(self.integral, -self.limit / self.ki if self.ki != 0 else 0,
                                                self.limit / self.ki if self.ki != 0 else 0)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
 
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -self.limit, self.limit)
 
if __name__ == "__main__":
    env = PyClopsEnv()
    model = env.model
    data = env.data

    # x_des = np.array([0.0, 0.0, 0.0, 0.0])
    # K = np.array([1.4142, -2.2646, 2.5880, -1.2551])
    # controller = LQRController(env)
    # controller = PIDController(kp=20.0, ki=5, kd=8.0, limit=5.0)

    theta_des = 0.0  # Desired pitch angle

    # --- Main loop ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        print_interval = int(0.1 / model.opt.timestep)

        print(f"{'time':>6s}  {'x':>7s} {'theta':>7s} {'x_dot':>7s} {'th_dot':>7s}  {'F':>7s} {'tau':>7s}")
        print("-" * 65)

        while viewer.is_running():
            # Read sensors and compute control
            state = env.estimate_state()
            # F = env.lqr_control(state, x_des, K)

            # u = controller.compute_control(state, x_des)
            # data.ctrl[0] = u[0]
            # data.ctrl[1] = u[1]

            # Convert force to wheel torque
            # tau = -F / WHEEL_RADIUS

            # tau = controller.compute(theta_des - state[1], model.opt.timestep)  # Control based on pitch error only

            # Apply to both wheels (same sign — check if this needs flipping)
            # data.ctrl[0] = -tau
            # data.ctrl[1] = tau

            if step % print_interval == 0:
                print(f"Time: {data.time:6.2f}  "
                        f"x: {state[0]:7.3f} "
                    f"Theta: {state[1]:7.3f} "
                    f"x_dot: {state[2]:7.3f} "
                    f"theta_dot: {state[3]:7.3f} ")
                    # f"U: {u[0]:7.3f} {u[1]:7.3f}")

            # Step physics ONCE
            mujoco.mj_step(model, data)

            step += 1
            viewer.sync()
            time.sleep(model.opt.timestep)