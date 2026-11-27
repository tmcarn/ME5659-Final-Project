from math import tau
import math

import mujoco
import mujoco.viewer
import time
import numpy as np

from scipy.linalg import solve_discrete_are

from data_logger import DataLogger

# Encoder resolution
TICKS_PER_REV = 1024
WHEEL_RADIUS = 0.0375   # m (7.5 cm diameter wheel)


class PyClopsEnv():
    def __init__(self, xml_path="./balancing_bot/scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
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
        # right_pos = self.data.sensor("right_wheel_pos").data[0]
        # left_pos = self.data.sensor("left_wheel_pos").data[0]
        # t = self.data.time

        # # Derive velocity via finite difference
        # dt = t - self.prev_time
        # if dt > 0:
        #     right_vel = (right_pos - self.prev_right_pos) / dt
        #     left_vel = (left_pos - self.prev_left_pos) / dt
        # else:
        #     right_vel = 0.0
        #     left_vel = 0.0

        # # Update previous state
        # self.prev_right_pos = right_pos
        # self.prev_left_pos = left_pos
        # self.prev_time = t

        print(f"q = {self.data.qpos}")
        print(f"dq = {self.data.qvel}")

        right_pos = self.data.qpos[-2]
        left_pos = self.data.qpos[-1]
        right_vel = self.data.qvel[-2]
        left_vel = self.data.qvel[-1]

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
 
        # Pitch rate directly from freejoint qvel: [vx, vy, vz, wx, wy, wz]
        # Linear part is in the world frame; angular part (wx, wy, wz) is in the body frame.
        # Pitch is rotation about x, so the rate is wx.
        pitch_rate = self.data.qvel[3]
 
        return pitch, pitch_rate


    def estimate_state(self):
        """Build state vector [x, theta, x_dot, theta_dot] from sensors."""
        accel, gyro = self.get_imu()
        right_pos, left_pos, right_vel, left_vel = self.get_encoders()

        print(f"RIGHT POS: {right_pos:.4f}, LEFT POS: {left_pos:.4f}")
        dt = self.model.opt.timestep

        # Pitch from complementary filter
        theta, dtheta = self.estimate_pitch(accel, gyro, dt)

        # Encoders measure wheel angle RELATIVE to the body (psi). The model state
        # is the ABSOLUTE wheel angle phi (x = r*phi), so phi = psi + theta.
        # Right joint axis = +pitch axis, left joint axis is flipped (see params.json
        # joint_sign), so average right - left.
        psi = (right_pos - left_pos) / 2.0
        dpsi = (right_vel - left_vel) / 2.0

        # Absolute wheel angle (MATLAB state), x = r*phi
        phi = psi + theta
        dphi = dpsi + dtheta

        print(f'RAW STATE: psi={psi}, dpsi={dpsi}, theta={theta}, dtheta={dtheta}')

        return np.array([phi, theta, dphi, dtheta])

    def lqr_control(self, state, x_des, K):
        """Compute control input: u = -K @ (state - x_des)"""
        error = state - x_des
        u = -K @ error
        return u

    def apply_control(self, u, dc_motor):
        """Send the controller output to both wheels (left joint axis is flipped).

        dc_motor=True : u is the motor VOLTAGE, applied in full to each motor
                        (the 2x for two motors is already inside the model).
        dc_motor=False: u is the TOTAL wheel torque, split between the wheels.
        """
        per_wheel = u if dc_motor else u / 2.0
        self.data.ctrl[0] = per_wheel
        self.data.ctrl[1] = -per_wheel

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


class LinearStateFeedbackController:
    '''
    A simple linear state feedback controller: u = -K @ (x - x_des)
    where K is the gain matrix, x is the current state, and x_des is the desired state.

    (Could be LQR or Pole placement gains, etc.)
    '''
    def __init__(self, K, limit=5.0):
        self.K = K
        self.limit = limit

    def compute_control(self, state):
        u = -self.K @ state
        return float(np.clip(u, -self.limit, self.limit))


# True : robot_dc.xml, ctrl = voltage, DC motor (back-EMF) modeled in the actuator
# False: robot.xml,    ctrl = torque
USE_DC_MOTOR = True

if __name__ == "__main__":
    # State for both: [phi, theta, dphi, dtheta], poles [-5.1 -5.2 -25.1 -25.2], Ia = 0.002.
    # Recompute K if armature or motor constants change!
    if USE_DC_MOTOR:
        # From MATLAB/balbot_linearized_model_dc.m (Ng=30, Kt=Kb=0.01, Rm=2, b=0.001)
        xml_path = "./balancing_bot/scene_dc.xml"
        K = np.array([-3.2661, -46.1537, -1.8316, -5.3276])
        U_LIMIT = 12.6        # V, battery voltage (= actuator ctrlrange)
        u_name, u_unit = "V", "V"
    else:
        # From MATLAB/balbot_linearized_model_torque.m
        xml_path = "./balancing_bot/scene.xml"
        K = np.array([-0.9798, -13.8461, -0.4585, -1.6893])
        U_LIMIT = 20.0        # N m, total torque = 2 wheels x ctrlrange 10
        u_name, u_unit = "Tau", "Nm"

    env = PyClopsEnv(xml_path)
    model = env.model
    data = env.data

    controller = LinearStateFeedbackController(K, limit=U_LIMIT)
    # controller = PIDController(kp=20.0, ki=5, kd=2.0, limit=5.0)

    theta_des = 0.0  # Desired pitch angle

    logger = DataLogger(state_names=("x", "theta", "x_dot", "theta_dot"),
                        output_names=(u_name,))

    # --- Main loop ---
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step = 0
            print_interval = 1 # int(0.1 / model.opt.timestep)

            while viewer.is_running():
                # Read sensors and compute control
                state = env.estimate_state()

                #F = env.lqr_control(state, x_des, K)

                # u = controller.compute_control(state, x_des, K)
                # data.ctrl[0] = u[0]
                # data.ctrl[1] = u[1]

                # Convert force to wheel torque
                #tau = -F / WHEEL_RADIUS

                #tau = controller.compute(theta_des - state[1], model.opt.timestep)  # Control based on pitch error only

                u = controller.compute_control(state)  # Control based on full state feedback

                logger.log(data.time, state, u)

                if abs(u) >= U_LIMIT:
                    print(f"{u_name} saturated at {u:.2f} {u_unit}")
                env.apply_control(u, USE_DC_MOTOR)

                if step % print_interval == 0:
                    print(f"Time: {data.time:6.2f}  "
                            f"x: {state[0]:7.3f} "
                        f"Theta: {state[1]:7.3f} "
                        f"x_dot: {state[2]:7.3f} "
                        f"theta_dot: {state[3]:7.3f} "
                        f"{u_name}: {u:7.3f}")
                        #f"U: {u[0]:7.3f} {u[1]:7.3f}")

                # Step physics ONCE
                mujoco.mj_step(model, data)

                step += 1
                viewer.sync()
                time.sleep(model.opt.timestep * 1)
    finally:
        if len(logger):
            logger.plot()  # Save plot of logged data