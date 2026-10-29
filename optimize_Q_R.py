from pyclops_env import PyClopsEnv
from pyclops_env import LQRController

from scipy.optimize import minimize
import mujoco
import mujoco.viewer
import time
import numpy as np

env = PyClopsEnv()
controller = LQRController(env)


def objective(log_params):
    params = np.exp(log_params)  # always positive
    q_x, q_theta, q_xdot, q_thetadot, r = params
    Q = np.diag([q_x, q_theta, q_xdot, q_thetadot])
    R = np.eye(2) * r
    try:
        K = controller.compute_lqr_gain(Q, R)
    except:
        return 1e6
    cost = simulate_and_score(env, K, duration=1.0)
    print(f"  params={params.round(3)}, cost={cost:.2f}")
    return cost


def simulate_and_score(env, K, duration=1.0):
    """Lower is better."""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[4] += 0.05  # small initial pitch perturbation
    mujoco.mj_forward(env.model, env.data)
    
    total_cost = 0.0
    steps = int(duration / env.model.opt.timestep)
    x_des = np.array([0.0, 0.0, 0.0, 0.0])  # desired state: [x, theta, x_dot, theta_dot]
    for _ in range(steps):
        state = env.estimate_state()
        u = controller.compute_control(state, x_des, K)
        for i in range(env.model.nu):
            env.data.ctrl[i] = np.clip(u[i], *env.model.actuator_ctrlrange[i])
        mujoco.mj_step(env.model, env.data)
        
        # Penalize tilt, position drift, and control effort
        total_cost += 10 * state[1]**2 + state[0]**2 + 0.01 * np.sum(u**2)
        
        if abs(state[1]) > 0.5:  # fell over
            return 1e10
    
    
    return total_cost

# Q = np.diag([1, 10, 1, 5])
# R = np.eye(2) * 0.1
# K = controller.compute_lqr_gain(Q, R)

# cost = simulate_and_score(env, K, duration=1.0)
# print("Initial parameters:", [1, 10, 1, 5, 0.1])
# print("Initial cost:", cost)




result = minimize(objective, 
                  x0=np.log([5, 50, 5, 50, 1]),
                  method='Nelder-Mead',
                  options={'maxiter': 500, 'xatol': 0.1, 'fatol': 1.0, 'disp': True})

print("Optimized parameters:", np.exp(result.x))
print("Optimized cost:", result.fun)