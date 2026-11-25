# ME5659-Final-Project

Two-wheeled self-balancing robot (Pyclops). The model is built in MATLAB, the gains come from
pole placement, and the controller is tested in MuJoCo.

## Workflow: regenerating gains

1. Edit physical parameters in `balancing_bot/robot.xml`.
2. Run `python get_params.py` to extract `p = [M m r l Ib Iw Ia g]` into `params.json`.
3. Copy the values into `MATLAB/balbot_linearized_model_torque.m` and run it. This regenerates
   `balbot_A.m`, `balbot_B.m`, `balbot_dynamics.m` and computes `K`.
4. Paste `K` into `balancing_bot_env.py`.

`K` is only valid for the parameters it was computed with. In particular, re-run all four steps
whenever the armature changes.

## Model conventions (MATLAB ↔ MuJoCo)

| Quantity | MATLAB model | MuJoCo |
|---|---|---|
| Wheel angle | `phi` = **absolute** wheel angle, `x = r*phi` | joint `qpos` = wheel angle **relative to body**, `psi` |
| Conversion | `phi = psi + theta`, `dphi = dpsi + dtheta` | `psi = (right - left)/2` (left joint axis is flipped) |
| Pitch `theta` | + tips body forward | rotation about world +x (axle, right→left); forward is world −y |
| Input `u` | **total** torque on both wheels, `u = -K x` | `ctrl[0] = u/2`, `ctrl[1] = -u/2` |

State vector: `x = [phi, theta, dphi, dtheta]`.

## Armature must act on the relative wheel angle

MuJoCo's joint `armature` models reflected motor-rotor inertia (N²·J_rotor). The motor housing
is fixed to the body, so the rotor spins with the **joint** angle `psi = phi - theta`, not with
the absolute wheel angle. In the Lagrangian:

```matlab
K_motor = 0.5*Ia*(dphi - dtheta)^2;   % correct
% NOT: lumping Ia into Iw (0.5*Iw*dphi^2)
```

Lumping it into `Iw` gets the wheel–body coupling term `D12` wrong (`M*r*l` instead of
`M*r*l - Ia`) and underestimates the body inertia `D22` by `Ia`. With the original
`armature=0.005`, `D12` came out with the wrong sign, and the "stable" K placed closed-loop
eigenvalues at **+89 and +12.7** on the actual plant.

`armature` is currently a **placeholder of 0.001 kg·m² per wheel** (J_rotor ≈ 1e-6 with a
~30:1 gearbox). Replace it with real motor data.

Current gains (poles `[-5.1 -5.2 -25.1 -25.2]`, `Ia = 0.002` total):
`K = [-0.9798 -13.8461 -0.4585 -1.6893]`. In MuJoCo these recover from 5°, 10° and 15° initial
tilts with a peak total torque of 2.2, 4.5 and 6.9 N·m respectively.

## Steady-state oscillation: Coulomb friction

After the fixes above, the robot balances but keeps oscillating. Measured in MuJoCo from a
5° start, over t = 5–15 s:

| `frictionloss` (per wheel) | θ peak-to-peak | x peak-to-peak | τ peak-to-peak | Frequency |
|---|---|---|---|---|
| 0.10 N·m (current) | 1.58° | 12.4 mm | 0.48 N·m | ~1.2 Hz |
| 0.02 N·m | 0.12° | 0.5 mm | 0.11 N·m | ~3.9 Hz |
| 0.00 N·m | 0.47° | 1.2 mm | 0.07 N·m | ~4.5 Hz |

The large oscillation is a friction limit cycle. `frictionloss` is Coulomb friction,
`0.1·sign(dpsi)`, which is the same order of magnitude as the control torques near
equilibrium. A small residual remains even with zero friction. The likely cause is the
faceted convex-hull collision mesh of the wheels, but this hasn't been confirmed yet.

### Why not just integral control?

Integral action cancels **constant** disturbances. Coulomb friction flips sign every time the
wheel reverses, so an integrator chases it and tends to cause stick-slip hunting instead.
Integral action on wheel position (augment the state with ∫phi) is still useful on hardware
for real constant offsets: a CG not over the axle, IMU mounting or bias offset, and floor slope.

## Plan for the physical robot

1. **Model the motor with voltage as the input.** Real drivers apply voltage (PWM), not torque.
   Back-EMF and viscous friction are linear and act on `dpsi`:
   `tau = (Kt/R)*V - (Kt*Kb/R)*dpsi - b*dpsi`. Include them in the Lagrangian model and design K
   with voltage as the input.
2. **Cancel Coulomb friction with feedforward**, outside the linear model:
   ```python
   u = -K @ x + tau_c * np.tanh(dpsi / eps)   # eps ~ 0.5–1 rad/s
   ```
   Use about 70–80% of the identified `tau_c`. Overcompensating, or using a hard `sign()`,
   causes chatter near zero speed.
3. **Add integral action on `phi`** for constant offsets.
4. **Identify parameters on the real hardware:**
   - Breakaway (Coulomb) friction: with the wheel off the ground, slowly ramp the PWM and note
     where it starts to turn.
   - Viscous friction and back-EMF: steady-state speed at several voltages, plus a coast-down test.
   - Reflected rotor inertia (armature): motor datasheet J_rotor × gear ratio².
5. **Be less aggressive.** Gearbox backlash usually causes more limit-cycling than friction.
   Poles at −25 rad/s are aggressive once encoder quantization, IMU filtering and loop delay
   are involved. LQR, or slower poles around −10 to −15, tends to transfer better.
6. **Stress-test in MuJoCo before hardware:** sensor noise (`noise=` on sensors), a one-sample
   control delay, ±20% parameter errors, and the complementary-filter pitch estimate instead of
   ground-truth `qpos`.
