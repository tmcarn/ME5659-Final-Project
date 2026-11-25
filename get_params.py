"""
Extract the planar balancing-bot parameters from the MuJoCo model.

Matches the MATLAB model in MATLAB/balbot_linearized_model_torque.m:
    p = [M m r l Ib Iw Ia g]

    M  : body (base_link) mass                                [kg]
    m  : total wheel mass (both wheels)                       [kg]
    r  : wheel radius                                         [m]
    l  : distance from wheel axle to body COM                 [m]
    Ib : body inertia about its COM, around the axle axis     [kg m^2]
    Iw : total wheel inertia about the axle (NO armature)     [kg m^2]
    Ia : total joint armature (both wheels)                   [kg m^2]
         MuJoCo applies armature to the joint DOF, i.e. the wheel angle
         RELATIVE to the body (psi = phi - theta), so it must enter the
         Lagrangian as 0.5*Ia*(dphi - dtheta)^2, not lumped into Iw.
    g  : gravity                                              [m/s^2]

Run:  python get_params.py
Writes params.json next to this file and prints a MATLAB-ready line.
"""

import json
import pathlib

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
XML_PATH = HERE / "balancing_bot" / "robot.xml"

BODY = "base_link"
WHEELS = {"right": ("dof_right_wheel", "right_wheel", "act_right_wheel"),
          "left":  ("dof_left_wheel",  "left_wheel",  "act_left_wheel")}


def unit(v):
    return v / np.linalg.norm(v)


def inertia_about_axis(model, data, body_id, axis):
    """Moment of inertia of a body about an axis through its COM (world frame)."""
    R = data.ximat[body_id].reshape(3, 3)          # principal-inertia frame
    I_world = R @ np.diag(model.body_inertia[body_id]) @ R.T
    return float(axis @ I_world @ axis)


def perp_dist(point, line_point, axis):
    """Distance from a point to a line (line_point, axis)."""
    d = point - line_point
    return float(np.linalg.norm(d - (d @ axis) * axis))


def wheel_radius(model, data, wheel_body_id, anchor, axis):
    """Largest distance of any collision-mesh vertex from the wheel axle."""
    r = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != wheel_body_id:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue                                   # skip visual-only geoms
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            r = max(r, model.geom_rbound[g])
            continue
        mid = model.geom_dataid[g]
        start, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[start:start + n]       # mesh frame
        R = data.geom_xmat[g].reshape(3, 3)
        world = data.geom_xpos[g] + verts @ R.T
        d = world - anchor
        d_perp = d - np.outer(d @ axis, axis)
        r = max(r, float(np.linalg.norm(d_perp, axis=1).max()))
    return r


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)                     # robot upright at qpos0

    body_id = model.body(BODY).id

    # --- Wheel joints: anchors and axle direction ---
    w = {}
    for side, (wbody, wjoint, wact) in WHEELS.items():
        jid = model.joint(wjoint).id
        w[side] = dict(
            body_id=model.body(wbody).id,
            joint_id=jid,
            act_id=model.actuator(wact).id,
            anchor=data.xanchor[jid].copy(),
            axis=unit(data.xaxis[jid].copy()),
        )

    # Pitch axis = axle direction, pointing from right wheel to left wheel
    axle = unit(w["left"]["anchor"] - w["right"]["anchor"])
    axle_center = 0.5 * (w["left"]["anchor"] + w["right"]["anchor"])
    z_up = unit(-model.opt.gravity)
    forward = unit(np.cross(axle, z_up))   # direction body tips for +theta

    # --- Body ---
    M = float(model.body_mass[body_id])
    com_b = data.xipos[body_id].copy()
    l = perp_dist(com_b, axle_center, axle)
    Ib = inertia_about_axis(model, data, body_id, axle)

    # --- Wheels ---
    m = Iw = Ia = 0.0
    radii, per_wheel = [], {}
    for side, d in w.items():
        bid, jid = d["body_id"], d["joint_id"]
        mw = float(model.body_mass[bid])
        # inertia about the wheel's own axle (parallel axis for small COM offset)
        I_com = inertia_about_axis(model, data, bid, d["axis"])
        off = perp_dist(data.xipos[bid], d["anchor"], d["axis"])
        I_axle = I_com + mw * off**2
        armature = float(model.dof_armature[model.jnt_dofadr[jid]])
        r_w = wheel_radius(model, data, bid, d["anchor"], d["axis"])

        m += mw
        Iw += I_axle          # spins with absolute wheel angle phi
        Ia += armature        # spins with relative joint angle psi = phi - theta
        radii.append(r_w)
        per_wheel[side] = dict(
            mass=mw,
            inertia_axle=I_axle,
            armature=armature,
            radius=r_w,
            com_offset_from_axle=off,
            frictionloss=float(model.dof_frictionloss[model.jnt_dofadr[jid]]),
            damping=float(model.dof_damping[model.jnt_dofadr[jid]]),
            # +1: positive joint angle = positive rotation about the pitch axis
            joint_sign=int(np.sign(d["axis"] @ axle)),
            gear=float(model.actuator_gear[d["act_id"], 0]),
            ctrlrange=model.actuator_ctrlrange[d["act_id"]].tolist(),
        )
    r = float(np.mean(radii))
    g = float(np.linalg.norm(model.opt.gravity))

    params = dict(M=M, m=m, r=r, l=l, Ib=Ib, Iw=Iw, Ia=Ia, g=g)
    extra = dict(
        timestep=float(model.opt.timestep),
        total_mass=float(model.body_subtreemass[body_id]),
        pitch_axis_world=axle.tolist(),
        forward_dir_world=forward.tolist(),
        wheels=per_wheel,
    )

    # --- Report ---
    print(f"Model: {XML_PATH.relative_to(HERE)}\n")
    print("Planar model parameters  (p = [M m r l Ib Iw Ia g])")
    units = dict(M="kg", m="kg", r="m", l="m", Ib="kg m^2", Iw="kg m^2", Ia="kg m^2",
                 g="m/s^2")
    for k, v in params.items():
        print(f"  {k:<3}= {v:.6g}  {units[k]}")

    print("\nDetails")
    print(f"  timestep          = {extra['timestep']} s")
    print(f"  total mass        = {extra['total_mass']:.6g} kg")
    print(f"  pitch axis (world)= {np.round(axle, 4)}   (+theta = rotation about this)")
    print(f"  forward (world)   = {np.round(forward, 4)}   (direction of +phi travel)")
    for side, pw in per_wheel.items():
        print(f"  {side:>5} wheel: radius={pw['radius']:.4f} m, joint_sign={pw['joint_sign']:+d}, "
              f"gear={pw['gear']:g}, ctrlrange={pw['ctrlrange']}, "
              f"armature={pw['armature']:g}, frictionloss={pw['frictionloss']:g}")

    print("\nMATLAB:")
    print("  M = {M:.6g}; m = {m:.6g}; r = {r:.6g}; l = {l:.6g}; "
          "Ib = {Ib:.6g}; Iw = {Iw:.6g}; Ia = {Ia:.6g}; g = {g:.6g};".format(**params))
    print("  p = [M m r l Ib Iw Ia g];")

    out = HERE / "params.json"
    out.write_text(json.dumps(dict(params=params, extra=extra), indent=2))
    print(f"\nSaved {out.name}")
    return params, extra


if __name__ == "__main__":
    main()
