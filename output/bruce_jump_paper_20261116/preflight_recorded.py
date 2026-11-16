"""Replay a RECORDED standing model on Khadas; opens no actuator interfaces.

No import of Startups.run_bear (it has live-memory dependencies): extract only
its arithmetic current-conversion function, with calls/imports prohibited.
This evaluates isolated QPs at one posture, not a time-domain jump simulation.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from jump_core import AccelerationTask, JumpQP, ballistic_requirements


def installed_current_map(root, torque_to_iq):
    path = root / "Startups/run_bear.py"
    tree = ast.parse(path.read_text())
    functions = [x for x in tree.body if isinstance(x, ast.FunctionDef)
                 and x.name == "joint_to_motor_iq"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one installed current mapping")
    fn = functions[0]
    fn.decorator_list = []
    allowed = (ast.FunctionDef, ast.arguments, ast.arg, ast.Expr, ast.Constant,
               ast.Assign, ast.Name, ast.Store, ast.Load, ast.BinOp, ast.UnaryOp,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
               ast.Return, ast.List, ast.Tuple)
    if any(not isinstance(node, allowed) for node in ast.walk(fn)):
        raise ValueError("Installed mapping is not pure arithmetic; manual review required")
    namespace = {"TORQUE2IQ": torque_to_iq, "__builtins__": {}}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    convert = namespace[fn.name]
    M = np.column_stack([np.asarray(convert(*v)).ravel() for v in np.eye(10)])
    for v in [np.zeros(10), np.linspace(-3, 4, 10), np.arange(10.)]:
        np.testing.assert_allclose(convert(*v), (M @ v)[:, None], atol=1e-12)
    return M


def run(root, fixture, out, repetitions):
    root = root.resolve()
    sys.path.insert(0, str(root))
    macros = importlib.import_module("Settings.BRUCE_macros")
    dyn = importlib.import_module("Library.ROBOT_MODEL.BRUCE_dynamics")
    kin = importlib.import_module("Library.ROBOT_MODEL.BRUCE_kinematics")
    f = json.loads(fixture.read_text())
    e = {k: np.asarray(v, dtype=float) for k, v in f["ESTIMATOR_STATE"].items()}
    q = np.asarray(f["LEG_STATE"]["joint_positions"])
    dq = np.asarray(f["LEG_STATE"]["joint_velocities"])
    # Recompute full dynamics using the installed ARM64 binary, not local sim.
    recomputed = dyn.robotID(e["body_rot_matrix"], e["body_position"],
                            e["body_ang_rate"], e["body_rot_matrix"].T @ e["body_velocity"],
                            *q, *dq)
    H, CG, AG, dAGdq = map(np.asarray, recomputed[:4])
    M = installed_current_map(root, macros.TORQUE2IQ)
    # Recompute contact Jacobians at exactly the SAME q/dq/R as H/CMM.
    # A recorded shared-memory fixture can contain fields from different ticks.
    fk = kin.legFK(*q, *dq)
    selected = [0, 2, 3, 4, 6, 7, 8, 10, 11, 14, 15, 16]
    world = kin.robotFK(e["body_rot_matrix"], e["body_position"],
                        e["body_ang_rate"], e["body_rot_matrix"].T @ e["body_velocity"],
                        *[fk[i] for i in selected + [i + 17 for i in selected]], *dq)
    Jc = np.array([world[i] for i in (2, 6, 20, 24)])
    bias = np.array([world[i] for i in (3, 7, 21, 25)])
    mass_diag = np.diag(H)[3:6]
    model_mass = float(macros.MASS_TOT)
    mass_consistent = bool(np.allclose(mass_diag, model_mass, atol=1e-6))
    cases = []
    for name, az, active in [("recorded_stance", 0., [1,1,1,1]),
                             ("upward_acceleration_screen", 3., [1,1,1,1]),
                             ("landing_deceleration_screen", 5., [1,1,1,1]),
                             ("right_contact_only_screen", 0., [1,1,0,0]),
                             ("left_contact_only_screen", 0., [0,0,1,1])]:
        tasks = [AccelerationTask(AG[3:], dAGdq[3:],
                                  model_mass*np.array([0.,0.,az]), np.full(3, 50.)),
                 AccelerationTask(AG[:3], dAGdq[:3], np.zeros(3), np.ones(3))]
        for i, on in enumerate(active):
            if on:
                tasks.append(AccelerationTask(Jc[i], bias[i], np.zeros(3), np.full(3, 1e3)))
        solver = JumpQP()
        inputs = dict(H=H, CG=CG, Jc=Jc, active=active, mu=.5,
                      maximum_normal_force=np.full(4, 100.), tasks=tasks,
                      torque_limit=np.full(10, macros.TORQUE_MAX),
                      joint_torque_to_current=M,
                      current_limit=np.full(10, macros.IQ_MAX))
        times = []
        try:
            r = solver.solve(**inputs)  # setup excluded from steady-state timing
            for _ in range(repetitions):
                t = time.perf_counter()
                r = solver.solve(**inputs)
                times.append(time.perf_counter() - t)
            case = dict(name=name, status="SOLVED_AT_RECORDED_POSTURE",
                        torque_abs_max_Nm=float(abs(r["torque_total"]).max()),
                        motor_current_abs_max_A=float(abs(r["current"]).max()),
                        constraint_violation=float(r["constraint_violation"]),
                        contact_normal_forces_N=r["forces"][:,2].tolist(),
                        mean_total_qp_ms=float(np.mean(times)*1000),
                        p99_total_qp_ms=float(np.percentile(times,99)*1000),
                        max_total_qp_ms=float(max(times)*1000),
                        solver_only_last_ms=float(r["solver_s"]*1000))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as ex:
            case = dict(name=name, status="REJECTED", reason=str(ex))
        cases.append(case)
        print(json.dumps(case), flush=True)
    paths = [root / "Settings/BRUCE_macros.py", root / "Startups/run_bear.py",
             root / "Startups/run_estimation.py", root / "Library/ROBOT_MODEL/BRUCE_DYNAMICS_AOT.py",
             Path(dyn.__file__), Path(kin.__file__), fixture]
    report = dict(status="OFFLINE_MODEL_SCREEN_ONLY", robot_root=str(root),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  model_mass_kg=model_mass, physical_mass_measured=False,
                  compiled_mass_diagonal_kg=mass_diag.tolist(),
                  source_mass_matches_binary=mass_consistent,
                  recorded_vs_current_H_max_abs=float(abs(H-e["H_matrix"]).max()),
                  current_map_A_per_Nm=M.tolist(),
                  thigh_m=float(macros.a3), shank_m=float(macros.a4),
                  ballistic_1cm_com_rise=ballistic_requirements(.01),
                  repetitions_per_case=repetitions, cases=cases,
                  hardware_actuation=False, simulation_executed=False,
                  control_loop_500Hz_verified=False, physical_jump_validated=False,
                  limitations=["One recorded posture; no motion/contact transitions integrated",
                               "Recomputed H/CG/CMM and all contact Jacobians from recorded q/dq/body state",
                               "Joint PD set to zero for screen; driver PD not characterized",
                               "Model mass/inertia not measured after arms removed and camera fitted",
                               "QP timing excludes sensors, model computation, communication and scheduling"])
    out.write_text(json.dumps(report, indent=2))
    print("SAVED", str(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-root", type=Path, default=Path("/home/khadas/BRUCE/BRUCE-OP"))
    p.add_argument("--fixture", type=Path, default=Path(__file__).with_name("recorded-standing-fixture.json"))
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("khadas-model-screen.json"))
    p.add_argument("--repetitions", type=int, default=100)
    a = p.parse_args()
    if not 1 <= a.repetitions <= 1000:
        p.error("repetitions must be 1..1000")
    run(a.robot_root, a.fixture, a.out, a.repetitions)
