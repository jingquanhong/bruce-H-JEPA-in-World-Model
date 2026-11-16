"""Run the real cross-step QP against private recorded state, never actuators."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, types, contextlib, ast, tempfile, time
from pathlib import Path
sys.path.insert(0, '/home/khadas/BRUCE/BRUCE-OP')
import numpy as np
import Settings.BRUCE_data as rds
import Play.Walking.walking_macros as walking
walking.zm_l = walking.zm_r = .015
from Util import crossstep_high_level as candidate
from Util.crossstep_targets import landing_targets, add_target_objective

root = Path(__file__).resolve().parent
fixture = json.loads(Path('/home/khadas/BRUCE/diagnostics/control-warmup-fixture.json').read_text())
temporary = tempfile.TemporaryDirectory(prefix='crossstep-private-')
tuning = Path(temporary.name)/'tuning.json'
tuning.write_text(json.dumps(dict(forward_bias_m=-.002, minimum_width_m=.10, lateral_bias_m=-.006)))
os.environ['BRUCE_STATIONARY_TUNING_FILE'] = str(tuning)
os.environ['BRUCE_STATIONARY_SMOOTH_STOP'] = '1'

# Directional invariants in the robot's yaw frame; feet never cross.
for yaw in (0., .7, -1.4):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    for side in (-1, 1):
        for direction in (-1, 1):
            swing, support = R @ [0., side*.07], R @ [0., -side*.07]
            goals = landing_targets(swing, support, R, direction)
            assert np.allclose(R.T @ (goals[0]-swing), [0., direction*.03])
            assert np.allclose(R.T @ (goals[1]-support), [0., direction*.03])
            assert np.allclose(R.T @ (goals[2]-swing), [0., direction*.06])
for bad in (.0, .041, float('nan')):
    try: landing_targets([0., -.07], [0., .07], np.eye(2), -1, step=bad)
    except ValueError: pass
    else: raise AssertionError('Invalid stride accepted')
try: landing_targets([0., -.09], [0., .09], np.eye(2), -1)
except ValueError: pass
else: raise AssertionError('Unreachable widening step accepted')

def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
base = Path('/home/khadas/BRUCE/BRUCE-OP/Util')
original = functions(base/'sidestep_high_level.py')
modified = functions(base/'crossstep_high_level.py')
for name in ('cost_update', 'constraint_update', 'get_swing_ref', 'get_swing_traj', 'get_DCM_nom'):
    assert original[name] == modified[name], name
parent_a, parent_b = functions(base/'stationary_sidestep_test.py'), functions(base/'stationary_crossstep_test.py')
for name in ('motor_worker', 'guard_worker', 'preload_worker', 'check_tilt'):
    assert parent_a[name] == parent_b[name], name
class_a = next(n for n in ast.parse((base/'stationary_sidestep_test.py').read_text()).body if isinstance(n, ast.ClassDef))
class_b = next(n for n in ast.parse((base/'stationary_crossstep_test.py').read_text()).body if isinstance(n, ast.ClassDef))
methods_a = {n.name: ast.dump(n, include_attributes=False) for n in class_a.body if isinstance(n, ast.FunctionDef)}
methods_b = {n.name: ast.dump(n, include_attributes=False) for n in class_b.body if isinstance(n, ast.FunctionDef)}
for name in ('watch', 'perform_steps', 'command', 'close'):
    assert methods_a[name] == methods_b[name], name

class Complete(Exception): pass
reports = []
for direction in (-1., 1.):
    clock = [0.]; commands = []; targets = []; solved = []; active = [False]
    def tick(robot):
        clock[0] += .002
        if clock[0] > 15: raise RuntimeError('Private test time budget exceeded')
        return clock[0]
    class Segment:
        def __init__(self, name, values):
            self.name = name; self.reads = 0
            self.data = {k: np.asarray(v, dtype=float).copy() for k, v in values.items()}
        def get(self):
            d = {k: v.copy() for k, v in self.data.items()}
            if self.name == 'USER_COMMAND':
                self.reads += 1
                d['mode'] = np.array([0. if self.reads <= 2 else 1.])
                d['com_xy_velocity'] = np.array([0., direction*.05])
                d['foot_clearance'] = np.array([.08])
            return d
        def set(self, data, opt=None):
            self.data.update({k: np.asarray(v).copy() for k, v in data.items()})
            if self.name == 'PLANNER_COMMAND' and self.data['phase'][0] in (1, 2):
                assert all(np.isfinite(v).all() for v in self.data.values())
                commands.append({k: v.tolist() for k, v in self.data.items()})
                if len(commands) >= 100: raise Complete()
    private = types.SimpleNamespace(**{k: Segment(k, v) for k, v in fixture.items()})
    private.THREAD_STATE.data = {k: np.ones(1) for k in private.THREAD_STATE.data}
    old = (candidate.MM, rds.MM, rds.safe_input, rds.BRUCE.get_time, candidate.time,
           candidate.add_target_objective, candidate.osqp)
    actual_osqp = candidate.osqp.OSQP
    class Solver:
        def __init__(self): self.inner = actual_osqp()
        def setup(self, **kw):
            self.A = kw['A'].copy(); self.l = kw['l']; self.u = kw['u']; self.inner.setup(**kw)
        def update(self, **kw):
            if 'Ax' in kw: self.A.data = kw['Ax'].copy()
            if 'l' in kw: self.l = kw['l'].copy()
            if 'u' in kw: self.u = kw['u'].copy()
            self.inner.update(**kw)
        def solve(self):
            solve_start = time.perf_counter()
            sol = self.inner.solve()
            solve_elapsed = time.perf_counter()-solve_start
            assert sol.info.status == 'solved', sol.info.status
            ax = self.A @ sol.x
            violation = max(float(np.max(self.l-ax)), float(np.max(ax-self.u)), 0.)
            assert violation < .002, violation
            if active[0]:
                solved.append(dict(foothold=sol.x[:2].tolist(), target=targets[-1][0],
                    target_error_m=float(np.linalg.norm(sol.x[:2]-np.array(targets[-1][0]))),
                    max_original_constraint_violation=violation,
                    solve_elapsed_s=solve_elapsed, iterations=sol.info.iter))
            return sol
    def objective(P, q, goals):
        targets.append(goals.tolist()); active[0] = True
        return add_target_objective(P, q, goals)
    try:
        candidate.MM = private; rds.MM = private; rds.safe_input = lambda _: 'y'
        rds.BRUCE.get_time = tick; candidate.time = types.SimpleNamespace(sleep=lambda _: None)
        candidate.add_target_objective = objective; candidate.osqp = types.SimpleNamespace(OSQP=Solver)
        with (root/'crossstep-private.log').open('a') as log, contextlib.redirect_stdout(log):
            try: candidate.main_loop()
            except Complete: pass
        assert len(commands) == 100 and len(solved) >= 20
        assert set(int(r['phase'][0]) for r in commands) == {1, 2}
        reports.append(dict(direction=direction, hardware_actuation=False,
            published_commands=len(commands), qp_samples=len(solved),
            maximum_constraint_violation=max(x['max_original_constraint_violation'] for x in solved),
            maximum_qp_solve_elapsed_s=max(x['solve_elapsed_s'] for x in solved),
            maximum_qp_iterations=max(x['iterations'] for x in solved),
            median_soft_target_error_m=float(np.median([x['target_error_m'] for x in solved])),
            first_target=targets[0], first_solution=solved[0]))
    finally:
        (candidate.MM, rds.MM, rds.safe_input, rds.BRUCE.get_time, candidate.time,
         candidate.add_target_objective, candidate.osqp) = old

(root/'crossstep-private-results.json').write_text(json.dumps(dict(
    result='PASS_RECORDED_STATE_PRIVATE_QP_ONLY', cases=reports,
    note='Fixed recorded state, not a physical dynamics simulation or proof of translation.'), indent=2))
print('PASS: explicit 3 cm directional foot targets, both support phases; original constraints and guards retained.')
temporary.cleanup()
