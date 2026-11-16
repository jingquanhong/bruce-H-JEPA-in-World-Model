#!/usr/bin/env python3
"""Record an unassisted standing check or an explicitly experimental 8 cm trial."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bruce_env as be
import bruce_expert as bx
from collect_bruce import interp_target
from standing_wbc import WholeBodyController
from qualification import motor_current_from_joint_torque
from support_wbc import SupportController
from support_reference import plan, support_cop, transfer_screen

ROOT = Path(__file__).resolve().parents[3]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    h5path = out / f'{args.trial}.h5'
    video_path = out / f'{args.trial}.mp4'
    if any(p.exists() for p in (h5path, video_path, out / f'{args.trial}.json')):
        raise FileExistsError('Choose a new --out directory to preserve existing results')
    bx.URDF = str(ROOT / 'BRUCE-OP-main/Simulation/models/bruce/urdf/bruce_physical.urdf')
    bx.Z_STANCE = .025
    bx._PIN['model'] = None
    initial = bx._pose({'r': (0, .025), 'l': (0, .025)}, -.03, .445)
    if args.trial in ('stand','support'):
        if args.trial == 'support':
            initial = bx._pose({'r': (0,.025), 'l': (0,.025)}, -.03, .40)
        trajectory = np.repeat(initial[None], 601, axis=0)
    else:
        trajectory, _ = bx.gen_step_over(h_obs=.08, stride=.28, T_prep=1.,
                                         T_lipm=1., lift_extra=.08)
        initial = trajectory[0].copy()
    specs = [dict(half=(.025, .15, .04), pos=(.15, 0., .04))]
    scene = be.BruceScene(urdf=bx.URDF, box_specs=specs, camera=True,
                          backend=args.backend, default_armature=0., seed=0)
    scene.set_friction(1.)
    scene.set_gains(np.zeros(22), np.zeros(22))
    scene.robot.set_dofs_force_range(-np.array([0.]*6+[10.5]*10+[1.]*6),
                                     np.array([0.]*6+[10.5]*10+[1.]*6))
    scene.reset_robot(initial[2], initial[6:], initial[0])
    controller = (SupportController(initial,attitude_weight=100.) if args.trial=='support'
                  else WholeBodyController(initial))
    screen = None
    box_start = scene.box_pos(scene.boxes[0]).copy()
    # 复位会残留加速度缓存；先受控运行 10ms，再记录 IMU 和视频。
    for _ in range(2):
        state=scene.read_state()
        if args.trial=='support':
            state['support_cop']=support_cop(scene)
            com,feet,stance=plan(0.,controller)
            command=controller.command(state,com,feet,stance)
        else:
            controller.command(state, initial)
            command=controller.full_command
        scene.robot.control_dofs_force(command,
                                        dofs_idx_local=np.arange(6, 22))
        for _ in range(10):
            scene.step()
    columns = {k: [] for k in ('timestamp', 'pixels', 'proprio', 'action',
                'sim_base_pos', 'sim_base_quat', 'sim_control_force', 'sim_box_pos')}
    if args.trial=='support':
        columns.update({k:[] for k in ('sim_com','sim_com_velocity','sim_support_cop','sim_foot_force')})
    stop = 'duration_complete'
    ticks = 0
    max_base = max_leg = max_motor_current = 0.
    max_box_motion = 0.
    min_height = float(initial[2])
    min_up = 1.
    font = ImageFont.truetype(FONT, 16)
    writer = imageio.get_writer(str(video_path), fps=25, codec='libx264', quality=8)
    try:
        for frame in range(300 if args.trial=='support' else 250):
            state = scene.read_state()
            if args.trial=='support':
                state['support_cop']=support_cop(scene)
            rgb = scene.camera.render()[0]
            R = be.quat_to_mat(state['base_quat'])
            proprio = np.r_[state['dof_pos'][6:16], state['dof_vel'][6:16],
                             R.T @ (state['base_acc'] + [0, 0, 9.81]),
                             R.T @ state['base_ang'], state['dof_force'][6:16]]
            actions, forces = [], []
            for sub in range(4):
                t = ticks * .005
                current = scene.read_state()
                try:
                    if args.trial=='support':
                        current['support_cop']=support_cop(scene)
                        com,feet,stance=plan(t,controller)
                        cm,fm,_=plan(t-.005,controller); cp,fp,_=plan(t+.005,controller)
                        torque=controller.command(current,com,feet,stance,
                            (cp-cm)/.01,(cp-2*com+cm)/.005**2,
                            (fp-fm)/.01,(fp-2*feet+fm)/.005**2)
                        if t>=3. and screen is None:
                            screen=transfer_screen(current,controller.debug)
                            if not screen['passed']:
                                stop='support_transfer_not_ready'
                                break
                    else:
                        controller.command(current, interp_target(trajectory, t),
                            interp_target(trajectory, t-.005), interp_target(trajectory, t+.005), dt=.005)
                        torque = controller.full_command.copy()
                except RuntimeError as exc:
                    stop = str(exc)
                    break
                scene.robot.control_dofs_force(torque, dofs_idx_local=np.arange(6, 22))
                for _ in range(10):
                    scene.step()
                    actual = be.to_np(scene.robot.get_dofs_control_force())
                    if not np.isfinite(actual).all():
                        raise FloatingPointError('Non-finite applied force')
                    max_base = max(max_base, float(np.abs(actual[:6]).max()))
                    max_leg = max(max_leg, float(np.abs(actual[6:16]).max()))
                    max_motor_current = max(max_motor_current, float(np.abs(
                        motor_current_from_joint_torque(actual[6:16])).max()))
                    max_box_motion = max(max_box_motion, float(np.linalg.norm(
                        scene.box_pos(scene.boxes[0])-box_start)))
                actions.append(torque)
                forces.append(actual.copy())
                ticks += 1
            if len(actions) != 4:
                break
            columns['timestamp'].append(frame*.02)
            columns['pixels'].append(np.asarray(Image.fromarray(rgb).resize((64, 64))))
            columns['proprio'].append(proprio)
            columns['action'].append(np.asarray(actions).reshape(64))
            columns['sim_base_pos'].append(state['base_pos'])
            columns['sim_base_quat'].append(state['base_quat'])
            columns['sim_control_force'].append(forces)
            columns['sim_box_pos'].append(scene.box_pos(scene.boxes[0]))
            if args.trial=='support':
                # Reconstruct COM at o_t, not from the previous control tick.
                q=controller.q0.copy(); q[:3]=state['base_pos']
                q[3:7]=np.r_[state['base_quat'][1:],state['base_quat'][0]]
                v=np.zeros(controller.model.nv)
                R0=be.quat_to_mat(state['base_quat'])
                v[:3]=R0.T@state['base_vel']; v[3:6]=R0.T@state['base_ang']
                for iq,iv,slot in controller.mapping:
                    q[iq]=state['dof_pos'][6+slot]; v[iv]=state['dof_vel'][6+slot]
                import pinocchio as pin
                pin.centerOfMass(controller.model,controller.data,q,v)
                columns['sim_com'].append(controller.data.com[0].copy())
                columns['sim_com_velocity'].append(controller.data.vcom[0].copy())
                columns['sim_support_cop'].append(state['support_cop'])
                columns['sim_foot_force'].append(state['foot_force'])
            terminal = scene.read_state()
            min_height = min(min_height, float(terminal['base_pos'][2]))
            min_up = min(min_up, float(be.quat_to_mat(terminal['base_quat'])[2, 2]))
            if frame % 2 == 0:
                im = Image.fromarray(rgb)
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 640, 65), fill='#172332')
                title = ('STANDING CHECK - not obstacle crossing' if args.trial == 'stand'
                         else 'SUPPORT TRANSFER CHECK - not crossing' if args.trial=='support'
                         else '8cm EXPERIMENT - crossing NOT validated')
                draw.text((8, 5), title, font=font, fill='white')
                draw.text((8, 27), f't={frame*.02:.2f}s | base force {max_base:.1e} | leg peak {max_leg:.2f} Nm', font=font, fill='#c5e2f7')
                draw.text((8, 46), 'Free base | physical mesh feet | no assistance', font=font, fill='#c5e2f7')
                writer.append_data(np.asarray(im))
            if frame % 25 == 0:
                print(f'{args.trial}: {frame*.02:.2f}s, base z={terminal["base_pos"][2]:.4f}', flush=True)
            if min_height < .22 or min_up < .5:
                stop = 'loss_of_balance'
                break
    except BaseException:
        stop = 'aborted_by_exception'
        raise
    finally:
        card = Image.new('RGB', (640, 480), '#172332')
        draw = ImageDraw.Draw(card)
        draw.text((24, 160), 'STANDING CHECK COMPLETE' if args.trial == 'stand' and stop == 'duration_complete'
                  else 'NO VALIDATED OBSTACLE CROSSING', fill='white', font=font)
        draw.text((24, 200), stop[:68], fill='#ffd1ac', font=font)
        draw.text((24, 240), 'Status card - no additional simulation time', fill='#b3c7df', font=font)
        for _ in range(20):
            writer.append_data(np.asarray(card))
        writer.close()
    physics_ok = max_base < 1e-6 and max_leg <= 10.5+1e-4 and max_motor_current <= 30.+1e-3
    stand_ok = bool(args.trial == 'stand' and stop == 'duration_complete'
                    and getattr(controller,'failures',0) == 0 and physics_ok and min_height > .42 and min_up > .98)
    report = dict(trial=args.trial, stop_reason=stop, recorded_frames=len(columns['timestamp']),
        simulated_seconds=ticks*.005, warmup_seconds=.01,
        total_simulated_seconds=.01+ticks*.005,
        record_hz=50, control_hz=200, physics_hz=2000,
        stand_check_passed=stand_ok, autonomous_crossing_success=False,
        base_control_force_peak_abs=max_base, leg_torque_peak_abs_Nm=max_leg,
        max_box_displacement_m=max_box_motion, min_base_height_m=min_height,
        min_base_up_dot=min_up, qp_failures=getattr(controller,'failures',0),
        support_transfer_screen=screen,
        coupled_motor_current_peak_abs_A=max_motor_current,
        urdf=bx.URDF, urdf_sha256=hashlib.sha256(Path(bx.URDF).read_bytes()).hexdigest(),
        default_armature=0., leg_armature_kg_m2=.00182,
        armature_source='BRUCE_macros.REFLECTED_INERTIA; diagonal joint approximation',
        armature_calibrated_on_hardware=False,
        obstacle=dict(height=.08, thickness=.05, width=.30, center_x=.15),
        action_kind='four_consecutive_16_joint_torque_commands_Nm',
        action_order='right_leg_5,left_leg_5,right_arm_3,left_arm_3',
        action_timing='o_t before action[t,0:4]; each subcommand held 5ms',
        proprio='q10,dq10,body_specific_force3,body_gyro3,executed_leg_torque10',
        pixels='64x64 RGB, fixed external camera; not robot onboard vision',
        sensor_kind='simulator_truth_not_hardware_measurement',
        visual_hjepa_ready=False,
        limitations=['Controller uses simulator ground truth.',
                     'No verified mesh clearance or stable obstacle landing.',
                     'Current proxy uses BRUCE differential mapping and nominal 0.35 Nm/A; not hardware calibrated.',
                     'Reflected motor inertia is diagonal; differential coupling inertia is omitted.',
                     'Partial final control block, if any, is excluded from HDF5 rows.'])
    with h5py.File(h5path, 'w') as f:
        for key, values in columns.items():
            f.create_dataset(key, data=np.asarray(values), compression='gzip')
        n = len(columns['timestamp'])
        f.create_dataset('ep_len', data=[n])
        f.create_dataset('ep_offset', data=[0])
        f.create_dataset('ep_idx', data=np.zeros(n, dtype=int))
        f.create_dataset('step_idx', data=np.arange(n))
        f.attrs['metadata'] = json.dumps(report, ensure_ascii=False)
        f.attrs['purpose'] = 'physics_debug_not_success_demonstration'
    (out / f'{args.trial}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if stand_ok else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', choices=('stand', 'box08','support'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=('gpu', 'cpu'), default='gpu')
    raise SystemExit(run(parser.parse_args()))
