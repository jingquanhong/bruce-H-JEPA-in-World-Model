"""Experimental weight-transfer reference and simulator-only support diagnostics."""
import numpy as np
import bruce_env as be


def smooth(s):
    s = np.clip(s,0.,1.)
    return s*s*s*(10+s*(-15+6*s))


def support_cop(scene):
    centers=[]
    contacts=scene.get_contacts()
    for name in be.ANKLE_LINKS:
        ankle=scene.get_link_pos(name)
        center=ankle.copy(); center[2]-=.025
        if contacts is not None:
            positions=[]; weights=[]
            for i,(ga,gb) in enumerate(zip(contacts['geom_a'],contacts['geom_b'])):
                if scene.geom_link.get(int(ga))==name:
                    force=contacts['force_a'][i]
                elif scene.geom_link.get(int(gb))==name:
                    force=contacts['force_b'][i]
                else: continue
                positions.append(contacts['position'][i]); weights.append(abs(float(force[2])))
            if sum(weights)>1.:
                center=np.average(positions,axis=0,weights=weights)
        centers.append(center.tolist())
    return centers


def plan(t, controller, lift=.10, shift_time=2., support_offset=0.):
    feet = np.array([p.translation.copy() for p in controller.feet0])
    com = controller.com0.copy()
    com[0] = .005
    com[1] = (feet[0,1]+support_offset)*smooth((t-.5)/shift_time)
    start = .5+shift_time+.5
    if t < start:
        stance = [0,1]
    elif t < start+.6:
        feet[1,2] += lift*smooth((t-start)/.6); stance = [0]
    elif t < start+1.3:
        feet[1,2] += lift; stance = [0]
    elif t < start+1.9:
        feet[1,2] += lift*(1-smooth((t-start-1.3)/.6)); stance = [0]
    else:
        stance = [0,1]
    return com,feet,stance


def transfer_screen(state, debug):
    """Conservative necessary conditions; passing never proves safe lifting."""
    com=np.asarray(debug['com']); velocity=np.asarray(debug['vcom'])
    cop=np.asarray(state['support_cop'][0])
    omega=np.sqrt(9.81/max(com[2]-cop[2],.1))
    capture=com[:2]+velocity[:2]/omega
    checks=dict(
        capture_near_support=bool(abs(capture[1]-cop[1])<.0015),
        lateral_speed_low=bool(abs(velocity[1])<.01),
        stance_loaded=bool(sum(state['foot_force'][:2])>25.),
        swing_unloaded=bool(sum(state['foot_force'][2:])<1.),
    )
    return dict(passed=all(checks.values()),checks=checks,
                capture_point=capture.tolist(),support_cop=cop.tolist(),
                lateral_speed_m_s=float(velocity[1]),
                note='Necessary-condition screen using simulator truth; not a safety certificate.')

