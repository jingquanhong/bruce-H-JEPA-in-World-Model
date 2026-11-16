"""One supported side-step pair, with explicit load transfer and foot paths.

References remain relative to the current support foot to cancel common-mode
odometry drift. This module never opens hardware or writes shared memory.
"""
import numpy as np

TRANSFER_S = 1.2
SWING_S = .24
EARLIEST_TOUCHDOWN_S = .20
TOUCHDOWN_WAIT_S = .04
SETTLE_S = .12
CENTER_S = .8
COM_HEIGHT_M = .365


def smooth(u):
    u = float(np.clip(u, 0., 1.))
    return u**3*(10.-15.*u+6.*u*u), 30.*u*u*(1.-u)**2


def yaw_rotation(rotation):
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError('Invalid orientation')
    forward = np.array([r[0, 0], r[1, 0], 0.])
    n = np.linalg.norm(forward)
    if n < .5: raise ValueError('Invalid heading')
    forward /= n
    return np.column_stack([forward, [-forward[1], forward[0], 0.], [0., 0., 1.]])


def validate(snapshot):
    for key in ('com', 'velocity', 'right_ankle', 'left_ankle', 'right_foot', 'left_foot'):
        value = np.asarray(snapshot[key])
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Invalid '+key)
    contacts = np.asarray(snapshot['contacts'])
    if contacts.shape != (4,) or not np.isfinite(contacts).all():
        raise ValueError('Invalid contacts')
    yaw_rotation(snapshot['rotation'])


def base_plan(snapshot, rotation):
    return dict(mode=np.zeros(1), phase=np.zeros(1),
        com_position=np.array(snapshot['com'], copy=True), com_velocity=np.zeros(3),
        body_rot_matrix=rotation.copy(), body_ang_rate=np.zeros(3),
        right_foot_position=np.array(snapshot['right_ankle'], copy=True),
        left_foot_position=np.array(snapshot['left_ankle'], copy=True),
        right_foot_velocity=np.zeros(3), left_foot_velocity=np.zeros(3),
        right_foot_rot_matrix=yaw_rotation(snapshot.get('right_rotation', rotation)),
        left_foot_rot_matrix=yaw_rotation(snapshot.get('left_rotation', rotation)),
        right_foot_ang_rate=np.zeros(3), left_foot_ang_rate=np.zeros(3))


def center(snapshot):
    goal = .5*(np.asarray(snapshot['right_foot'])+snapshot['left_foot'])
    goal = goal.copy(); goal[2] += COM_HEIGHT_M
    return goal


class CrossStep:
    def __init__(self, snapshot, direction, now, distance=.04, lift=.015, support_lean_deg=2.,
                 tracking_integral_gain=1.5):
        validate(snapshot)
        if direction not in (-1, 1) or not .025 <= distance <= .04 or not .01 <= lift <= .02:
            raise ValueError('Cross-step outside supported diagnostic envelope')
        if not np.isfinite(support_lean_deg) or not 0. <= support_lean_deg <= 3.:
            raise ValueError('Support lean must be finite and within 0..3 degrees')
        if not np.isfinite(tracking_integral_gain) or not 0. <= tracking_integral_gain <= 2.:
            raise ValueError('Tracking integral gain must be finite and within 0..2 per second')
        if not any(snapshot['contacts'][:2]) or not any(snapshot['contacts'][2:]):
            raise ValueError('Both feet must contact before cross-step')
        self.rotation = yaw_rotation(snapshot['rotation'])
        self.foot_rotations = {side: yaw_rotation(snapshot.get(side+'_rotation',self.rotation))
                               for side in ('right','left')}
        self.left_axis = self.rotation[:, 1]
        width = float(self.left_axis @ (np.asarray(snapshot['left_ankle'])-snapshot['right_ankle']))
        if not .10 <= width <= .17:
            raise ValueError('Initial ankle separation must be 10..17 cm')
        self.distance = min(distance, .195-width)
        if self.distance < .025:
            raise ValueError('Insufficient room for a visible cross-step')
        self.direction = direction; self.lift = lift
        self.lean = float(np.deg2rad(support_lean_deg)); self.roll = 0.
        self.leading = 'left' if direction > 0 else 'right'
        self.leg = self.leading; self.leg_index = 0
        self.events = []; self.done = False; self.aborted = None; self.stop_after_landing = False
        self.air_seen = False; self.stage = None
        self.contact_lost_since = None
        self.integral_gain = float(tracking_integral_gain)
        self.integral_offset = 0.
        self.last_sample_time = now
        self.last_com_reference = np.array(snapshot['com'], copy=True)
        self._transfer(snapshot, now)

    def _event(self, name, now, **details):
        self.events.append(dict(stage=name, time=now, **details))

    def _set_stage(self, name, now):
        self.stage = name; self.t0 = now; self.contact_lost_since = None
        self._event(name, now, leg=self.leg)

    def _support(self, snapshot, ankle=False):
        side = 'right' if self.leg == 'left' else 'left'
        return np.asarray(snapshot[side+('_ankle' if ankle else '_foot')])

    def _contact(self, snapshot, side):
        return bool(any(snapshot['contacts'][:2] if side == 'right' else snapshot['contacts'][2:]))

    def _require_contact(self, present, now, message):
        if present:
            self.contact_lost_since = None
        elif self.contact_lost_since is None:
            self.contact_lost_since = now
        elif now-self.contact_lost_since > .1:
            raise RuntimeError(message)

    def _transfer(self, snapshot, now):
        self.start_com_offset = self.last_com_reference-self._support(snapshot)
        self.integral_offset = 0.
        forward = self.rotation[:,0]
        self.support_com_offset = forward*float(forward @ self.start_com_offset)
        self.support_com_offset[2] = COM_HEIGHT_M
        self.roll_start = self.roll
        self.roll_target = self.lean if self.leg=='left' else -self.lean
        self._set_stage('transfer', now)

    def _center(self, snapshot, now, reason=None):
        self.center_start_offset = self.last_com_reference-center(snapshot)
        self.center_start_roll = self.roll
        self.aborted = reason
        self._set_stage('recenter', now)

    def _tracking_offset(self, snapshot, dt, active):
        # Integrate the real support-relative goal error, not the augmented
        # reference error. Saturation is an explicit anti-windup bound.
        old = self.integral_offset
        if active and dt > 0.:
            goal = self._support(snapshot)+self.support_com_offset
            error = float(self.left_axis @ (goal-np.asarray(snapshot['com'])))
            self.integral_offset = float(np.clip(old+self.integral_gain*error*dt,-.02,.02))
        rate = (self.integral_offset-old)/dt if dt > 0. else 0.
        return self.left_axis*self.integral_offset, self.left_axis*rate

    def sample(self, snapshot, now, continue_walking=True):
        validate(snapshot)
        dt = float(np.clip(now-self.last_sample_time, 0., .02))
        self.last_sample_time = now
        plan = base_plan(snapshot, self.rotation)
        if not continue_walking: self.stop_after_landing = True
        elapsed = max(0., now-self.t0)
        roll_rate = 0.
        if self.stage == 'transfer':
            if not continue_walking:
                self._center(snapshot, now, 'operator_requested_balance')
                return self.sample(snapshot, now, False)
            # A foot switch measures loading, not geometric floor contact.
            # Unloading either foot in double support is expected; the newly
            # selected support must carry load before the swing can start.
            self._require_contact(bool(any(snapshot['contacts'])), now,
                                  'Both feet lost contact during weight transfer')
            s, ds = smooth(elapsed/TRANSFER_S)
            self.roll = (1.-s)*self.roll_start+s*self.roll_target
            roll_rate = ds/TRANSFER_S*(self.roll_target-self.roll_start)
            offset = (1.-s)*self.start_com_offset+s*self.support_com_offset
            plan['com_position'] = self._support(snapshot)+offset
            plan['com_velocity'] = ds/TRANSFER_S*(self.support_com_offset-self.start_com_offset)
            correction, correction_rate = self._tracking_offset(snapshot,dt,elapsed>=.8*TRANSFER_S)
            plan['com_position'] += correction
            plan['com_velocity'] += correction_rate
            error = np.asarray(snapshot['com'])-(self._support(snapshot)+self.support_com_offset)
            ready = (abs(float(self.left_axis @ error)) < .008
                     and np.linalg.norm(np.asarray(snapshot['velocity'])[:2]) < .03
                     and self._contact(snapshot, 'right' if self.leg=='left' else 'left'))
            if elapsed >= TRANSFER_S and ready:
                self.swing_offset = np.asarray(snapshot[self.leg+'_ankle'])-self._support(snapshot, True)
                signed = float(self.left_axis @ self.swing_offset)
                target_signed = signed+self.direction*self.distance
                if not .10 <= abs(target_signed) <= .198 or signed*target_signed <= 0:
                    self._center(snapshot, now, 'foot_spacing_preflight_failed')
                else:
                    self.air_seen = False; self._set_stage('swing', now)
                    self._event('swing_target', now, relative_start=self.swing_offset.tolist(),
                        relative_landing=(self.swing_offset+self.direction*self.distance*self.left_axis).tolist(),
                        lateral_step_m=self.direction*self.distance)
            elif elapsed > TRANSFER_S+.7:
                self._center(snapshot, now, 'weight_transfer_not_reached')
        elif self.stage in ('swing', 'touchdown'):
            support_side = 'right' if self.leg == 'left' else 'left'
            self._require_contact(self._contact(snapshot, support_side), now,
                                  'Support foot lost contact during cross-step')
            plan['mode'][:] = 1; plan['phase'][:] = 1 if support_side == 'right' else 2
            plan['com_position'] = self._support(snapshot)+self.support_com_offset
            correction, correction_rate = self._tracking_offset(snapshot,dt,True)
            plan['com_position'] += correction
            plan['com_velocity'] = correction_rate
            u = float(np.clip(elapsed/SWING_S, 0., 1.)) if self.stage == 'swing' else 1.
            s, ds = smooth(u)
            relative = self.swing_offset+self.direction*self.distance*self.left_axis*s
            velocity = self.direction*self.distance*self.left_axis*ds/SWING_S
            if self.stage == 'swing':
                relative = relative.copy(); velocity = velocity.copy()
                relative[2] += 16.*self.lift*u*u*(1.-u)**2-.002*s
                velocity[2] += 32.*self.lift*u*(1.-u)*(1.-2.*u)/SWING_S-.002*ds/SWING_S
                if not self._contact(snapshot, self.leg): self.air_seen = True
                if elapsed >= EARLIEST_TOUCHDOWN_S and self.air_seen and self._contact(snapshot,self.leg):
                    self._set_stage('settle',now)
                elif elapsed >= SWING_S:
                    self._set_stage('touchdown', now)
            else:
                relative = relative.copy(); relative[2] -= .002+min(.002,.05*elapsed)
                velocity = np.zeros(3); velocity[2] = -.05 if elapsed < TOUCHDOWN_WAIT_S else 0.
                if self._contact(snapshot, self.leg):
                    if not self.air_seen:
                        self._center(snapshot, now, 'no_liftoff_detected')
                    else:
                        self._set_stage('settle', now)
                elif elapsed > TOUCHDOWN_WAIT_S:
                    raise RuntimeError('Swing foot failed to regain contact')
            plan[self.leg+'_foot_position'] = self._support(snapshot, True)+relative
            plan[self.leg+'_foot_velocity'] = velocity
        elif self.stage == 'settle':
            plan['com_position'] = self._support(snapshot)+self.support_com_offset
            correction, correction_rate = self._tracking_offset(snapshot,dt,True)
            plan['com_position'] += correction
            plan['com_velocity'] = correction_rate
            support_side = 'right' if self.leg=='left' else 'left'
            self._require_contact(self._contact(snapshot, support_side), now,
                                  'Loaded support foot lost contact during settling')
            if elapsed >= SETTLE_S:
                if self.leg_index == 0 and not self.stop_after_landing:
                    self.leg_index = 1; self.leg = 'right' if self.leading == 'left' else 'left'
                    self._transfer(snapshot, now)
                else: self._center(snapshot, now)
        elif self.stage == 'recenter':
            s, ds = smooth(elapsed/CENTER_S)
            self.roll = (1.-s)*self.center_start_roll
            roll_rate = -ds/CENTER_S*self.center_start_roll
            plan['com_position'] = center(snapshot)+(1.-s)*self.center_start_offset
            plan['com_velocity'] = -ds/CENTER_S*self.center_start_offset
            if elapsed >= CENTER_S:
                self.done = True; self._set_stage('done', now)
        elif self.stage == 'done':
            self.roll = 0.
            plan['com_position'] = center(snapshot)
        else: raise RuntimeError('Unknown cross-step stage')
        if not all(np.isfinite(v).all() for v in plan.values()):
            raise RuntimeError('Nonfinite cross-step reference')
        for side in ('right','left'):
            plan[side+'_foot_rot_matrix'] = self.foot_rotations[side].copy()
        c,s = np.cos(self.roll),np.sin(self.roll)
        plan['body_rot_matrix'] = self.rotation @ np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
        plan['body_ang_rate'] = np.array([roll_rate,0.,0.])
        self.last_com_reference = plan['com_position'].copy()
        return plan
