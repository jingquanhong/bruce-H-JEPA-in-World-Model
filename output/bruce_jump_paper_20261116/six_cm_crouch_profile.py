"""Slow double-support crouch, screened up to 6 cm body descent; no flight."""
import math


def checked_depth(value):
    value=float(value)
    if not math.isfinite(value) or not .005<=value<=.06:
        raise ValueError('Body crouch depth must be 0.005..0.06 m')
    return value


def ramp_seconds(depth):
    return max(2.,100.*checked_depth(depth))


def duration(depth):
    return 2*ramp_seconds(depth)+.5+2.


def smooth(u):
    u=min(1.,max(0.,u))
    return u**3*(10.-15.*u+6.*u*u)


def offset_at(elapsed,depth):
    depth=checked_depth(depth);r=ramp_seconds(depth)
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid elapsed time')
    if elapsed<r:return -depth*smooth(elapsed/r)
    if elapsed<r+.5:return -depth
    if elapsed<2*r+.5:return -depth*(1.-smooth((elapsed-r-.5)/r))
    return 0.


def offset_velocity(elapsed,depth):
    depth=checked_depth(depth);r=ramp_seconds(depth)
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid elapsed time')
    if elapsed<r:
        u=elapsed/r
        return -depth*30*u*u*(1-u)*(1-u)/r
    if r+.5<=elapsed<2*r+.5:
        u=(elapsed-r-.5)/r
        return depth*30*u*u*(1-u)*(1-u)/r
    return 0.


class PreparationCycle:
    def __init__(self,now,depth,initial_com_z):
        if not math.isfinite(now) or not math.isfinite(initial_com_z):raise ValueError('Invalid start state')
        self.started=now;self.depth=checked_depth(depth);self.initial_z=initial_com_z
        self.minimum_z=initial_com_z;self.maximum_tilt=0.;self.last_time=now;self.contact_lost=None

    def update(self,now,com_z,tilt,contacts):
        if not all(math.isfinite(v) for v in [now,com_z,tilt]) or now<self.last_time:
            raise RuntimeError('Invalid crouch feedback/time')
        if len(contacts)!=4 or not all(v in (0,1) for v in contacts):raise RuntimeError('Invalid contact feedback')
        self.last_time=now;self.minimum_z=min(self.minimum_z,com_z);self.maximum_tilt=max(self.maximum_tilt,tilt)
        if tilt>8.:raise RuntimeError('Crouch tilt exceeded 8 degrees')
        grounded=any(contacts[:2]) and any(contacts[2:])
        if grounded:self.contact_lost=None
        elif self.contact_lost is None:self.contact_lost=now
        elif now-self.contact_lost>.1:raise RuntimeError('Unexpected loss of double support')
        elapsed=now-self.started
        return offset_at(elapsed,self.depth),elapsed>=duration(self.depth)

    def result(self,final_z):
        excursion=self.initial_z-self.minimum_z
        recovered=abs(final_z-self.initial_z)<.004
        response=excursion>=self.depth*.65
        return dict(result='CROUCH_RESPONSE_SEEN' if response and recovered else 'CROUCH_RESPONSE_INCOMPLETE',
                    commanded_body_crouch_depth_m=self.depth,estimated_com_excursion_m=excursion,
                    estimated_final_z_error_m=final_z-self.initial_z,maximum_tilt_deg=self.maximum_tilt,
                    real_jump_attempted=False,ground_truth=False,feet_expected_in_contact=True)
