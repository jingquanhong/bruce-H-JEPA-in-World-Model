"""Bounded step-wise IMU heading feedback. No hardware or shared memory."""
import math


def wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))


def yaw_and_rate(rotation,omega):
    flat=[float(v) for row in rotation for v in row]+[float(v) for v in omega]
    if len(flat)!=12 or not all(math.isfinite(v) for v in flat):
        raise ValueError('Invalid IMU orientation/rate')
    yaw=math.atan2(rotation[1][0],rotation[0][0])
    roll=math.atan2(rotation[2][1],rotation[2][2])
    cp=math.hypot(rotation[0][0],rotation[1][0])
    if cp<.5:raise ValueError('Heading singularity')
    rate=(math.sin(roll)*omega[1]+math.cos(roll)*omega[2])/cp
    return yaw,rate


class HeadingHold:
    def __init__(self):
        self.target=None;self.offset=0.
        self.kp=1.;self.kd=.2
        self.max_rate=math.radians(3.)
        self.max_offset=math.radians(5.)

    def reset(self,rotation):
        self.target=yaw_and_rate(rotation,[0.,0.,0.])[0];self.offset=0.

    def sample(self,rotation,omega,dt):
        if self.target is None:raise ValueError('Heading reference was not set')
        if not math.isfinite(dt) or not 0.<dt<=.3:raise ValueError('Invalid step interval')
        yaw,rate=yaw_and_rate(rotation,omega)
        error=wrap(self.target-yaw)
        demand=self.kp*error-self.kd*rate
        if abs(error)<math.radians(.3) and abs(rate)<math.radians(.5):demand=0.
        demand=max(-self.max_rate,min(self.max_rate,demand))
        old=self.offset
        self.offset=max(-self.max_offset,min(self.max_offset,old+demand*dt))
        applied=(self.offset-old)/dt
        return self.target+self.offset,applied,dict(target_rad=self.target,measured_rad=yaw,
            error_rad=error,gyro_yaw_rate_rad_s=rate,offset_rad=self.offset,
            applied_yaw_rate_rad_s=applied)
