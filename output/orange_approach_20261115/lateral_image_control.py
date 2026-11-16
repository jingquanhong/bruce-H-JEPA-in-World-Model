"""Small image-based lateral foothold correction; no hardware access.

This is a landing-trim command in meters, not a calibrated lateral velocity.
Positive trim requests a correction to robot left, to be checked on hardware.
"""
import math


class BearingTrimPD:
    def __init__(self,kp=.025,kd=.001,maximum=.002,rate=.001):
        self.kp,self.kd,self.maximum,self.rate=kp,kd,maximum,rate
        self.reset()

    def reset(self):
        self.value=0.;self.previous_time=None;self.previous_error=None;self.derivative=0.

    def update(self,bearing_deg,now):
        if not all(math.isfinite(v) for v in (bearing_deg,now)) or abs(bearing_deg)>8:
            self.reset();return 0.
        error=-math.radians(bearing_deg)  # image left -> robot-left correction
        deadband=math.radians(1.)
        error=math.copysign(max(0.,abs(error)-deadband),error)
        dt=.1 if self.previous_time is None else now-self.previous_time
        if not .01<=dt<=.35:
            self.derivative=0.;dt=.1
        elif self.previous_error is not None:
            self.derivative=.8*self.derivative+.2*(error-self.previous_error)/dt
        requested=max(-self.maximum,min(self.maximum,self.kp*error+self.kd*self.derivative))
        self.value+=max(-self.rate*dt,min(self.rate*dt,requested-self.value))
        self.previous_error,self.previous_time=error,now
        return self.value


def self_test():
    c=BearingTrimPD();last=0.
    for i in range(50):
        value=c.update(-5.,10+i*.1)
        assert 0<=value<=.002 and abs(value-last)<=.000100001
        last=value
    assert value>.001
    assert c.update(12.,16.)==0.
    for i in range(50):value=c.update(5.,20+i*.1)
    assert -.002<=value<-.001
    assert c.update(float('nan'),30.)==0.
    for i in range(30):value=c.update(0.,40+i*.1)
    assert value==0.
    print('Lateral image trim checks passed: sign, bound, slew, off-axis and invalid-data reset.')


if __name__=='__main__':self_test()
