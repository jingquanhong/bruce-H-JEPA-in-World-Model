"""Pure range-control policy and command validation; no hardware access."""
from collections import deque
import math


def checked_command(data, now, session):
    if data.get('session_id') != session:
        raise ValueError('vision session mismatch')
    command_time=float(data['monotonic'])
    measurement_time=float(data['measurement_monotonic'])
    speed=float(data['forward_velocity_m_s'])
    if not all(math.isfinite(v) for v in (now,command_time,measurement_time,speed)):
        raise ValueError('nonfinite vision command')
    if not 0 <= now-command_time <= .30:
        raise ValueError('vision command stale')
    mode=data['mode']
    if mode not in ('walk','balance') or not 0 <= speed <= .05:
        raise ValueError('invalid vision mode or speed')
    if mode=='walk' and not 0 <= now-measurement_time <= .35:
        raise ValueError('target measurement stale')
    if mode=='balance' or speed < .005:
        return 0.
    return speed


class RangeFilter:
    def __init__(self):
        self.values=deque()
        self.last_sequence=None

    def add(self, sequence, timestamp, distance, now):
        if sequence==self.last_sequence:
            return
        self.last_sequence=sequence
        if (not all(math.isfinite(v) for v in (timestamp,distance,now))
                or not 0 <= now-timestamp <= .35 or not .3 <= distance <= 4.):
            return
        if self.values and abs(distance-self.values[-1][1]) > .15:
            self.values.clear()
            return
        self.values.append((timestamp,distance))

    def read(self, now):
        while self.values and now-self.values[0][0] > .6:
            self.values.popleft()
        if len(self.values)<3 or not 0 <= now-self.values[-1][0] <= .35:
            return None
        distances=sorted(v[1] for v in self.values)
        if distances[-1]-distances[0] > .12:
            return None
        n=len(distances)
        median=(distances[n//2]+distances[(n-1)//2])/2
        return median,self.values[-1][0]


class RangePID:
    def __init__(self, target=1., kp=.12, ki=0., kd=.025):
        self.target=target
        self.kp,self.ki,self.kd=kp,ki,kd
        self.reset()

    def reset(self):
        self.previous_time=None
        self.previous_distance=None
        self.derivative=0.
        self.integral=0.
        self.speed=0.
        self.in_band=0

    def update(self, distance, now):
        if not all(math.isfinite(v) for v in (distance,now)) or not .3 <= distance <= 4.:
            self.reset()
            return dict(mode='balance',speed=0.,reason='invalid_range',reached=False)
        error=distance-self.target
        if distance < self.target-.10:
            self.reset()
            return dict(mode='balance',speed=0.,reason='too_close',reached=False)
        if distance < self.target-.05:
            self.reset()
            return dict(mode='balance',speed=0.,reason='inside_target_distance',reached=False)
        if error <= .05:
            self.in_band+=1
            self.speed=0.
            return dict(mode='balance',speed=0.,reason='stop_band',reached=self.in_band>=3)
        self.in_band=0
        dt=.1 if self.previous_time is None else now-self.previous_time
        if not .01 <= dt <= .5:
            self.derivative=0.
            dt=.1
        elif self.previous_distance is not None:
            measured_derivative=(distance-self.previous_distance)/dt
            self.derivative=.8*self.derivative+.2*measured_derivative
        self.previous_time,self.previous_distance=now,distance
        if self.speed < .05:
            self.integral=max(0.,min(.5,self.integral+error*dt))
        requested=self.kp*error+self.ki*self.integral+self.kd*self.derivative
        requested=max(0.,min(.05,requested))
        # Gentle acceleration; decreasing the request is immediate.
        self.speed=min(requested,self.speed+.03*dt)
        if self.speed < .005:
            return dict(mode='balance',speed=0.,reason='starting_or_slow',reached=False)
        return dict(mode='walk',speed=self.speed,reason='pid',reached=False)
