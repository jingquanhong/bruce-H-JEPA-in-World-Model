"""Slow bounded DCM trim feedback for a supported pure-lateral diagnostic.

Velocity/foot position are contact+encoder+IMU estimates, not absolute position.
No hardware/shared-memory access. Updates average about two walking steps.
"""
import math

def clip(value,limit):return max(-limit,min(limit,value))

class DriftFeedback:
 def __init__(self,yaw,anchor_xy,target_left_velocity):
  values=[yaw,*anchor_xy,target_left_velocity]
  if len(anchor_xy)!=2 or not all(math.isfinite(v) for v in values) or not 0<abs(target_left_velocity)<=.03:raise ValueError('Invalid lateral feedback reference')
  self.c=math.cos(yaw);self.s=math.sin(yaw);self.anchor=list(anchor_xy);self.target=target_left_velocity
  self.last=None;self.elapsed=0.;self.velocity_sum=[0.,0.];self.cross_sum=0.;self.integral=[0.,0.];self.bias=[0.,0.]
 def sample(self,now,velocity_xy,foot_midpoint_xy):
  if len(velocity_xy)!=2 or len(foot_midpoint_xy)!=2 or not all(math.isfinite(v) for v in [now,*velocity_xy,*foot_midpoint_xy]):raise ValueError('Invalid drift measurement')
  if self.last is None:self.last=now;return None
  dt=now-self.last
  if not 0<dt<.2:raise ValueError('Drift sample interval invalid')
  self.last=now
  vx=self.c*velocity_xy[0]+self.s*velocity_xy[1];vy=-self.s*velocity_xy[0]+self.c*velocity_xy[1]
  dx=foot_midpoint_xy[0]-self.anchor[0];dy=foot_midpoint_xy[1]-self.anchor[1]
  cross=self.c*dx+self.s*dy
  self.elapsed+=dt;self.velocity_sum[0]+=vx*dt;self.velocity_sum[1]+=vy*dt;self.cross_sum+=cross*dt
  if self.elapsed<.48:return None
  duration=self.elapsed;measured=[v/duration for v in self.velocity_sum];cross=self.cross_sum/duration
  errors=[clip(-measured[0],.10),clip(self.target-measured[1],.10)]
  target=[]
  for i,error in enumerate(errors):
   error=0. if abs(error)<.005 else error
   self.integral[i]=clip(self.integral[i]+error*duration,.06)
   value=.10*error+.04*self.integral[i]+(.04*clip(-cross,.05) if i==0 else 0.)
   target.append(clip(value,.008))
   self.bias[i]=clip(self.bias[i]+clip(target[i]-self.bias[i],.003*duration),.008)
  self.elapsed=0.;self.velocity_sum=[0.,0.];self.cross_sum=0.
  return {'forward_bias_m':self.bias[0],'lateral_bias_m':self.bias[1],'mean_forward_velocity_m_s':measured[0],'mean_left_velocity_m_s':measured[1],'estimated_cross_track_m':cross,'target_left_velocity_m_s':self.target,'mean_window_s':duration,'bias_target_m':target}
