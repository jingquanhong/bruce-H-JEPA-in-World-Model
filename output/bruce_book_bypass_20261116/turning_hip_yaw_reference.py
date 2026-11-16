"""Preserve native turning IK, retain saved stance-yaw anchoring otherwise."""
import numpy as np
from Util.hip_yaw_reference import checked_reference,hold_stance_hip_yaw

class TurningHipYawReference:
 def __init__(self,reference):self.reference=np.asarray(reference).copy();self.turn_pending=False
 def apply(self,positions,velocities,phase,measured_positions,commanded_yaw_rate):
  if phase not in (0,1,2) or not np.isfinite(commanded_yaw_rate):raise ValueError('Invalid turning reference input')
  if phase in (1,2) and abs(commanded_yaw_rate)>1e-6:self.turn_pending=True
  if self.turn_pending and phase==0:
   self.reference=checked_reference(measured_positions);self.turn_pending=False
  if not self.turn_pending:hold_stance_hip_yaw(positions,velocities,phase,self.reference)
