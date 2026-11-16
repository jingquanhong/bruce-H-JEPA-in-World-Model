"""Derive an isolated stance supervisor. Original files remain untouched."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
s = (root / "stance_supervisor_original.py").read_text()
original = s
needle = "    def command(self,mode,forward_velocity=0.):\n"
assert s.count(needle) == 1
addition = '''        # Independent, slow double-support check; never enters flight.
        from Util.jump_preparation_profile import PreparationCycle, checked_depth
        cycle=getattr(self,'preparation_cycle',None)
        request=self.folder/'preparation-request.json'
        if cycle is None and request.exists() and self.phase=='holding_balance':
            state=self.mm.ESTIMATOR_STATE.get();sense=self.mm.SENSE_STATE.get()
            tilt=float(np.degrees(np.arccos(np.clip(state['body_rot_matrix'][2,2],-1,1))))
            contacts=sense['foot_contacts'].tolist()
            if tilt>3 or np.linalg.norm(state['com_velocity'])>.03 or not(any(contacts[:2]) and any(contacts[2:])):
                raise RuntimeError('Preparation requires upright, quiet double support')
            depth=checked_depth(json.loads(request.read_text())['depth_m'])
            request.unlink()
            cycle=PreparationCycle(time.monotonic(),depth,float(state['com_position'][2]))
            self.preparation_cycle=cycle;self.phase='crouch_preparation'
            atomic_json(self.folder/'last-preparation-request.json',{'depth_m':depth,'started':cycle.started})
            print('SLOW_PREPARATION_BEGIN_NO_JUMP',depth,flush=True)
        offset=0.
        if cycle is not None:
            if mode!=0:raise RuntimeError('Moving command during preparation')
            state=self.mm.ESTIMATOR_STATE.get();sense=self.mm.SENSE_STATE.get()
            tilt=float(np.degrees(np.arccos(np.clip(state['body_rot_matrix'][2,2],-1,1))))
            offset,done=cycle.update(time.monotonic(),float(state['com_position'][2]),tilt,sense['foot_contacts'].tolist())
            if done:
                result=cycle.result(float(state['com_position'][2]))
                atomic_json(self.folder/'preparation-result.json',result)
                print('SLOW_PREPARATION_END',json.dumps(result),flush=True)
                self.preparation_cycle=None;self.phase='holding_balance'
'''
s = s.replace(needle, needle + addition)
start = s.index(needle)
end = s.index('    def checkpoint(', start)
command = s[start:end]
assert command.count("'com_position_change_scaled':np.zeros(3)") == 1
command = command.replace("'com_position_change_scaled':np.zeros(3)",
                          "'com_position_change_scaled':np.array([0.,0.,offset])")
s = s[:start] + command + s[end:]
def methods(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)}
before,after=methods(original),methods(s)
assert before.keys()==after.keys()
assert [name for name in before if before[name]!=after[name]]==['command']
out=root/'stationary_jump_preparation_test.py'
out.write_text(s)
print('BUILT',out,'only command changed; all guards, motor worker and shutdown unchanged')
