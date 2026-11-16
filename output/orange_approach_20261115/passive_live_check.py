"""Motors-OFF read-only bus check alongside RGB-D and vision processing.

Only PICO nominal/idle sensor commands are sent; BEAR access is reads only.
Refuses to run if either serial port is owned or any motor is enabled.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/khadas/BRUCE/BRUCE-OP')
sys.path.insert(0,str(REPO))


def sensor_probe(output,seconds):
    from Library.BRUCE_SENSE.Manager import SenseManager
    from Settings.BRUCE_macros import PICO_port,PICO_baudrate
    sensor=SenseManager(port=PICO_port,baudrate=PICO_baudrate)
    rows=[];started=time.monotonic();last=None
    try:
        sensor.send_data(pico_mode='nominal')
        while time.monotonic()-started<seconds:
            if sensor.read_data():
                now=time.monotonic()
                rows.append(dict(monotonic=now,gap_ms=None if last is None else (now-last)*1000,
                                 accel=sensor.accel.tolist(),contacts=sensor.foot_contact.tolist()))
                last=now
            time.sleep(.001)
    finally:
        sensor.send_data(pico_mode='idle');sensor.close_port()
        gaps=[row['gap_ms'] for row in rows if row['gap_ms'] is not None]
        result=dict(frames=len(rows),gap_ms_p50_p99_max=np.percentile(gaps,[50,99,100]).tolist() if gaps else None,
                    gaps_over_150ms=sum(v>150 for v in gaps),records=rows)
        (output/'sensor-check.json').write_text(json.dumps(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensor-child',type=Path)
    args=parser.parse_args()
    if args.sensor_child:
        sensor_probe(args.sensor_child,32);return
    from Library.ACTUATORS.BEAR_controller import TimedBEAR
    from Settings.BRUCE_macros import BEAR_port,BEAR_baudrate,PICO_port
    import Startups.memory_manager as mm
    assert mm.LEG_COMMAND.get()['BEAR_enable'][0]==0,'Software motor enable is not zero'
    for port in (BEAR_port,PICO_port):
        assert subprocess.run(['fuser',port],capture_output=True).returncode==1,('Port owned or cannot check',port)
    output=ROOT/('passive-live-'+time.strftime('%H%M%S'));output.mkdir()
    ram=Path('/dev/shm')/('bruce-passive-live-'+str(os.getpid()));ram.mkdir()
    bus=TimedBEAR(port=BEAR_port,baudrate=BEAR_baudrate,timeout=.005,bulk_timeout=.010)
    children=[];logs=[];rows=[];before=None;after=None
    def launch(name,argv):
        log=(output/(name+'.log')).open('w');logs.append(log)
        child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        children.append(child);return child
    try:
        before=[bus.get_torque_enable(i)[0][0] for i in range(1,11)]
        assert before==[0]*10,('Motor is not disabled',before)
        sense=launch('sensor',['taskset','-c','2','python3','-u',__file__,'--sensor-child',str(output)])
        started=time.monotonic();camera=None;vision=None;camera_end=None
        while time.monotonic()-started<32:
            tick=time.monotonic();elapsed=tick-started
            if camera is None and elapsed>=5:
                camera=launch('camera',['nice','-n','10','taskset','-c','1',str(ROOT/'camera_stream'),str(ram),'22'])
                vision=launch('vision',['nice','-n','10','taskset','-c','1','python3','-u',str(ROOT/'live_range_control.py'),
                              str(ram),'--camera-only','--seconds','20','--fruit-seed','505','171','--reference-seed','470','162'])
            if camera is not None and camera.poll() is not None and camera_end is None:
                camera_end=tick
            phase='baseline' if camera is None else 'camera_running' if camera_end is None else 'after_camera'
            begin=time.monotonic()
            result=bus.bulk_read(list(range(1,11)),['present_position','present_velocity','present_iq','input_voltage'])
            duration=(time.monotonic()-begin)*1000
            okay=(result is not None and len(result)==10 and all(x is not None and x[0] is not None for x in result))
            rows.append(dict(monotonic=tick,elapsed=elapsed,phase=phase,duration_ms=duration,okay=okay,
                             codes=[x[1] for x in result] if okay else None,
                             voltage=[x[0][3] for x in result] if okay else None))
            assert mm.LEG_COMMAND.get()['BEAR_enable'][0]==0,'Software motor enable changed'
            time.sleep(max(0.,.005-(time.monotonic()-tick)))
        after=[bus.get_torque_enable(i)[0][0] for i in range(1,11)]
        assert after==[0]*10,('Motor mode changed',after)
        for child in children:
            child.wait(timeout=5)
        summary=dict(torque_enable_before=before,torque_enable_after=after,
                     motors_ever_enabled_by_test=False,camera_ram=str(ram),
                     child_exit_codes=[c.returncode for c in children])
        for phase in ('baseline','camera_running','after_camera'):
            group=[r for r in rows if r['phase']==phase]
            summary[phase]=dict(reads=len(group),failed=sum(not r['okay'] for r in group),
                duration_ms_p50_p99_max=np.percentile([r['duration_ms'] for r in group],[50,99,100]).tolist() if group else None)
        sensor=json.loads((output/'sensor-check.json').read_text())
        summary['sensor']={k:v for k,v in sensor.items() if k!='records'}
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print('PASSIVE_RESULT',output,json.dumps(summary),flush=True)
    finally:
        bus.ser.close()
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()
        (output/'motor-reads.json').write_text(json.dumps(rows))


if __name__=='__main__':
    main()
