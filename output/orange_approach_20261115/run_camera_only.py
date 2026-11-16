"""Launch RGB-D/PID observation without importing or commanding robot actuators."""
import argparse,json,math,subprocess,tempfile,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fruit-seed',nargs=2,type=float,required=True,metavar=('X','Y'))
    parser.add_argument('--reference-seed',nargs=2,type=float,required=True,metavar=('X','Y'))
    parser.add_argument('--seconds',type=float,default=20.)
    args=parser.parse_args()
    if not 1<=args.seconds<=120:parser.error('seconds must be 1..120')
    for x,y in (args.fruit_seed,args.reference_seed):
        if not all(math.isfinite(v) for v in (x,y)) or not (0<=x<640 and 0<=y<480):
            parser.error('Seeds must be finite image coordinates within 640x480')
    root=Path(__file__).resolve().parent
    ram=Path(tempfile.mkdtemp(prefix='bruce-camera-only-',dir='/dev/shm'))
    log_path=root/('camera-only-'+str(time.time_ns())+'.log')
    print('OBSERVATION_ONLY',json.dumps(dict(ram=str(ram),camera_log=str(log_path))),flush=True)
    with log_path.open('w') as log:
        camera=subprocess.Popen(['nice','-n','10','taskset','-c','1',str(root/'camera_stream'),str(ram),str(args.seconds+4)],
                                cwd=root,stdout=log,stderr=subprocess.STDOUT)
        try:
            subprocess.run(['nice','-n','10','taskset','-c','1','python3','-u',str(root/'live_range_control.py'),
                str(ram),'--camera-only','--seconds',str(args.seconds),
                '--fruit-seed',*map(str,args.fruit_seed),'--reference-seed',*map(str,args.reference_seed)],
                cwd=root,check=True,timeout=args.seconds+12)
        finally:
            if camera.poll() is None:
                camera.terminate()
                try:camera.wait(timeout=3)
                except subprocess.TimeoutExpired:camera.kill();camera.wait()
    print('CAMERA_EXIT_CODE',camera.returncode,flush=True)


if __name__=='__main__':main()
