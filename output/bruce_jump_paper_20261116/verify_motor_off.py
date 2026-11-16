"""Read torque-enable/status only. Refuse serial access if a driver owns it."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-root", type=Path, default=Path("/home/khadas/BRUCE/BRUCE-OP"))
    a = p.parse_args()
    sys.path.insert(0, str(a.robot_root))
    from Settings.BRUCE_macros import BEAR_port, BEAR_baudrate
    owner = subprocess.run(["fuser", BEAR_port], capture_output=True)
    if owner.returncode == 0:
        raise RuntimeError("Serial port owned by another process; no access")
    if not Path(BEAR_port).exists():
        raise RuntimeError("BEAR port not present")
    from Library.ACTUATORS.BEAR_controller import TimedBEAR
    b = TimedBEAR(port=BEAR_port, baudrate=BEAR_baudrate, timeout=.01, bulk_timeout=.02)
    results = []
    try:
        for i in range(1, 11):
            result = b.get_torque_enable(i)
            results.append({"id": i, "readback": result,
                            "confirmed_off_normal": result == [(0, 128)]})
    finally:
        b.ser.close()
    report = {"time_utc": datetime.now(timezone.utc).isoformat(),
              "serial_writes": "read-register requests only, no parameter/enable writes",
              "all_off_normal": all(r["confirmed_off_normal"] for r in results),
              "motors": results}
    Path(__file__).with_name("motor-off-readback.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not report["all_off_normal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
