#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_ROOT"
SIM_PYTHON="${BRUCE_SIM_PYTHON:-/home/user/miniconda3/envs/nudt/bin/python}"
DEMO_OUTPUT="${BRUCE_DEMO_OUTPUT:-output/bruce_physics_recheck}"
VALIDATION_OUTPUT="${BRUCE_VALIDATION_OUTPUT:-output/bruce_physics_check}"
case "${1:-help}" in
  play)
    xdg-open "$PROJECT_ROOT/BRUCE物理复核_20261114_站立.mp4"
    ;;
  compare)
    xdg-open "$PROJECT_ROOT/BRUCE仿真对比.mp4"
    ;;
  validate|trial08|support)
    TRIAL=stand
    [[ "$1" == trial08 ]] && TRIAL=box08
    [[ "$1" == support ]] && TRIAL=support
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    "$SIM_PYTHON" scripts/data/bruce/validate_physics.py \
      --trial "$TRIAL" --out "$VALIDATION_OUTPUT"
    ;;
  audit)
    mkdir -p "$VALIDATION_OUTPUT"
    "$SIM_PYTHON" scripts/data/bruce/audit_reference.py \
      --json "$VALIDATION_OUTPUT/reference_audit.json"
    ;;
  crossings)
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    "$SIM_PYTHON" scripts/data/bruce/probe_dynamic_step.py \
      --out "${BRUCE_CROSSING_OUTPUT:-output/bruce_crossing_next}" \
      --heights .01 .02 .03 .04 .06 .08 --swing-time .22 \
      --second-time .60 --second-profile arch --body-height .36 \
      --center-x .17 --thickness .03 --width .5 --friction .8 \
      --capture-mode recapture --momentum-weight 1000 --record
    ;;
  assisted|joint)
    if [[ "$1" == assisted ]]; then
      CONTROL_MODE=assisted
      DATASET_NAME=bruce_assisted_demo
    else
      CONTROL_MODE=joint_only
      DATASET_NAME=bruce_joint_demo
    fi
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    "$SIM_PYTHON" scripts/data/bruce/collect_bruce.py \
      --config scripts/data/bruce/config/bruce_stepover.yaml \
      --control-mode "$CONTROL_MODE" --episodes 4 --out "$DATASET_NAME" \
      --cache-dir "$DEMO_OUTPUT/$1" \
      --video-dir "$DEMO_OUTPUT/$1/videos" --video-episodes 4
    ;;
  check)
    for DATASET_NAME in bruce_assisted_reference_50 bruce_joint_trials_40; do
      "$SIM_PYTHON" scripts/data/bruce/check_dataset.py \
        --dataset "$DATASET_NAME" --cache-dir output/bruce_sim_20261113
    done
    ;;
  help|-h|--help)
    printf '%s\n' '用法: bash BRUCE仿真命令.sh {play|compare|validate|trial08|support|audit|crossings|assisted|joint|check}' \
      'play: 播放站立验证；compare: 历史对比；validate: 重跑站立；trial08: 8cm实验（未通过时退出码2）。' \
      'assisted/joint: 调试采集；check: 结构检查及自主成功资格复核。' \
      'support: 重心转移诊断（未就绪时退出码2）；audit: 原8cm参考的支撑范围筛查。' \
      'crossings: 六种高度的越障实验采集，保存RGB/动作/2kHz物理记录；没有合格成功时退出码2。' \
      'assisted 为基座辅助参考；joint 为自由基座尝试，当前策略会失衡。'
    ;;
  *)
    printf '未知命令: %s\n' "$1" >&2
    exit 2
    ;;
esac
