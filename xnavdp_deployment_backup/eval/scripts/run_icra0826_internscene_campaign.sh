#!/usr/bin/env bash
# Run a resumable Diff-Wheelbot paper-evaluation campaign over InternScene.
#
# Commercial example:
#   bash eval/scripts/run_icra0826_internscene_campaign.sh \
#     --checkpoint /absolute/path/checkpoint_mpc_12000.pth \
#     --scene-type commercial \
#     --campaign-id gate-retention-v1-12k-commercial-matched \
#     --max-v 0.5 --max-omega 0.5

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-/home/jesse/IsaacLab/isaaclab.sh}"

CHECKPOINT=""
SCENE_TYPE="commercial"
CAMPAIGN_ID=""
EPISODES_PER_SCENE=100
SEED=1234
MAX_V=0.5
MAX_OMEGA=0.5
COLLISION_HORIZONTAL_THRESHOLD=6.5
COLLISION_CONFIRM_STEPS=2
COLLISION_PEAK_THRESHOLD=13.0
METRIC_PROTOCOL_VERSION="horizontal-contact-v1_linear-accel-rms-v1"
START_SCENE=0
END_SCENE=-1
RECORD_VIDEO=0
RERUN_COMPLETED=0
STOP_ON_ERROR=0
DRY_RUN=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Run a resumable Diff-Wheelbot paper campaign over InternScene.

Options:
  --checkpoint PATH          Required checkpoint path.
  --scene-type TYPE          commercial or home (default: commercial).
  --campaign-id ID           Stable output folder name; generated if omitted.
  --episodes-per-scene N     Default: 100 (released metadata has 100).
  --seed N                   Default: 1234.
  --max-v V                  Execution linear-speed cap (default: 0.5).
  --max-omega W              Execution angular-speed cap (default: 0.5).
  --collision-threshold N    Horizontal contact force in N (default: 6.5).
  --collision-confirm N      Consecutive crossings required (default: 2).
  --collision-peak N         Strong single-frame horizontal force (default: 13.0).
  --start-scene N            First eval-split index (default: 0).
  --end-scene N              Last eval-split index (default: all).
  --record-video             Record all episodes; not recommended for metrics.
  --rerun-completed          Re-run scenes already containing all metrics.
  --stop-on-error            Stop instead of continuing after a failed scene.
  --dry-run                  Validate and print commands without launching Isaac.
  -- ARGS...                 Extra evaluator arguments; explicit values win.
EOF
}

while (($#)); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --scene-type) SCENE_TYPE="$2"; shift 2 ;;
        --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
        --episodes-per-scene) EPISODES_PER_SCENE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-v) MAX_V="$2"; shift 2 ;;
        --max-omega) MAX_OMEGA="$2"; shift 2 ;;
        --collision-threshold) COLLISION_HORIZONTAL_THRESHOLD="$2"; shift 2 ;;
        --collision-confirm) COLLISION_CONFIRM_STEPS="$2"; shift 2 ;;
        --collision-peak) COLLISION_PEAK_THRESHOLD="$2"; shift 2 ;;
        --start-scene) START_SCENE="$2"; shift 2 ;;
        --end-scene) END_SCENE="$2"; shift 2 ;;
        --record-video) RECORD_VIDEO=1; shift ;;
        --rerun-completed) RERUN_COMPLETED=1; shift ;;
        --stop-on-error) STOP_ON_ERROR=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint does not exist: ${CHECKPOINT:-<missing>}" >&2
    exit 2
fi
if [[ "$SCENE_TYPE" != "commercial" && "$SCENE_TYPE" != "home" ]]; then
    echo "--scene-type must be commercial or home" >&2
    exit 2
fi
if ((EPISODES_PER_SCENE < 1 || START_SCENE < 0)); then
    echo "Episode and scene counts must be positive" >&2
    exit 2
fi
python3 -c '
import sys
threshold, peak = map(float, sys.argv[1:3])
confirm = int(sys.argv[3])
if threshold <= 0 or peak < threshold or confirm < 1:
    raise SystemExit("Invalid collision thresholds/confirmation count")
' "$COLLISION_HORIZONTAL_THRESHOLD" "$COLLISION_PEAK_THRESHOLD" "$COLLISION_CONFIRM_STEPS"
if [[ ! -x "$ISAACLAB_SH" ]]; then
    echo "IsaacLab launcher is not executable: $ISAACLAB_SH" >&2
    exit 2
fi

CONFIG_FILE="eval/config/eval_pointgoal/wheeled_internscene_${SCENE_TYPE}.yaml"
SPLIT_FILE="${REPO_ROOT}/data/scenes/scene_split.json"
SPLIT_KEY="${SCENE_TYPE}_eval"
DATASET_NAME="internscenes_${SCENE_TYPE}"
SCENE_ROOT="${REPO_ROOT}/data/scenes/${DATASET_NAME}/scenes_${SCENE_TYPE}"
METADATA_ROOT="${REPO_ROOT}/data/scenes/navigation_metadata/${DATASET_NAME}"

mapfile -t SCENES < <(
    python3 -c \
        'import json,sys; print("\n".join(json.load(open(sys.argv[1]))[sys.argv[2]]))' \
        "$SPLIT_FILE" "$SPLIT_KEY"
)
SCENE_COUNT=${#SCENES[@]}
if ((END_SCENE < 0)); then END_SCENE=$((SCENE_COUNT - 1)); fi
if ((START_SCENE >= SCENE_COUNT || END_SCENE >= SCENE_COUNT || START_SCENE > END_SCENE)); then
    echo "Scene range ${START_SCENE}..${END_SCENE} is invalid for ${SCENE_COUNT} scenes" >&2
    exit 2
fi
SELECTED_SCENE_COUNT=$((END_SCENE - START_SCENE + 1))
if ((EPISODES_PER_SCENE > 100)); then
    echo "Released InternScene metadata contains only 100 episodes per scene" >&2
    exit 2
fi

for ((INDEX=START_SCENE; INDEX<=END_SCENE; INDEX++)); do
    SCENE_NAME="${SCENES[$INDEX]}"
    SCENE_DIR="${SCENE_ROOT}/${SCENE_NAME}"
    POINT_DIR="${METADATA_ROOT}/pointgoal_start_pair/${SCENE_NAME}"
    ESDF_FILE="${METADATA_ROOT}/esdf/${SCENE_NAME}/navigable.ply"
    if [[ ! -d "$SCENE_DIR" || ! -d "$POINT_DIR" || ! -f "$ESDF_FILE" ]]; then
        echo "Missing assets for scene index ${INDEX}: ${SCENE_NAME}" >&2
        exit 2
    fi
    if ! find "$POINT_DIR" -maxdepth 1 -type f -name '*.npy' -print -quit | grep -q .; then
        echo "Missing point-goal samples: $POINT_DIR" >&2
        exit 2
    fi
done

if [[ -z "$CAMPAIGN_ID" ]]; then
    RUN_NAME="$(basename "$(dirname "$CHECKPOINT")")"
    CKPT_NAME="$(basename "$CHECKPOINT" .pth)"
    CAMPAIGN_ID="icra0826-${RUN_NAME}-${CKPT_NAME}-${SCENE_TYPE}-v${MAX_V}-w${MAX_OMEGA}-seed${SEED}"
fi
CAMPAIGN_ID="${CAMPAIGN_ID//[^A-Za-z0-9_.-]/-}"
if ((DRY_RUN)); then
    CAMPAIGN_ROOT="/tmp/xnavdp-campaign-dry-run/${CAMPAIGN_ID}"
else
    CAMPAIGN_ROOT="${REPO_ROOT}/outputs/evaluation/wheeled_internscene_${SCENE_TYPE}/${CAMPAIGN_ID}"
fi
LOG_ROOT="${CAMPAIGN_ROOT}/_campaign_logs"
mkdir -p "$LOG_ROOT"

export TERM=xterm
export PYTHONPATH="$REPO_ROOT"
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-/home/jesse/ICRA_sota/acados}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/xnavdp_mpl}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/xnavdp_pycache}"

VIDEO_ARG="--no_record_video"
if ((RECORD_VIDEO)); then VIDEO_ARG="--record_video"; fi

# The full-system evaluator has separate angular limits for pre-alignment,
# terminal docking, and stuck recovery. Keep every auxiliary controller inside
# the campaign-wide max_omega envelope, especially for the matched 0.5 rad/s
# protocol. Preserve the original lower default when max_omega is more lenient.
RECOVERY_TURN_OMEGA="$(python3 -c 'import sys; print(min(0.70, float(sys.argv[1])))' "$MAX_OMEGA")"
PREALIGN_MAX_OMEGA="$(python3 -c 'import sys; print(min(1.00, float(sys.argv[1])))' "$MAX_OMEGA")"
PREALIGN_SEARCH_OMEGA="$(python3 -c 'import sys; print(min(0.70, float(sys.argv[1])))' "$MAX_OMEGA")"
TERMINAL_MAX_OMEGA="$(python3 -c 'import sys; print(min(0.80, float(sys.argv[1])))' "$MAX_OMEGA")"

MANIFEST="${CAMPAIGN_ROOT}/campaign_manifest.txt"
CHECKPOINT_REAL="$(readlink -f "$CHECKPOINT")"
CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
PROTOCOL_SIGNATURE="$(
    printf '%s\n' \
        "$CHECKPOINT_REAL" "$CHECKPOINT_SHA256" "$SCENE_TYPE" \
        "$EPISODES_PER_SCENE" "$SEED" "$MAX_V" "$MAX_OMEGA" \
        "$COLLISION_HORIZONTAL_THRESHOLD" "$COLLISION_CONFIRM_STEPS" \
        "$COLLISION_PEAK_THRESHOLD" "$METRIC_PROTOCOL_VERSION" \
        "$RECORD_VIDEO" "${EXTRA_ARGS[*]:-}" | sha256sum | awk '{print $1}'
)"
if [[ -f "$MANIFEST" ]]; then
    EXISTING_SIGNATURE="$(awk -F= '$1 == "protocol_signature" {print $2}' "$MANIFEST")"
    if [[ -n "$EXISTING_SIGNATURE" && "$EXISTING_SIGNATURE" != "$PROTOCOL_SIGNATURE" ]]; then
        echo "Campaign ID already exists with a different protocol: $CAMPAIGN_ID" >&2
        echo "Choose a new --campaign-id instead of mixing paper results." >&2
        exit 2
    fi
fi
{
    echo "campaign_id=${CAMPAIGN_ID}"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "protocol_signature=${PROTOCOL_SIGNATURE}"
    echo "checkpoint=${CHECKPOINT_REAL}"
    echo "checkpoint_sha256=${CHECKPOINT_SHA256}"
    echo "scene_type=${SCENE_TYPE}"
    echo "config_file=${CONFIG_FILE}"
    echo "scene_count=${SCENE_COUNT}"
    echo "selected_scene_count=${SELECTED_SCENE_COUNT}"
    echo "episodes_per_scene=${EPISODES_PER_SCENE}"
    echo "seed=${SEED}"
    echo "max_v=${MAX_V}"
    echo "max_omega=${MAX_OMEGA}"
    echo "collision_horizontal_threshold=${COLLISION_HORIZONTAL_THRESHOLD}"
    echo "collision_confirm_steps=${COLLISION_CONFIRM_STEPS}"
    echo "collision_peak_threshold=${COLLISION_PEAK_THRESHOLD}"
    echo "metric_protocol_version=${METRIC_PROTOCOL_VERSION}"
    echo "recovery_turn_omega=${RECOVERY_TURN_OMEGA}"
    echo "prealign_max_omega=${PREALIGN_MAX_OMEGA}"
    echo "prealign_search_omega=${PREALIGN_SEARCH_OMEGA}"
    echo "terminal_max_omega=${TERMINAL_MAX_OMEGA}"
    echo "record_video=${RECORD_VIDEO}"
    echo "timeout_seconds_per_episode=122"
    echo "extra_args=${EXTRA_ARGS[*]:-}"
} > "$MANIFEST"

metric_rows() {
    local metric_file="$1"
    if [[ ! -f "$metric_file" ]]; then echo 0; return; fi
    python3 -c \
        'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline=""))))' \
        "$metric_file"
}

echo "Campaign: $CAMPAIGN_ID"
echo "Scenes: ${START_SCENE}..${END_SCENE} of $SCENE_COUNT"
echo "Episodes per scene: $EPISODES_PER_SCENE"
echo "Protocol: max_v=$MAX_V, max_omega=$MAX_OMEGA, seed=$SEED"
echo "Collision: horizontal>${COLLISION_HORIZONTAL_THRESHOLD}N for ${COLLISION_CONFIRM_STEPS} steps, peak>=${COLLISION_PEAK_THRESHOLD}N"
echo "Smoothness: linear acceleration RMS (m/s^2, lower is smoother)"
echo "Videos: $RECORD_VIDEO"
echo "Output: $CAMPAIGN_ROOT"

FAILURES=0
cd "$REPO_ROOT"
for ((INDEX=START_SCENE; INDEX<=END_SCENE; INDEX++)); do
    SCENE_NAME="${SCENES[$INDEX]}"
    METRIC_FILE="${CAMPAIGN_ROOT}/${DATASET_NAME}/${SCENE_NAME}/metric.csv"
    ROWS="$(metric_rows "$METRIC_FILE")"
    if ((RERUN_COMPLETED == 0 && ROWS >= EPISODES_PER_SCENE)); then
        echo "[skip] scene=${INDEX}/${SCENE_COUNT} ${SCENE_NAME}: ${ROWS} episodes complete"
        continue
    fi

    LOG_FILE="${LOG_ROOT}/scene_$(printf '%02d' "$INDEX")_${SCENE_NAME}.log"
    CMD=(
        "$ISAACLAB_SH" -p
        eval/scripts/evaluate_icra_code_0826.py
        --config_file "$CONFIG_FILE"
        --checkpoint "$CHECKPOINT"
        --campaign_id "$CAMPAIGN_ID"
        --scene_index "$INDEX"
        --num_episodes "$EPISODES_PER_SCENE"
        --seed "$SEED"
        --max_v "$MAX_V"
        --max_omega "$MAX_OMEGA"
        --contact_force_threshold "$COLLISION_HORIZONTAL_THRESHOLD"
        --collision_force_consecutive_steps "$COLLISION_CONFIRM_STEPS"
        --collision_peak_force_threshold "$COLLISION_PEAK_THRESHOLD"
        --recovery_turn_omega "$RECOVERY_TURN_OMEGA"
        --prealign_max_omega "$PREALIGN_MAX_OMEGA"
        --prealign_search_omega "$PREALIGN_SEARCH_OMEGA"
        --terminal_max_omega "$TERMINAL_MAX_OMEGA"
        "$VIDEO_ARG"
    )
    CMD+=("${EXTRA_ARGS[@]}")
    printf '[run] scene=%d/%d %s\n' "$INDEX" "$SCENE_COUNT" "$SCENE_NAME"
    printf '      command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    if ((DRY_RUN)); then continue; fi

    set +e
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
    set -e
    ROWS="$(metric_rows "$METRIC_FILE")"
    if ((STATUS != 0 || ROWS < EPISODES_PER_SCENE)); then
        echo "[failed] scene=${INDEX} status=${STATUS} metric_rows=${ROWS}/${EPISODES_PER_SCENE}" >&2
        FAILURES=$((FAILURES + 1))
        if ((STOP_ON_ERROR)); then break; fi
    else
        echo "[done] scene=${INDEX} metric_rows=${ROWS}/${EPISODES_PER_SCENE}"
    fi
done

if ((DRY_RUN)); then
    echo "Dry run complete; Isaac Sim was not launched."
    exit 0
fi

python3 eval/scripts/summarize_icra_campaign.py \
    "$CAMPAIGN_ROOT" \
    --expected-scenes "$SELECTED_SCENE_COUNT" \
    --expected-episodes "$EPISODES_PER_SCENE"

if ((FAILURES)); then
    echo "Campaign finished with ${FAILURES} failed/incomplete scenes. Re-run the same command to resume." >&2
    exit 1
fi
echo "Campaign finished successfully."
