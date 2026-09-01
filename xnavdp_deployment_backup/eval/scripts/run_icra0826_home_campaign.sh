#!/usr/bin/env bash
# Run the Diff-Wheelbot paper campaign on the official InternScenes Home split.
#
# This is intentionally a thin wrapper around the shared InternScene campaign
# runner so Home and Commercial use exactly the same controller, collision
# metric, smoothness metric, resume behavior, and summary format.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for arg in "$@"; do
    if [[ "$arg" == "--scene-type" ]]; then
        echo "run_icra0826_home_campaign.sh fixes --scene-type to home; remove the explicit --scene-type argument." >&2
        exit 2
    fi
done

exec "${SCRIPT_DIR}/run_icra0826_internscene_campaign.sh" \
    --scene-type home \
    "$@"
