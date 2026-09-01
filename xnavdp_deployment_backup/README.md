# X-NavDP deployment backup for icra_code_0826

Backup date: 2026-09-01

This directory preserves the X-NavDP deployment and paper-evaluation code used
with the models trained in `/home/jesse/ICRA2027/icra_code_0826`.

It is an overlay backup, not a standalone simulator.  Runtime still requires:

- X-NavDP at `/home/jesse/ICRA_sota/NavDP-master/baselines/x-navdp`
- IsaacLab at `/home/jesse/IsaacLab`
- Scene-N1 assets and X-NavDP navigation metadata
- acados at `/home/jesse/ICRA_sota/acados`

## Contents

- `eval/scripts/evaluate_icra_code_0826.py`: model-specific entry point
- `eval/scripts/evaluate_diff_wheelbot.py`: deployment controller and metrics
- `eval/scripts/run_icra0826_internscene_campaign.sh`: shared Commercial/Home campaign
- `eval/scripts/run_icra0826_home_campaign.sh`: Home campaign wrapper
- `eval/scripts/summarize_icra_campaign.py`: paper metric summarizer
- `eval/environment.py`: scene loader and Isaac Sim 5.1 scene sanitization
- `eval/config_utils.py`: evaluation YAML loader
- `eval/config/eval_pointgoal/*.yaml`: Commercial and Home configurations
- `SHA256SUMS`: integrity and version record

## Restore

Copy the `eval` directory over the corresponding X-NavDP checkout, preserving
the directory structure.  Review differences before overwriting a newer
checkout.

## Commercial evaluation

```bash
cd /home/jesse/ICRA_sota/NavDP-master/baselines/x-navdp

CHECKPOINT=/absolute/path/to/checkpoint_mpc_12000.pth

bash eval/scripts/run_icra0826_internscene_campaign.sh \
  --checkpoint "$CHECKPOINT" \
  --scene-type commercial \
  --campaign-id MODEL-commercial-paper \
  --episodes-per-scene 100 \
  --seed 1234 \
  --max-v 0.5 \
  --max-omega 0.5
```

## Home evaluation

```bash
cd /home/jesse/ICRA_sota/NavDP-master/baselines/x-navdp

CHECKPOINT=/absolute/path/to/checkpoint_mpc_12000.pth

bash eval/scripts/run_icra0826_home_campaign.sh \
  --checkpoint "$CHECKPOINT" \
  --campaign-id MODEL-home-paper \
  --episodes-per-scene 100 \
  --seed 1234 \
  --max-v 0.5 \
  --max-omega 0.5
```

Re-running the identical command resumes a campaign and skips scenes whose
metric CSV already contains the requested number of episodes.
