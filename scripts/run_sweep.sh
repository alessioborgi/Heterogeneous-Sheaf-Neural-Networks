#!/usr/bin/env bash
# Create a Weights & Biases sweep from a YAML file under sweeps/ and start an agent.
#
# Usage:
#   scripts/run_sweep.sh sweeps/nc/dblp/diag_sheaf.yaml            # create sweep + run agent
#   SWEEP_ID=abc123 scripts/run_sweep.sh                            # attach an agent to an existing sweep
#   WANDB_ENTITY=my-team WANDB_PROJECT=HetSheaf scripts/run_sweep.sh sweeps/gc/mutag/general_poly_sheaf.yaml
#
# Environment variables (all optional):
#   WANDB_PROJECT  W&B project name           (default: HetSheaf)
#   WANDB_ENTITY   W&B user or team           (default: the logged-in user)
#   SWEEP_ID       existing sweep id          (default: create a new sweep from the YAML)
#   COUNT          max runs for this agent    (default: unlimited)
set -euo pipefail

SWEEP_CONFIG="${1:-}"
WANDB_PROJECT="${WANDB_PROJECT:-HetSheaf}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SWEEP_ID="${SWEEP_ID:-}"
COUNT="${COUNT:-}"

cd "$(dirname "$0")/.."

if [ -z "$SWEEP_ID" ]; then
  if [ -z "$SWEEP_CONFIG" ]; then
    echo "Usage: $0 <sweeps/<task>/<dataset>/<model>.yaml>   (or set SWEEP_ID)" >&2
    exit 1
  fi
  echo "Creating sweep from $SWEEP_CONFIG in project $WANDB_PROJECT ..."
  SWEEP_ID="$(uv run wandb sweep ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} --project "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1 \
    | tee /dev/stderr | awk -F': ' '/Creating sweep with ID:/ {print $NF}' | tail -n1)"
fi

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: could not determine the sweep id." >&2
  exit 1
fi

if [ -z "$WANDB_ENTITY" ]; then
  WANDB_ENTITY="$(uv run python -c 'import wandb; print(wandb.Api().default_entity)')"
fi

echo "Starting agent for $WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
exec uv run wandb agent ${COUNT:+--count "$COUNT"} "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
