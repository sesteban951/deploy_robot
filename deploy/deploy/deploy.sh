#!/usr/bin/env bash
#
# Base launcher: opens a 4-pane tmux session in the repo root, activates the conda env in
# every pane, and types one command per pane (without pressing Enter).
set -e

SESSION="deploy"
ENV="deploy"

# one command per pane: from the launcher that sourced this file, else empty panes
if [ -z "${COMMANDS+x}" ]; then
    COMMANDS=(
      ""
      ""
      ""
      ""
    )
fi
if [ "${#COMMANDS[@]}" -ne 4 ]; then
    echo "COMMANDS must have 4 entries (one per pane), got ${#COMMANDS[@]}" >&2
    exit 1
fi

# repo root: DEPLOY_ROOT_DIR from the active env, else from the env's config vars
# (set by `make install`), else two levels up from this script
DIR="${DEPLOY_ROOT_DIR:-}"
if [ -z "$DIR" ] && command -v conda >/dev/null 2>&1; then
    DIR="$(conda env config vars list -n "$ENV" 2>/dev/null | sed -n 's/^DEPLOY_ROOT_DIR = //p')"
fi
if [ -z "$DIR" ]; then
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [ ! -d "$DIR" ]; then
    echo "deploy root not found: '$DIR' (run 'make install' to set DEPLOY_ROOT_DIR)" >&2
    exit 1
fi

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -c "$DIR"
tmux set-option -t "$SESSION" mouse on
tmux split-window -h -t "$SESSION" -c "$DIR"
tmux split-window -v -t "$SESSION:0.0" -c "$DIR"
tmux split-window -v -t "$SESSION:0.1" -c "$DIR"
tmux select-layout -t "$SESSION" tiled

mapfile -t PANES < <(tmux list-panes -t "$SESSION" -F '#{pane_id}')

for pane in "${PANES[@]}"; do
    tmux send-keys -t "$pane" "conda activate $ENV" C-m
done

for i in "${!PANES[@]}"; do
    tmux send-keys -t "${PANES[$i]}" "${COMMANDS[$i]}"
done

tmux select-pane -t "${PANES[0]}"
tmux attach-session -t "$SESSION"
