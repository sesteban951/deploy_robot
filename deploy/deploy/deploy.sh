#!/usr/bin/env bash
set -e

SESSION="deploy"
ENV="deploy"

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

COMMANDS=(
#   "python script1.py"
#   "python script2.py"
#   "ros2 topic echo /some_topic"
#   "htop"
  ""
  ""
  ""
  ""
)

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -c "$DIR"
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