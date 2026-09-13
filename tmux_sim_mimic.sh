#!/usr/bin/env bash
# Open a 4-pane tmux session with the mimic sim commands typed but not run.
SESSION=${SESSION:-sim_mimic}
CONFIG=${1:-g1_29dof_mimic.yaml}
ROOT=$(cd "$(dirname "$0")" && pwd)

tmux has-session -t "$SESSION" 2>/dev/null && exec tmux attach -t "$SESSION"

tmux new-session -d -s "$SESSION" -c "$ROOT"
tmux split-window -h -t "$SESSION":0 -c "$ROOT"
tmux split-window -v -t "$SESSION":0.0 -c "$ROOT"
tmux split-window -v -t "$SESSION":0.2 -c "$ROOT"
tmux select-layout -t "$SESSION":0 tiled

CMDS=(
  "python deploy/joystick/joystick_pygame.py"
  "python deploy/simulation/control_29dof_mimic.py --config $CONFIG"
  "python deploy/simulation/simulation.py --config $CONFIG --force --delay 0"
  "python deploy/logger/log.py --mode sim"
)

for i in "${!CMDS[@]}"; do
  tmux send-keys -t "$SESSION":0.$i "conda activate deploy" Enter
  tmux send-keys -t "$SESSION":0.$i "${CMDS[$i]}"
done

tmux select-pane -t "$SESSION":0.0
exec tmux attach -t "$SESSION"
