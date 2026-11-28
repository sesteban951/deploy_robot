"""Foreground-terminal operator controls; no ROS, joystick driver or GUI."""

import os
import select
import sys
import termios
import tty

from utils.finite_state_machine import FiniteStateMachine
from utils.joystick_utils import JoystickState

KEY_HELP = 'Keyboard: d=damp, h=home, c=control, t=KCF track; Space/x=stop; q=quit (Ctrl+C also quits).'
# key -> joystick button chord; damp needs LB + RB together (utils/finite_state_machine.py)
BUTTONS = {'d': ('LB', 'RB'), 'h': ('A',), 'c': ('LMB',), 't': ('RMB',)}
STOP_KEYS = ' x\x1b\x03\x04q'


class KeyboardOperator:
    def __init__(self):
        self.fsm = FiniteStateMachine()
        self.quit_requested = False

    @property
    def state(self):
        return self.fsm.state

    def stop(self):
        self.fsm.step(JoystickState(LB=1, RB=1))

    def hold(self):
        # lost/invalid feedback: track -> control (zero twist), the walking policy keeps balancing
        self.fsm.step(JoystickState(LMB=1))

    def update(self, keys, *, feedback_ready, not_ready_reason='KCF feedback not ready'):
        keys = keys.lower()
        # A stop anywhere in a queued batch dominates every arming key.
        if any(key in STOP_KEYS for key in keys):
            self.stop()
            self.quit_requested |= 'q' in keys or '\x03' in keys or '\x04' in keys
            return
        for key in keys:
            chord = BUTTONS.get(key)
            if chord is None:
                continue
            # d/h/c never depend on KCF; only track needs complete, fresh feedback
            if key == 't' and not feedback_ready:
                print(f"t ignored: {not_ready_reason}. d/h/c and Space/x/q still work.", flush=True)
                continue
            self.fsm.step(JoystickState(**{button: 1 for button in chord}))


class TerminalKeyboard:
    """Nonblocking cbreak input, restored on normal exit and exceptions.

    None from read_keys means this process no longer owns the foreground
    terminal. EOF is an explicit disconnect, not permission to keep tracking.
    """

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.fd = self.stream.fileno()
        self.saved = None

    def __enter__(self):
        if not self.stream.isatty() or os.tcgetpgrp(self.fd) != os.getpgrp():
            raise RuntimeError('KCF keyboard control requires a foreground terminal (tmux is supported).')
        self.saved = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd, termios.TCSANOW)
            self.discard()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def discard(self):
        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except (OSError, termios.error) as error:
            raise EOFError('KCF keyboard terminal closed') from error

    def read_keys(self):
        try:
            if os.tcgetpgrp(self.fd) != os.getpgrp():
                return None
            if not select.select([self.fd], [], [], 0)[0]:
                return ''
            data = os.read(self.fd, 256)
        except (OSError, termios.error) as error:
            raise EOFError('KCF keyboard terminal closed') from error
        if not data:
            raise EOFError('KCF keyboard terminal closed')
        # Any control/escape sequence is treated as a stop, never an arm.
        return data.decode('ascii', errors='ignore')

    def __exit__(self, *_):
        if self.saved is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
            except (OSError, termios.error):
                pass  # A closed terminal still must not skip controller cleanup.
            self.saved = None
