"""Runtime housekeeping; independent of the MPC model and optimization."""

import gc
from threading import Lock


class StartupGCFreeze:
    """Keep the startup heap out of repeated cyclic-GC scans.

    Use after loading/compiling in the dedicated controller process. New
    objects remain collectable and reference counting is unchanged. The last
    owner restores normal scanning on close. Never unfreeze somebody else's
    pre-existing permanent generation or enable GC if the caller disabled it.
    """

    _lock = Lock()
    _owners = 0

    def __init__(self, enabled: bool):
        self.active = False
        if not enabled:
            return
        with self._lock:
            if not gc.isenabled():
                return
            if type(self)._owners == 0:
                if gc.get_freeze_count():
                    return
                # This one full scan happens before the controller connects,
                # never in solve()/get(). Remove startup garbage first.
                gc.collect()
                gc.freeze()
            type(self)._owners += 1
            self.active = True

    def close(self):
        with self._lock:
            if not self.active:
                return
            self.active = False
            type(self)._owners -= 1
            if type(self)._owners == 0:
                gc.unfreeze()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
