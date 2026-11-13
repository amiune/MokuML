"""Interrupt local source reads when a run is cancelled or released."""

import threading
from contextlib import contextmanager


@contextmanager
def interruptible(cancel_event, interrupt):
    finished = threading.Event()

    def watch():
        while not finished.wait(0.05):
            if cancel_event is not None and cancel_event.is_set():
                try:
                    interrupt()
                except Exception:
                    pass
                return

    thread = (
        threading.Thread(target=watch, daemon=True)
        if cancel_event is not None
        else None
    )
    if thread:
        thread.start()
    try:
        yield
    finally:
        finished.set()
        if thread:
            thread.join(timeout=0.2)
