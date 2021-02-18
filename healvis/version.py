# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

from . import __version__
import inspect
from pathlib import Path


def history_string(notes=""):
    """Creates a standardized history string that all functions that write to disk can use. Optionally add notes."""

    # [3] is the name of the function that called this function
    # inspect.stack()[1][1] is path to the file that contains the function that called this function
    stack = inspect.stack()[1]
    call_fnc = stack[3]
    call_mod = Path(stack[1]).name
    notes = f"\nNotes:\n{notes}" if notes else ""
    return f"""
    ------------
    This file was produced by the function {call_fnc}() in {call_mod} using healvis {__version__}.
    {notes}
    ------------
    """
