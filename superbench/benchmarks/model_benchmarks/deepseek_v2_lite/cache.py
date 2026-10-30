# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded removal of one V2-Lite run's downloaded model cache."""

import subprocess
import sys
import time
from pathlib import Path

from superbench.common.utils import logger

FAILURE_CLEANUP_GRACE_SECONDS = 60
_REMOVE_TREE = 'import os, shutil, sys\nif os.path.exists(sys.argv[1]):\n    shutil.rmtree(sys.argv[1])'


def remove_model_cache(cache: Path, deadline: float) -> None:
    """Delete one run's model cache before a monotonic deadline, keeping reports and logs.

    A missing budget, an expired deletion or a SuperBench SIGTERM raises TimeoutError. Other
    filesystem errors are logged so that a validated measurement is not discarded.
    """
    start = time.monotonic()
    remaining = deadline - start
    if remaining <= 0:
        raise TimeoutError('No time remains to remove the V2-Lite model cache at {}.'.format(cache))
    try:
        # A child process can be stopped and reaped at the deadline; an in-process rmtree cannot.
        subprocess.run(
            [sys.executable, '-c', _REMOVE_TREE, str(cache)],
            timeout=remaining,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError('V2-Lite model cache cleanup exceeded the deadline at {}.'.format(cache)) from error
    except TimeoutError:
        # The SuperBench SIGTERM handler raises TimeoutError, an OSError subclass; keep it fatal.
        raise
    except subprocess.CalledProcessError as error:
        logger.error('Failed to remove the V2-Lite model cache %s: %s', cache, error.output or error)
        return
    except OSError as error:
        logger.error('Failed to remove the V2-Lite model cache %s: %s', cache, error)
        return
    logger.info('V2-Lite model cache cleanup completed for %s in %.1f seconds.', cache, time.monotonic() - start)


def remove_model_cache_after_failure(cache: Path, deadline: float) -> None:
    """Remove the cache of a failed run without replacing the run's original error.

    Cleanup may run until the job deadline or for FAILURE_CLEANUP_GRACE_SECONDS, whichever ends later,
    so a timed-out worker does not leave its checkpoint behind only because the budget is spent.
    """
    try:
        remove_model_cache(cache, max(deadline, time.monotonic() + FAILURE_CLEANUP_GRACE_SECONDS))
    except TimeoutError as error:
        logger.error('V2-Lite model cache cleanup did not finish after a failed run: %s', error)
