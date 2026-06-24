# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for deterministic model training and validation."""


def record_step_loss(loss, curr_step, losses_list, logger=None):
    """Record per-step loss value for determinism tracking.

    Args:
        loss: Loss tensor or float value.
        curr_step (int): Current training step.
        losses_list (list): List to append loss values to.
        logger: Optional logger for warnings.

    Returns:
        float: Converted loss value, or None if conversion failed.
    """
    try:
        v = float(loss.detach().item()) if hasattr(loss, 'detach') else float(loss)
        losses_list.append(v)
        return v
    except Exception:
        if logger:
            logger.info(f'Unable to convert loss to float at step {curr_step}')
        losses_list.append(None)
        return None


def _record_loss_fingerprint(curr_step, loss_value, periodic_dict, logger):
    """Record loss fingerprint at current step."""
    try:
        if 'loss' in periodic_dict and isinstance(periodic_dict['loss'], list):
            periodic_dict['loss'].append(loss_value if loss_value is not None else None)
        else:
            periodic_dict['loss'] = [loss_value if loss_value is not None else None]

        if logger:
            logger.info(f'Loss at step {curr_step}: {loss_value}')
        periodic_dict.setdefault('step', []).append(curr_step)
    except Exception:
        if logger:
            logger.warning(f'Unable to log loss at curr_step {curr_step}')


def _record_activation_fingerprint(curr_step, logits, periodic_dict, logger):
    """Record activation mean fingerprint at current step."""
    try:
        if logits is not None:
            act_mean = (
                float(logits[0].detach().float().mean().item()) if hasattr(logits[0], 'detach') else float(logits[0])
            )
            if logger:
                logger.info(f'ActMean at step {curr_step}: {act_mean}')
            periodic_dict.setdefault('act_mean', []).append(act_mean)
        else:
            periodic_dict.setdefault('act_mean', []).append(None)
    except Exception:
        if logger:
            logger.warning(f'Unable to log act_mean at curr_step {curr_step}')
        periodic_dict.setdefault('act_mean', []).append(None)


def _record_activation_chunks(curr_step, logits, periodic_dict, logger, num_chunks):
    """Record per-chunk checksums of the activation (Approach A: granular fingerprint).

    The activation is flattened and split into ``num_chunks`` segments, and the sum of
    each segment is recorded separately. Unlike a single mean over the whole tensor, a
    corruption that lands inside the normal value range still changes its own chunk's
    checksum, and errors in different chunks cannot cancel each other out.

    Args:
        curr_step (int): Current training step.
        logits: Logits tensor for the activation fingerprint.
        periodic_dict (dict): Dictionary to store periodic data; appends to 'act_chunks'.
        logger: Optional logger for warnings.
        num_chunks (int): Number of segments to split the activation into.
    """
    try:
        if logits is None or num_chunks <= 0:
            periodic_dict.setdefault('act_chunks', []).append(None)
            return
        import torch

        flat = logits[0].detach().float().flatten()
        # double() accumulation so small bit-flips are not masked by fp32 rounding.
        chunk_sums = [float(c.double().sum().item()) for c in torch.chunk(flat, num_chunks)]
        periodic_dict.setdefault('act_chunks', []).append(chunk_sums)
    except Exception:
        if logger:
            logger.warning(f'Unable to log act_chunks at curr_step {curr_step}')
        periodic_dict.setdefault('act_chunks', []).append(None)


def _record_activation_hash(curr_step, logits, periodic_dict, logger):
    """Record a bitwise hash of the activation (Approach B: exact fingerprint).

    Hashes the raw tensor bytes of the activation. A deterministic run should produce
    byte-identical activations, so any difference -- even a single flipped bit deep
    inside the normal value range -- changes the digest and is caught. The digest is
    stored as an int so it flows through the existing scalar-based diagnosis rules.

    Args:
        curr_step (int): Current training step.
        logits: Logits tensor for the activation fingerprint.
        periodic_dict (dict): Dictionary to store periodic data; appends to 'act_hash'.
        logger: Optional logger for warnings.
    """
    try:
        if logits is None:
            periodic_dict.setdefault('act_hash', []).append(None)
            return
        import hashlib

        raw = logits[0].detach().contiguous().cpu().numpy().tobytes()
        # Truncate to 15 hex digits so the value fits comfortably in a 64-bit-ish int
        # while still making collisions astronomically unlikely.
        digest = int(hashlib.sha1(raw).hexdigest()[:15], 16)
        periodic_dict.setdefault('act_hash', []).append(digest)
    except Exception:
        if logger:
            logger.warning(f'Unable to log act_hash at curr_step {curr_step}')
        periodic_dict.setdefault('act_hash', []).append(None)


def combine_hashes(hash_values):
    """Combine an ordered list of per-checkpoint hash ints into a single stable hash int.

    Any change to any checkpoint's hash (value or position) changes the combined result,
    so the whole run collapses to one scalar that the diagnosis can compare for equality.

    Args:
        hash_values (list): Ordered list of per-checkpoint hash ints (None entries allowed).

    Returns:
        int: Combined hash, or None if there are no valid hashes.
    """
    import hashlib

    valid = [h for h in hash_values if h is not None]
    if not valid:
        return None
    joined = ','.join(str(h) for h in valid).encode('utf-8')
    return int(hashlib.sha1(joined).hexdigest()[:15], 16)


def record_periodic_fingerprint(
    curr_step,
    loss_value,
    logits,
    periodic_dict,
    check_frequency,
    enable_determinism,
    logger=None,
    num_chunks=0,
    enable_hash=False,
):
    """Record periodic fingerprints (loss and activation mean) for deterministic runs.

    Args:
        curr_step (int): Current training step.
        loss_value: Pre-converted loss float value (or None).
        logits: Logits tensor for activation fingerprint.
        periodic_dict (dict): Dictionary to store periodic data ('loss', 'act_mean', 'step').
        check_frequency (int): Frequency for fingerprint logging.
        enable_determinism (bool): Whether determinism is enabled.
        logger: Optional logger for info/warnings.
        num_chunks (int): If > 0, also record per-chunk activation checksums (Approach A).
        enable_hash (bool): If True, also record a bitwise activation hash (Approach B).
    """
    # Defensively handle invalid check_frequency values to avoid ZeroDivisionError and
    # undefined behavior for non-positive frequencies.
    if check_frequency is None or check_frequency <= 0:
        if logger:
            logger.warning(
                f'Invalid check_frequency={check_frequency} at step {curr_step}; '
                'skipping periodic fingerprint recording.'
            )
        return
    if not enable_determinism or (curr_step % check_frequency != 0):
        return

    _record_loss_fingerprint(curr_step, loss_value, periodic_dict, logger)
    _record_activation_fingerprint(curr_step, logits, periodic_dict, logger)
    if num_chunks and num_chunks > 0:
        _record_activation_chunks(curr_step, logits, periodic_dict, logger, num_chunks)
    if enable_hash:
        _record_activation_hash(curr_step, logits, periodic_dict, logger)
