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


def record_periodic_fingerprint(
    curr_step, loss_value, logits, periodic_dict, check_frequency, enable_determinism, logger=None
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
