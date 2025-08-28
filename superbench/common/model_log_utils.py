import json
import torch


def save_model_log(filepath, metadata, losses, fingerprints):
    """Save model run log to a JSON file.

    Args:
        filepath (str): Path to save the log file.
        metadata (dict): Model and run metadata.
        losses (list): List of per-step loss values.
        fingerprints (dict): Dictionary of periodic fingerprints (loss, act_mean, step).
    """
    data = {
        'schema_version': 1,
        'metadata': metadata,
        'per_step_fp32_loss': [float(x) for x in losses],
        'fingerprints': fingerprints,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_model_log(filepath):
    """Load model run log from a JSON file.

    Args:
        filepath (str): Path to the log file.

    Returns:
        dict: Loaded log data.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_model_logs(current, reference):
    """Compare two model run logs for determinism.

    Args:
        current (dict): Current run log data.
        reference (dict): Reference run log data.

    Returns:
        bool: True if logs match (deterministic), False otherwise.

    Raises:
        ValueError: If metadata does not match.
    """
    # Check metadata match (model, params, etc.)
    for key in [
        'model_name',
        'precision',
        'seed',
        'batch_size',
        'seq_len',
        'num_steps',
    ]:
        if str(current['metadata'].get(key)) != str(reference['metadata'].get(key)):
            raise ValueError(
                f'Metadata mismatch for {key}: {current["metadata"].get(key)} vs {reference["metadata"].get(key)}'
            )
    # Compare per-step loss (full series)
    curr_loss = torch.tensor(current['per_step_fp32_loss'])
    ref_loss = torch.tensor(reference['per_step_fp32_loss'])
    equal_loss = torch.equal(curr_loss, ref_loss)

    # Compare fingerprints: ensure steps align, then compare loss/act_mean values
    curr_fp = current.get('fingerprints') or {}
    ref_fp = reference.get('fingerprints') or {}

    # Steps must match exactly (order and values)
    curr_steps = curr_fp.get('step') or []
    ref_steps = ref_fp.get('step') or []
    steps_match = curr_steps == ref_steps

    def _cmp_series(curr_list, ref_list):
        """Compare two lists of values for exact equality using torch.

        Args:
            curr_list (list): Current values.
            ref_list (list): Reference values.

        Returns:
            bool: True if lists are equal, False otherwise.
        """
        if curr_list is None or ref_list is None:
            return False
        curr_t = torch.tensor(curr_list)
        ref_t = torch.tensor(ref_list)

        return torch.equal(curr_t, ref_t)

    equal_fp_loss = _cmp_series(curr_fp.get('loss'), ref_fp.get('loss'))
    equal_fp_act = _cmp_series(curr_fp.get('act_mean'), ref_fp.get('act_mean'))

    return bool(equal_loss and steps_match and equal_fp_loss and equal_fp_act)
