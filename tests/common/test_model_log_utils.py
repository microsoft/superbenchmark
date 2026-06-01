# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for model_log_utils module."""

from unittest.mock import Mock
from superbench.common import model_log_utils


class TestRecordStepLoss:
    """Tests for record_step_loss function."""
    def test_record_loss_conversion_failure(self):
        """Test exception handling when loss conversion fails."""
        logger = Mock()
        losses_list = []

        # Create a mock object that raises exception on conversion
        bad_loss = Mock()
        bad_loss.detach.side_effect = RuntimeError('Conversion failed')

        result = model_log_utils.record_step_loss(bad_loss, curr_step=5, losses_list=losses_list, logger=logger)

        assert result is None
        assert losses_list == [None]
        logger.info.assert_called_once_with('Unable to convert loss to float at step 5')

    def test_record_loss_success(self):
        """Test successful loss recording."""
        logger = Mock()
        losses_list = []

        # Create a mock tensor with detach and item methods
        loss = Mock()
        loss.detach.return_value.item.return_value = 2.5

        result = model_log_utils.record_step_loss(loss, curr_step=10, losses_list=losses_list, logger=logger)

        assert result == 2.5
        assert losses_list == [2.5]

    def test_record_loss_from_float(self):
        """Test recording loss from plain float value."""
        losses_list = []

        result = model_log_utils.record_step_loss(1.234, curr_step=1, losses_list=losses_list, logger=None)

        assert result == 1.234
        assert losses_list == [1.234]


class TestRecordPeriodicFingerprint:
    """Tests for record_periodic_fingerprint function."""
    def test_skips_when_determinism_disabled(self):
        """Test that fingerprint is not recorded when determinism is disabled."""
        periodic_dict = {}
        model_log_utils.record_periodic_fingerprint(
            curr_step=100,
            loss_value=1.0,
            logits=None,
            periodic_dict=periodic_dict,
            check_frequency=10,
            enable_determinism=False,
            logger=None
        )
        assert periodic_dict == {}

    def test_skips_when_not_at_frequency(self):
        """Test that fingerprint is not recorded when not at check frequency."""
        periodic_dict = {}
        model_log_utils.record_periodic_fingerprint(
            curr_step=15,
            loss_value=1.0,
            logits=None,
            periodic_dict=periodic_dict,
            check_frequency=10,
            enable_determinism=True,
            logger=None
        )
        assert periodic_dict == {}

    def test_records_at_frequency(self):
        """Test that fingerprint is recorded at check frequency."""
        periodic_dict = {}
        model_log_utils.record_periodic_fingerprint(
            curr_step=20,
            loss_value=1.5,
            logits=None,
            periodic_dict=periodic_dict,
            check_frequency=10,
            enable_determinism=True,
            logger=None
        )
        assert 'loss' in periodic_dict
        assert periodic_dict['loss'] == [1.5]
        assert 'step' in periodic_dict
        assert periodic_dict['step'] == [20]
