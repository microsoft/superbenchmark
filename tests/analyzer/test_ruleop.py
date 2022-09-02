# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RuleOp module."""

import unittest

import pandas as pd

from superbench.analyzer import RuleOp, DiagnosisRuleType


class TestRuleOp(unittest.TestCase):
    """Test for Diagnosis Rule Ops."""
    def test_rule_op(self):
        """Test for defined rule operators."""
        # Test - get_rule_func
        # Negative case
        assert (not RuleOp.get_rule_func('fake'))
        # Positive case
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.VARIANCE)
        assert (rule_op == RuleOp.variance)

        # Test - variance and value rule function
        # Check whether arguments are valid
        # Negative case
        details = []
        categories = set()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)
        data = {'kernel-launch/event_overhead:0': 3.1, 'kernel-launch/event_overhead:1': 2}
        data_row = pd.Series(data)
        false_rule_and_baselines = [
            {
                'categories': 'KernelLaunch',
                'criteria': '>',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': '5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': '>5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x+1',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }
        ]

        for rule in false_rule_and_baselines:
            self.assertRaises(Exception, RuleOp.variance, data_row, rule, summary_data_row, details, categories)
            self.assertRaises(Exception, RuleOp.value, data_row, rule, summary_data_row, details, categories)

        # Negative case, if baseline is 0 or None in 'variance' function, raise error
        false_rule_and_baselines = [
            {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x>0.5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 0,
                }
            },
            {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x>0.5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:1': None,
                }
            },
        ]

        for rule in false_rule_and_baselines:
            self.assertRaises(ValueError, RuleOp.variance, data_row, rule, summary_data_row, details, categories)

        # Positive case
        true_baselines = [
            {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x>0.5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2,
                    'kernel-launch/event_overhead:1': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x<-0.5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2,
                    'kernel-launch/event_overhead:1': 2
                }
            }, {
                'categories': 'KernelLaunch2',
                'criteria': 'lambda x:x>0',
                'function': 'value',
                'metrics': {
                    'kernel-launch/event_overhead:0': 0
                }
            }
        ]
        # Check results
        details = []
        categories = set()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)
        # variance
        data = {'kernel-launch/event_overhead:0': 3.1, 'kernel-launch/event_overhead:1': 2}
        data_row = pd.Series(data)
        violated_metric_num = rule_op(data_row, true_baselines[0], summary_data_row, details, categories)
        assert (violated_metric_num == 1)
        assert (categories == {'KernelLaunch'})
        assert (details == ['kernel-launch/event_overhead:0(B/L: 2.0000 VAL: 3.1000 VAR: 55.00% Rule:lambda x:x>0.5)'])

        data = {'kernel-launch/event_overhead:0': 1.5, 'kernel-launch/event_overhead:1': 1.5}
        data_row = pd.Series(data)
        violated_metric_num = rule_op(data_row, true_baselines[1], summary_data_row, details, categories)
        assert (violated_metric_num == 0)
        assert (categories == {'KernelLaunch'})

        # value
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.VALUE)
        violated_metric_num = rule_op(data_row, true_baselines[2], summary_data_row, details, categories)
        assert (categories == {'KernelLaunch', 'KernelLaunch2'})
        assert ('kernel-launch/event_overhead:0(VAL: 1.5000 Rule:lambda x:x>0)' in details)
        assert ('kernel-launch/event_overhead:0(B/L: 2.0000 VAL: 3.1000 VAR: 55.00% Rule:lambda x:x>0.5)' in details)

    def test_multi_rules_op(self):
        """multi-rule check."""
        details = []
        categories = set()
        data_row = pd.Series()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)
        false_baselines = [
            {
                'categories': 'KernelLaunch',
                'criteria': 'lambda label:True if label["rule2"]>=2 else False',
                'function': 'multi_rules'
            }
        ]
        label = {}
        for rule in false_baselines:
            self.assertRaises(KeyError, RuleOp.multi_rules, false_baselines[0], details, categories, label)

        true_baselines = [
            {
                'name': 'rule1',
                'categories': 'TMP',
                'criteria': 'lambda x:x<-0.5',
                'store': True,
                'function': 'variance',
                'metrics': {
                    'resnet_models/pytorch-resnet152/throughput_train_float32': 300,
                }
            }, {
                'name': 'rule2',
                'categories': 'TMP',
                'criteria': 'lambda x:x<-0.5',
                'store': True,
                'function': 'variance',
                'metrics': {
                    'vgg_models/pytorch-vgg11/throughput_train_float32': 300
                }
            }, {
                'name': 'rule3',
                'categories': 'CNN',
                'criteria': 'lambda label:True if label["rule1"]+label["rule2"]>=2 else False',
                'store': False,
                'function': 'multi_rules'
            }
        ]
        # label["rule1"]+label["rule2"]=1, rule3 pass
        data = {
            'resnet_models/pytorch-resnet152/throughput_train_float32': 300,
            'vgg_models/pytorch-vgg11/throughput_train_float32': 100
        }
        data_row = pd.Series(data)
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType(true_baselines[0]['function']))
        label[true_baselines[0]['name']] = rule_op(data_row, true_baselines[0], summary_data_row, details, categories)
        label[true_baselines[1]['name']] = rule_op(data_row, true_baselines[1], summary_data_row, details, categories)
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType(true_baselines[2]['function']))
        violated_metric_num = rule_op(true_baselines[2], details, categories, label)
        assert (violated_metric_num == 0)
        # label["rule1"]+label["rule2"]=2, rule3 not pass
        data = {
            'resnet_models/pytorch-resnet152/throughput_train_float32': 100,
            'vgg_models/pytorch-vgg11/throughput_train_float32': 100
        }
        data_row = pd.Series(data)
        details = []
        categories = set()
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType(true_baselines[0]['function']))
        label[true_baselines[0]['name']] = rule_op(data_row, true_baselines[0], summary_data_row, details, categories)
        label[true_baselines[1]['name']] = rule_op(data_row, true_baselines[1], summary_data_row, details, categories)
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType(true_baselines[2]['function']))
        violated_metric_num = rule_op(true_baselines[2], details, categories, label)
        assert (violated_metric_num)
        assert ('TMP' not in categories)
        assert ('CNN' in categories)
        assert (
            details == [
                'resnet_models/pytorch-resnet152/throughput_train_float32' +
                '(B/L: 300.0000 VAL: 100.0000 VAR: -66.67% Rule:lambda x:x<-0.5)',
                'vgg_models/pytorch-vgg11/throughput_train_float32' +
                '(B/L: 300.0000 VAL: 100.0000 VAR: -66.67% Rule:lambda x:x<-0.5)',
                'rule3:lambda label:True if label["rule1"]+label["rule2"]>=2 else False'
            ]
        )

    def test_failure_check_op(self):
        """Test for failure_check op."""
        details = []
        categories = set()
        data_row = pd.Series()
        summary_data_row = pd.Series(dtype=float)

        # invalid rule
        false_baselines = [{'categories': 'FailedTest', 'criteria': 'lambda x:x!=0', 'function': 'failure_check'}]
        label = {}
        for rule in false_baselines:
            self.assertRaises(
                Exception, RuleOp.failure_check, data_row, rule, summary_data_row, details, categories, rule
            )

        true_baselines = [
            {
                'name': 'rule1',
                'categories': 'FailedTest',
                'criteria': 'lambda x:x!=0',
                'function': 'failure_check',
                'metrics': {
                    'gemm-flops/return_code:0': -1,
                    'gemm-flops/return_code:1': -1,
                    'resnet_models/pytorch-resnet152/return_code': -1,
                }
            }, {
                'name': 'rule2',
                'categories': 'FailedTest',
                'criteria': 'lambda x:x!=0',
                'function': 'failure_check',
                'metrics': {
                    'gemm-flops/return_code:0': -1,
                    'gemm-flops/return_code:1': -1,
                    'gemm-flops/return_code:2': -1,
                    'resnet_models/pytorch-resnet152/return_code': -1,
                }
            }
        ]
        # positive case
        data = {
            'gemm-flops/return_code:0': 0,
            'gemm-flops/return_code:1': 0,
            'resnet_models/pytorch-resnet152/return_code': 0,
        }
        data_row = pd.Series(data)
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType(true_baselines[0]['function']))
        label[true_baselines[0]['name']
              ] = rule_op(data_row, true_baselines[0], summary_data_row, details, categories, true_baselines[0])
        assert (label[true_baselines[0]['name']] == 0)

        # negative cases
        # 1. return_code != 0
        data = {
            'gemm-flops/return_code:0': 0,
            'gemm-flops/return_code:1': -1,
            'resnet_models/pytorch-resnet152/return_code': -1,
        }
        details = []
        categories = set()
        data_row = pd.Series(data)
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType(true_baselines[0]['function']))
        label[true_baselines[0]['name']
              ] = rule_op(data_row, true_baselines[0], summary_data_row, details, categories, true_baselines[0])
        assert (label[true_baselines[0]['name']] != 0)
        assert ({'FailedTest'} == categories)
        assert (
            details == [
                'gemm-flops/return_code:1(VAL: -1.0000 Rule:lambda x:x!=0)',
                'resnet_models/pytorch-resnet152/return_code(VAL: -1.0000 Rule:lambda x:x!=0)',
            ]
        )

        # 2. metric not in raw_data or the value is none, miss test
        data = {
            'gemm-flops/return_code:0': 0,
            'gemm-flops/return_code:1': 0,
            'resnet_models/pytorch-resnet152/return_code': 0,
        }
        details = []
        categories = set()
        data_row = pd.Series(data)
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType(true_baselines[0]['function']))
        label[true_baselines[1]['name']
              ] = rule_op(data_row, true_baselines[1], summary_data_row, details, categories, true_baselines[1])
        assert (label[true_baselines[1]['name']] != 0)
        assert ({'FailedTest'} == categories)
        assert (details == ['gemm-flops/return_code:2_miss'])

        # 3. metric_regex written in rules is not matched by any metric, miss test
        data = {
            'gemm-flops/return_code:0': 0,
            'gemm-flops/return_code:1': 0,
            'resnet_models/pytorch-resnet152/return_code': 0,
        }
        details = []
        categories = set()
        data_row = pd.Series(data)
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType(true_baselines[0]['function']))
        label[true_baselines[1]['name']
              ] = rule_op(data_row, true_baselines[0], summary_data_row, details, categories, true_baselines[1])
        assert (label[true_baselines[1]['name']] != 0)
        assert ({'FailedTest'} == categories)
        assert (details == ['gemm-flops/return_code:2_miss'])
