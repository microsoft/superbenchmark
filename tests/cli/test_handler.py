# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI handler test."""

import unittest

import superbench.cli._handler as cli_handler


class CLIHandlerTestCase(unittest.TestCase):
    """A class for CLI handler test cases."""
    def test_split_docker_domain(self):
        """Test split_docker_domain function which splits Docker image name to domain and remainder part.

        Test cases are ported from
        https://github.com/distribution/distribution/blob/v2.7.1/reference/normalize_test.go#L468-L528.
        """
        test_cases = [
            {
                'input': 'test.com/foo',
                'domain': 'test.com',
                'name': 'foo',
            },
            {
                'input': 'test_com/foo',
                'domain': 'docker.io',
                'name': 'test_com/foo',
            },
            {
                'input': 'docker/migrator',
                'domain': 'docker.io',
                'name': 'docker/migrator',
            },
            {
                'input': 'test.com:8080/foo',
                'domain': 'test.com:8080',
                'name': 'foo',
            },
            {
                'input': 'test-com:8080/foo',
                'domain': 'test-com:8080',
                'name': 'foo',
            },
            {
                'input': 'foo',
                'domain': 'docker.io',
                'name': 'library/foo',
            },
            {
                'input': 'xn--n3h.com/foo',
                'domain': 'xn--n3h.com',
                'name': 'foo',
            },
            {
                'input': 'xn--n3h.com:18080/foo',
                'domain': 'xn--n3h.com:18080',
                'name': 'foo',
            },
            {
                'input': 'docker.io/foo',
                'domain': 'docker.io',
                'name': 'library/foo',
            },
            {
                'input': 'docker.io/library/foo',
                'domain': 'docker.io',
                'name': 'library/foo',
            },
            {
                'input': 'docker.io/library/foo/bar',
                'domain': 'docker.io',
                'name': 'library/foo/bar',
            },
        ]
        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                domain, name = cli_handler.split_docker_domain(test_case['input'])
                self.assertEqual(domain, test_case['domain'])
                self.assertEqual(name, test_case['name'])
