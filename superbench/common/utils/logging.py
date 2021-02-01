# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench loggin module."""

import socket
import logging
import sys


class Logger:
    """Logger class which creates logger instance."""
    @staticmethod
    def create_logger(name, level=logging.INFO, stream=sys.stdout):
        """Create logger instance with customized format.

        Args:
            name (str): project name.
            level (int): logging level, default is INFO.
            stream (TextIOBase): stream object, such as stdout or file object,
              default is sys.stdout.

        Return:
            logger with the specified name, level and stream.
        """
        formatter = logging.Formatter(
            '%(asctime)s - %(hostname)s - '
            '%(filename)s:%(lineno)d - '
            '%(levelname)s: %(message)s'
        )

        handler = logging.StreamHandler(stream=stream)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger = logging.LoggerAdapter(logger, extra={'hostname': socket.gethostname()})

        return logger


logger = Logger.create_logger('SuperBench', level=logging.INFO)
