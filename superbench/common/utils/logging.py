# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench loggin module."""

import socket
import logging
import sys
import io


class LoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter class which add customized function for log error and assert."""
    def log_assert(self, condition, msg, *args):
        """Log error and assert.

        Args:
            condition (bool): condation result.
            msg (str): logging message.
            args (dict): arguments dict for message.
        """
        if not condition:
            self.error(msg, *args)
            assert (False), msg % args


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
        is_level_valid = True
        if level not in logging._levelToName.keys():
            invalid_level = level
            level = logging.INFO
            is_level_valid = False

        is_stream_valid = True
        if not isinstance(stream, io.IOBase):
            invalid_stream = stream
            stream = sys.stdout
            is_stream_valid = False

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
        logger = LoggerAdapter(logger, extra={'hostname': socket.gethostname()})

        if not is_level_valid:
            logger.error(
                'Log level is invalid, replace it to logging.INFO - level: {}, expected: {}'.format(
                    invalid_level, ' '.join(str(x) for x in logging._levelToName.keys())
                )
            )

        if not is_stream_valid:
            logger.error('Stream is invalid, replace it to sys.stdout - stream type: {}'.format(type(invalid_stream)))

        return logger


logger = Logger.create_logger('SuperBench', level=logging.INFO)
