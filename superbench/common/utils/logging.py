# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import socket
import logging
import sys


class Logger:
    '''Logger class with customized format.

    Args:
        name: project name.
        level: logging level, default is INFO.
        stream: stream object, such as stdout or file object,
                default is stdout.
    '''
    @staticmethod
    def create_logger(name, level=logging.INFO, stream=sys.stdout):
        formatter = logging.Formatter(
            '%(asctime)s - %(hostname)s - '
            '%(filename)s:%(lineno)d - '
            '%(levelname)s: %(message)s')

        handler = logging.StreamHandler(stream=stream)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger = logging.LoggerAdapter(
            logger, extra={'hostname': socket.gethostname()})

        return logger


logger = Logger.create_logger('SuperBench', level=logging.INFO)
