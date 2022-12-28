# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench stdout logging module."""

import sys


class SuperBenchStdoutLogger:
    """Logger class to enable or disable to redirect STDOUT and STDERR to file."""
    class StdoutLoggerStream:
        """StdoutLoggerStream class which redirect the sys.stdout to file."""
        def __init__(self, filename, rank):
            """Init the class with filename.

            Args:
                filename (str): the path of the file to save the log
                rank (int): the rank id
            """
            self.terminal = sys.stdout
            self.rank = rank
            self.log = open(filename, 'a')

        def __getattr__(self, attr):
            """Override __getattr__.

            Args:
                attr (str): Attribute name.

            Returns:
                Any: Attribute value.
            """
            return getattr(self.terminal, attr)

        def write(self, message):
            """Write the message to the stream.

            Args:
                message (str): the message to log.
            """
            message = f'[{self.rank}]: {message}'
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            """Override flush."""
            pass

    def add_file_handler(self, filename):
        """Init the class with filename.

        Args:
            filename (str): the path of file to save the log
        """
        self.filename = filename

    def start(self, rank):
        """Start the logger to redirect the sys.stdout to file.
        
        Args:
            rank (int): the rank id
        """
        self.logger_stream = self.StdoutLoggerStream(self.filename, rank)
        sys.stdout = self.logger_stream
        sys.stderr = sys.stdout

    def stop(self):
        """Restore the sys.stdout to termital."""
        if self.logger_stream:
            sys.stdout.log.close()
            sys.stdout = sys.stdout.terminal

    def log(self, message):
        """Write the message into the logger.

        Args:
            message (str): the message to log.
        """
        self.logger_stream.write(message)


stdout_logger = SuperBenchStdoutLogger()
