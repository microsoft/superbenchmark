# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench stdout logging module."""

import sys


class SuperBenchStdoutLogger:
    """Logger class to enable or disable to redirect STDOUT and STDERR to file."""
    def __init__(self, filename):
        """Init the class with filename.

        Args:
            filename (str): the path of the file to save the log
        """
        self.filename = filename

    class StdoutLoggerStream:
        """StdoutLoggerStream class which redirect the sys.stdout to file."""
        def __init__(self, filename):
            """Init the class with filename.

            Args:
                filename (str): the path of the file to save the log
            """
            self.terminal = sys.stdout
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
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            """Override flush."""
            pass

    def start(self):
        """Start the logger to redirect the sys.stdout to file."""
        sys.stdout = self.StdoutLoggerStream(self.filename)
        sys.stderr = sys.stdout

    def stop(self):
        """Restore the sys.stdout to termital."""
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
