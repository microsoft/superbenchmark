"""Main module."""

import argparse  # noqa: E402


from .report_gen import ReportGen  # noqa: E402
from .report_service import ReportService  # Import the new service


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, help='user question, query or inquery', default=None)
    args = parser.parse_args()
    return args



def main():
    """Main function."""
    args = parse_args()
    report_gen = ReportGen(args)
    service = ReportService(report_gen)
    service.run()  # Start the Flask server


if __name__ == '__main__':
    """Entry point."""
    main()
