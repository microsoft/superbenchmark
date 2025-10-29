import logging
import sys
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal text and force colors
init(autoreset=True, strip=False)

# Define a function to set up the logging configuration
def setup_logging():
    """
    Configures the root logger for the application.
    Messages of INFO level and higher will be printed to stderr (console).
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Set the overall minimum level for the logger
    root_logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        # Create a StreamHandler to output logs to the console (stderr by default)
        console_handler = logging.StreamHandler(sys.stderr)
        
        # Set the level for this specific handler
        console_handler.setLevel(logging.INFO)

        # Create a formatter for the log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set the formatter for the console handler
        console_handler.setFormatter(formatter)
        
        # Add the console handler to the root logger
        root_logger.addHandler(console_handler)

        # Optional: Add a FileHandler for logging to a file
        # file_handler = logging.FileHandler('app.log')
        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(formatter)
        # root_logger.addHandler(file_handler)

# Call setup_logging when this module is imported
setup_logging()

# Expose the root logger so all modules use the same logger instance
class SimpleLogger:
    def info(self, msg):
        print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} {msg}")
    def error(self, msg):
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")
    def debug(self, msg):
        print(f"{Fore.CYAN}[DEBUG]{Style.RESET_ALL} {msg}")
    def warning(self, msg):
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {msg}")

logger = SimpleLogger()
