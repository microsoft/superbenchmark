"""Configuration for the Report Gen Agent."""

import importlib.resources as pkg_resources
import os
from .utils.logger import logger

SERVICE_MODE = os.getenv('SERVICE_MODE', 'local')  # Default to 'local' if not set
logger.info(f"Service mode set to: {SERVICE_MODE}")

if SERVICE_MODE == 'local':
    DATA_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "data")
elif SERVICE_MODE == 'docker':
    DATA_DIR = "/mnt/data"
elif SERVICE_MODE == 'remote':
    DATA_DIR = "/mnt/data"
else:
    DATA_DIR = ""
    logger.error(f"Unknown SERVICE_MODE: {SERVICE_MODE}. Defaulting DATA_DIR to empty string.")


PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
BLOB_LOCAL_DIR = os.getenv('MOUNT_PATH', '/mnt/blob/agents/infrawise-report-gen') # default value: "/mnt/blob/agents/infrawise-report-gen"
CASE_DIR = os.path.join(DATA_DIR, "case")
BUILD_DIR = os.path.join(DATA_DIR, "build_pdf")
BUILD_TEMP_DIR = os.path.join(DATA_DIR, "build_temp")

GPT_VER = 'azure/gpt-4o'
#"azure/gpt-4-32k "
#"azure/gpt-4o"


AGENT_PORT = int(os.getenv('AGENT_PORT', '50000'))