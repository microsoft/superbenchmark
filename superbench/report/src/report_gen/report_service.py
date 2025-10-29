import os
from flask import Flask, jsonify, request
import threading

from .report_gen import ReportGen
from .utils import logger
from .config import AGENT_PORT

class ReportService:
	"""Flask app and endpoint manager for ReportGen."""
	def __init__(self, report_gen: ReportGen):
		self.report_gen = report_gen
		self.host = os.getenv('AGENT_HOST', '127.0.0.1')
		self.app = Flask(__name__)
		self.app.add_url_rule('/reportgen/api/status', view_func=self.status, methods=['GET'])
		self.app.add_url_rule('/reportgen/api/operation', view_func=self.instance_operation, methods=['POST'])

	def status(self):
		"""GET endpoint for health/status check."""
		return jsonify({"status": "running"})

	def instance_operation(self):
		"""POST endpoint to handle reportgen operations."""
		logger.info("Received request at /reportgen/api/operation")
		try:
			data = request.get_json()
			in_parameters = self.report_gen.build_in_parameters(data)
			out_parameters = self.report_gen.perform_operation(in_parameters)
			response = {
				"status": "success",
				"data": out_parameters.__dict__
			}
			return jsonify(response), 200
		except Exception as e:
			logger.info(f"Error handling reportgen/api/operation: {e}")
			return jsonify({"status": "error", "message": str(e)}), 500

	def run_http_server(self):
		"""Start the Flask HTTP server."""
		port = AGENT_PORT
		logger.info(f'ReportGen HTTP server running on {self.host}:{port}')
		self.app.run(port=port, host=self.host)

	def run(self):
		"""Start the HTTP server in a separate thread."""
		http_thread = threading.Thread(target=self.run_http_server)
		http_thread.start()
		http_thread.join()
