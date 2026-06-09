"""
Detalled request logging system.

Logs all user requests with responses, models, timing, and errors to CSV format.
"""

import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RequestLogger:
    """
    Logger for detailed request tracking in CSV format.

    Format: Request #, User ID, Query, Response, Model, Response Time, Error Code
    """

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        self.log_file = os.path.join(logs_dir, "requests_log.csv")
        self._ensure_log_dir()
        self._init_log_file()
        self._request_counter = self._get_last_request_number()
        self._lock = None

    def _get_last_request_number(self) -> int:
        """Get the last request number from existing log file."""
        if not os.path.exists(self.log_file):
            return 0

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row

                if last_row and "Request_Number" in last_row:
                    return int(last_row["Request_Number"])
        except (ValueError, KeyError, csv.Error) as e:
            logger.warning(f"Failed to read last request number: {e}")

        return 0

    def _get_key_index(self, api_key: str) -> int:
        """
        Determine the index of an API key based on its unique suffix.

        This helps identify which key was used when multiple keys have similar prefixes.
        """
        if not api_key:
            return 0

        # Extract the unique suffix (last 4 characters) to identify the key
        suffix = api_key[-4:] if len(api_key) >= 4 else api_key

        # Common suffix patterns for Gemini API keys
        suffix_patterns = {
            "wRiw": 0,  # Example: AIza...wRiw
            "NMiw": 1,  # Example: AIza...NMiw
            "dxXs": 2,  # Example: AIza...dxXs
            "tNzA": 3,  # Example: AIza...tNzA
        }

        return suffix_patterns.get(suffix, 0)

    def _ensure_log_dir(self):
        """Create logs directory if it doesn't exist."""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def _init_log_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        file_exists = os.path.exists(self.log_file)

        if not file_exists:
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Request_Number",
                        "Timestamp",
                        "User_ID",
                        "Platform",
                        "Query",
                        "Response",
                        "Model",
                        "API_Key",
                        "Response_Time_sec",
                        "Error_Code",
                        "Session_ID",
                    ]
                )
            logger.info(f"Created request log file: {self.log_file}")

    def log_request(
        self,
        user_id: str,
        platform: str,
        query: str,
        response: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        response_time: Optional[float] = None,
        error_code: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Log a request with all details.

        Args:
            user_id: User identifier (Telegram ID, etc.)
            platform: Platform name (telegram, web, etc.)
            query: User's query text
            response: System response text (truncated if too long)
            model: Model name that generated response
            api_key: API key used for the request
            response_time: Response time in seconds
            error_code: Error code if request failed
            session_id: Session identifier
        """
        try:
            self._request_counter += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Truncate long texts for CSV
            max_length = 500
            query_truncated = query[:max_length] + "..." if len(query) > max_length else query
            response_truncated = (
                response[:max_length] + "..." if response and len(response) > max_length else (response or "")
            )

            # Format response time
            response_time_str = f"{response_time:.2f}" if response_time is not None else ""

            # Enhanced API key format: show first 4, last 4, and key index
            api_key_masked = ""
            if api_key:
                if len(api_key) > 8:
                    # Format: AIza...xX42 (Index: 0)
                    api_key_masked = f"{api_key[:4]}...{api_key[-4:]} (Index: {self._get_key_index(api_key)})"
                else:
                    # Format: AIz... (Index: 0)
                    api_key_masked = f"{api_key[:4]}... (Index: {self._get_key_index(api_key)})"

            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self._request_counter,
                        timestamp,
                        user_id,
                        platform,
                        query_truncated,
                        response_truncated,
                        model or "",
                        api_key_masked,
                        response_time_str,
                        error_code or "",
                        session_id or "",
                    ]
                )

            logger.debug(f"Logged request #{self._request_counter} for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to log request: {e}")

    def log_success(
        self,
        user_id: str,
        platform: str,
        query: str,
        response: str,
        model: str,
        response_time: float,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Log successful request."""
        self.log_request(
            user_id=user_id,
            platform=platform,
            query=query,
            response=response,
            model=model,
            api_key=api_key,
            response_time=response_time,
            session_id=session_id,
        )

    def log_error(
        self,
        user_id: str,
        platform: str,
        query: str,
        error_code: str,
        api_key: Optional[str] = None,
        response_time: Optional[float] = None,
        session_id: Optional[str] = None,
    ):
        """Log failed request."""
        self.log_request(
            user_id=user_id,
            platform=platform,
            query=query,
            api_key=api_key,
            error_code=error_code,
            response_time=response_time,
            session_id=session_id,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics from the log file."""
        try:
            if not os.path.exists(self.log_file):
                return {"total_requests": 0, "errors": 0, "success_rate": 0}

            total = 0
            errors = 0

            with open(self.log_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    if row["Error_Code"]:
                        errors += 1

            success_rate = ((total - errors) / total * 100) if total > 0 else 0

            return {"total_requests": total, "errors": errors, "success_rate": round(success_rate, 2)}

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_requests": 0, "errors": 0, "success_rate": 0}
