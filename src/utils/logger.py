import os
from datetime import datetime
import os
from datetime import datetime
import src.constants as constants


class Logger:
    def __init__(self, logs_path: str = "logs"):
        self.logs_path = constants.PATH_LOGS
        os.makedirs(self.logs_path, exist_ok=True)

    def log(self, username: str, event: str):
        """
        Append an activity log for a given user.
        Each log entry includes a timestamp.
        """
        log_file = os.path.join(self.logs_path, f"{username}_activity.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {event}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def get_logs(self, username: str) -> str:
        log_file = os.path.join(self.logs_path, f"{username}_activity.log")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                return f.read()
        return "No activity yet for this user."

    def clear_logs(self, username: str):
        log_file = os.path.join(self.logs_path, f"{username}_activity.log")
        if os.path.exists(log_file):
            os.remove(log_file)