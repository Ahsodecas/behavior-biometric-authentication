import json
import os
import hashlib
import src.constants as constants

class UserManagementUtility:
    def __init__(self, filename="users.json"):
        self.path = os.path.join(constants.PATH_METRICS, filename)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def hash_password(self, password: str) -> str:
        """Return the SHA-256 hash of the password."""
        return hashlib.sha256(password.encode()).hexdigest()

    def get_superuser_username(self):
        with open(self.path, "r") as f:
            users = json.load(f)
        for u in users:
            if u.get("superuser", False):
                return u.get("username", "Unknown")
        return None

    def get_local_user_username(self):
        with open(self.path, "r") as f:
            users = json.load(f)
        for u in users:
            if not u.get("superuser", False):
                return u.get("username", "Unknown")
        return None

    def create_superuser(self, username: str, password: str):
        hashed_pass = self.hash_password(password)
        with open(self.path, "r") as f:
            users = json.load(f)
        users.append({"username": username, "password": hashed_pass, "superuser": True})
        with open(self.path, "w") as f:
            json.dump(users, f)

    def create_local_user(self, username: str, password: str):
        hashed_pass = self.hash_password(password)
        with open(self.path, "r") as f:
            users = json.load(f)
        for u in users:
            if u.get("username") == username:
                return
        users.append({"username": username, "password": hashed_pass, "superuser": False})
        with open(self.path, "w") as f:
            json.dump(users, f)

    def verify_user(self, username: str, password: str) -> bool:
        """Check if the username and password match the stored superuser."""
        with open(self.path, "r") as f:
            users = json.load(f)
        for u in users:
            if u["username"] == username:
                if u["password"] == self.hash_password(password):
                    return True
        return False