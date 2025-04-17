from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime

class ChatHistory:
    def __init__(self, history_file: str = "chat_histories.json", max_history_per_user: int = 3):
        """Initializes ChatHistory to manage multiple user histories."""
        self.history_file = history_file
        self.max_history_per_user = max_history_per_user
        # Use a dictionary to store histories, keyed by user_key (e.g., user_id or 'anonymous')
        self.histories: Dict[str, List[Dict[str, Any]]] = self._load_histories()
        print(f"Chat histories loaded for keys: {list(self.histories.keys())}") # Debug log

    def _load_histories(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all user chat histories from a single file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Ensure data is a dictionary
                    if isinstance(data, dict):
                        return data
                    else:
                        print("Warning: History file does not contain a dictionary. Creating new.")
                        # Optionally, try to migrate old list format if needed, or just start fresh
                        return {}
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {self.history_file}. Starting fresh.")
                return {}
            except Exception as e:
                print(f"Error loading chat histories: {e}. Starting fresh.")
                return {}
        return {}

    def _save_histories(self):
        """Save all user chat histories to the file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.histories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving chat histories: {e}")

    def add_chat(self, user_key: str, query: str, response: str):
        """Add a new chat entry for a specific user key."""
        if user_key not in self.histories:
            self.histories[user_key] = []

        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response
        }
        self.histories[user_key].append(chat_entry)

        # Keep only the last max_history entries for this user
        if len(self.histories[user_key]) > self.max_history_per_user:
            self.histories[user_key] = self.histories[user_key][-self.max_history_per_user:]

        self._save_histories() # Save after modification

    def get_history(self, user_key: str) -> List[Dict[str, Any]]:
        """Get all chat history for a specific user key."""
        return self.histories.get(user_key, [])

    def get_recent_history(self, user_key: str) -> str:
        """Get recent chat history for a specific user key as a formatted string."""
        user_history = self.histories.get(user_key, [])
        if not user_history:
            return ""

        history_text = "Lịch sử trò chuyện gần đây:\n"
        for entry in user_history:
            history_text += f"Q: {entry['query']}\n"
            history_text += f"A: {entry['response']}\n"
        return history_text.strip() # Remove trailing newline

    def clear_history(self, user_key: str):
        """Clear chat history for a specific user key."""
        if user_key in self.histories:
            self.histories[user_key] = []
            self._save_histories()
            print(f"Chat history cleared for user_key: {user_key}")
        else:
            print(f"No chat history found to clear for user_key: {user_key}") 