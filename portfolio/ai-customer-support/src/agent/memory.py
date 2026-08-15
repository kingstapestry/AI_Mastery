# ======================
# Class: Conversation Memory
# ======================
class ConversationMemory:
    def __init__(self):
        self._history: list[dict[str, str]] = []

    def add_user_message(self, message: str):
        # Add a message with role "user"
        self._history.append({
            "role": "user", 
            "content": message
        })

    def add_ai_message(self, message: str):
        # Add a message with role "assistant"
        self._history.append({
            "role": "assistant",
            "content": message
        })

    def get_history(self) -> list[dict[str, str]]:
        # Return the full history
        return self._history.copy()     # prevent accidental modifications

    def clear(self):
        self._history = []

