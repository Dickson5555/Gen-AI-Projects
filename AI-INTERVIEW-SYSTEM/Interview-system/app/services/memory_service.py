class SessionMemory:

    def __init__(self):
        self.memory = {}

    def get_history(self, session_id):

        return self.memory.get(
            session_id,
            []
        )

    def add_message(
        self,
        session_id,
        role,
        message
    ):

        if session_id not in self.memory:
            self.memory[session_id] = []

        self.memory[session_id].append(
            f"{role}: {message}"
        )


memory_store = SessionMemory()