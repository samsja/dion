import json


class FakeWandB:
    def __init__(self, log_file: str):
        print(f"Wandb disabled, logging to file instead: {log_file}")
        self.log_file = log_file

    def init(self, *args, **kwargs):
        pass

    def login(self, *args, **kwargs):
        pass

    def log(self, data):
        log_str = json.dumps(data)
        with open(self.log_file, "a") as f:
            f.write(log_str + "\n")
