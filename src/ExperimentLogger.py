# src/ExperimentLogger.py


import os


class ExperimentLogger:

    def __init__(self, output_dir):

        self.path = os.path.join(
            output_dir,
            "report.txt"
        )

    def log(self, text):

        with open(self.path, "a") as f:
            f.write(text + "\n")
