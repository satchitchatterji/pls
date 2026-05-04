import os

from pls.workflows.execute_workflow import train

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    config_file = os.path.join(cwd, "seed1", "config.json")
    train(config_file)
