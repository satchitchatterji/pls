"""FrozenLake researcher integration quickstart.

This script demonstrates zero edits to `pls/` internals by registering
runtime + sensor from a researcher-owned file before calling train.
"""

import os

from pls.workflows.execute_workflow import train
from research_registration import register_for_research


if __name__ == "__main__":
    register_for_research()
    cwd = os.path.dirname(__file__)
    config_file = os.path.join(cwd, "seed1", "config.json")
    train(config_file)
