# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import sys

from pace_agent.llm.client import LLMClient
from pace_agent.profile.profiler import import_agent
from pace_agent.replay import Player, Trajectory


def main(run_dir):
    traj = Trajectory.load(run_dir)
    agent = import_agent(traj.meta.get("agent"))
    llm = LLMClient(endpoint="", model="")
    Player().replay(traj, agent, llm, "naive", 3)


if __name__ == "__main__":
    main(sys.argv[1])
