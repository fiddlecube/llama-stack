# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval import Eval
from llama_stack.apis.eval.eval import EvalTaskConfig, EvaluateResponse


class FiddleCubeRedTeamingAdapter(Eval):
    def __init__(self):
        self.jobs = {}

    async def run_eval(self, eval_id: str, eval_config: EvalTaskConfig) -> Job:
        # call the FiddleCube API to run the red-teaming
        # convert EvalTaskConfig to FiddleCube API input
        # Get the FiddleCube response
        # convert FiddleCube response to EvaluateResponse
        # set the job_id to the length of the jobs list
        # in the dict of jobs, set the job_id to the EvaluateResponse
        # return the job
        # refer to llama_stack/providers/inline/eval/meta_reference/eval.py for the implementation of run_eval
        return Job(job_id=len(self.jobs))

    async def evaluate_rows(
        self, eval_id: str, input_rows: List[Dict[str, Any]], scoring_functions: List[str], eval_config: EvalTaskConfig
    ) -> EvaluateResponse:
        raise NotImplementedError("FiddleCube Red Teaming Adapter does not support evaluate_rows")

    async def job_status(self, eval_id: str, job_id: str) -> JobStatus:
        if job_id in self.jobs:
            return JobStatus.completed

        return None

    async def job_result(self, eval_id: str, job_id: str) -> EvaluateResponse:
        status = await self.job_status(eval_id, job_id)
        if not status or status != JobStatus.completed:
            raise ValueError(f"Job is not completed, Status: {status.value}")

        return self.jobs[job_id]

    async def job_cancel(self, eval_id: str, job_id: str) -> None:
        raise NotImplementedError("FiddleCube Red Teaming Adapter does not support job_cancel")
