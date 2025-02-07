# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
import httpx

from typing import Any, Dict, List

from llama_stack.apis.inference import Message

from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate

from .config import FiddlecubeSafetyConfig


logger = logging.getLogger(__name__)


class FiddlecubeSafetyAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: FiddlecubeSafetyConfig) -> None:
        self.config = config
        self.registered_shields = []

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        print("Shield", shield)
        pass

    async def run_shield(
        self, shield_id: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        # Convert the `messages` into the format FiddleCube expects
        content_messages = [{"text": {"text": message.content}} for message in messages]
        logger.debug(f"run_shield::final:messages::{json.dumps(content_messages, indent=2)}:")
        print("URL::::", self.config.api_url)
        # Make a call to the FiddleCube API for guardrails
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.config.api_url + '/safety/redteam/benchmark?model_name=gpt-4o &system_prompt=what%20is%20the%20red%20color%20on%20your%20face'
                )

            print("Response:::", response.status_code)

        # Check if the response is successful
        if response.status_code != 200:
            print(f"FiddleCube API error: {response.status_code} - {response.text}")
            raise RuntimeError("Failed to run shield with FiddleCube API")

        # Convert the response into the format RunShieldResponse expects
        response_data = response.json()
        print("Response data", response_data)


        return RunShieldResponse()
