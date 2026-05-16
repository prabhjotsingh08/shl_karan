"""Simulate multi-turn conversations against the chat agent or API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from ..agent.chat_agent import ChatAgent
from ..data_models import ChatMessage, ChatResponse


Handler = Callable[[Sequence[ChatMessage]], ChatResponse]


@dataclass
class SimulatedTurn:
    user: str
    response: ChatResponse
    messages_after: List[ChatMessage] = field(default_factory=list)


class ConversationSimulator:
    """Drive scripted user turns through a chat handler."""

    def __init__(self, handler: Handler | None = None, agent: ChatAgent | None = None) -> None:
        if handler is None:
            chat_agent = agent or ChatAgent()
            handler = chat_agent.handle
        self._handler = handler
        self.messages: List[ChatMessage] = []

    def reset(self) -> None:
        self.messages = []

    def send(self, user_text: str) -> SimulatedTurn:
        self.messages.append(ChatMessage(role="user", content=user_text))
        response = self._handler(self.messages)
        self.messages.append(ChatMessage(role="assistant", content=response.reply))
        turn = SimulatedTurn(
            user=user_text,
            response=response,
            messages_after=list(self.messages),
        )
        return turn

    def run_script(self, user_turns: Sequence[str]) -> List[SimulatedTurn]:
        return [self.send(text) for text in user_turns]
