from dataclasses import dataclass

import torch


@dataclass
class TokenBook:
    world_tokens: int
    clean_world_tokens: int
    agent_condition_tokens: int
    action_tokens: int
    consequence_tokens: int = 0

    @property
    def agent_tokens(self) -> int:
        return self.agent_condition_tokens + self.action_tokens + self.consequence_tokens

    @property
    def total(self) -> int:
        return self.world_tokens + self.agent_tokens

    @property
    def world(self) -> slice:
        return slice(0, self.world_tokens)

    @property
    def clean(self) -> slice:
        return slice(0, self.clean_world_tokens)

    @property
    def noisy_world(self) -> slice:
        return slice(self.clean_world_tokens, self.world_tokens)

    @property
    def agent(self) -> slice:
        return slice(self.world_tokens, self.total)

    @property
    def condition(self) -> slice:
        return slice(self.world_tokens, self.world_tokens + self.agent_condition_tokens)

    @property
    def action(self) -> slice:
        s = self.world_tokens + self.agent_condition_tokens
        return slice(s, s + self.action_tokens)

    @property
    def consequence(self) -> slice:
        s = self.world_tokens + self.agent_condition_tokens + self.action_tokens
        return slice(s, s + self.consequence_tokens)


def build_joint_mask(book: TokenBook, device=None) -> torch.Tensor:
    mask = torch.ones(book.total, book.total, dtype=torch.bool, device=device)
    mask[book.clean, :] = False
    mask[book.clean, book.clean] = True
    return mask
