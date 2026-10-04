"""Bounded validation counters and examples."""

from dataclasses import dataclass, field


@dataclass
class ValidationReport:
    max_examples: int = 5
    rows_seen: int = 0
    failures: int = 0
    examples: list[str] = field(default_factory=list)

    def add_failure(self, message: str) -> None:
        self.failures += 1
        if len(self.examples) < self.max_examples:
            self.examples.append(message[:300])
