from __future__ import annotations

import re

from .base_parser import BaseParser


class RankingParser(BaseParser):
    def parse(self, completion: str, max_num: int = 4) -> list[int]:
        content = completion or ""
        pattern = r"\[([1234567]),\s*([1234567])\]"
        matches = re.findall(pattern, content)

        try:
            match = matches[-1]
            tops = [int(match[0]) - 1, int(match[1]) - 1]

            def clip(x: int) -> int:
                if x < 0:
                    return 0
                if x > max_num - 1:
                    return max_num - 1
                return x

            return [clip(x) for x in tops]
        except Exception:
            return list(range(min(2, max_num)))
