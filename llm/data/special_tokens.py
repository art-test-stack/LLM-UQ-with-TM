from dataclasses import dataclass
from typing import List

@dataclass
class SpecialTokens:
    pad: str = "<|padding|>"
    unknown: str = "<|unknown|>"
    tab: str = "<|tab|>"
    new_line: str = "<|new_line|>"
    start_of_text: str = "<|begin_of_text|>"
    end_of_text: str = "<|end_of_text|>"
    start_of_header: str = "<|start_of_header|>"
    end_of_header: str = "<|end_of_header|>"
    eot_id: str = "<|eot_id|>"
    start_of_answer: str = "<|startofanswer|>"
    
    def list(self) -> List[str]:
        """
        Returns a list of all special tokens.
        """
        return list(self.__dict__.values())
