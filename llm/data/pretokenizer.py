import regex
from llm.data.special_tokens import SpecialTokens

from tokenizers import NormalizedString, PreTokenizedString
import regex

# TODO: handle special tokens in a better way
def split(text: str) -> list[str]:
    if text == '':
        return []

    safe_control_tokens = [regex.escape(c) for c in SpecialTokens().list()]
    # reg = r'(' + r'|'.join(safe_control_tokens) + r'|\d+|\s+|\p{L}+|[^\d\p{L}\s' + r''.join([f'[{i}]' for i in safe_control_tokens]) + r']+)'
    reg = (
        r'(' + 
        r'|'.join(safe_control_tokens) +  # Match control tokens first
        r'|\d+(?:\.\d+)?' +  # Match numbers, including decimals
        # r'|\d' +  # Match individual digits separately
        r'|\s+' +  # Match spaces
        r'|\p{L}+' +  # Match words (letters)
        r'|[^\d\p{L}\s' + ''.join([f'[{i}]' for i in safe_control_tokens]) + r']+)'  # Match any remaining special characters
    )
    words = regex.split(reg, text, flags = regex.UNICODE, concurrent = False)
    words = list(filter(None, words))

    temp = []
    i = 0
    while i < len(words) - 1:
        if words[i] == ' ' and words[i + 1] not in SpecialTokens().list():
            temp.append(' ' + words[i + 1])
            i += 2
            continue
        if words[i].endswith(' ') and words[i + 1] not in SpecialTokens().list():
            temp.extend([words[i][:-1], ' ' + words[i + 1]])
            i += 2
            continue

        temp.append(words[i])
        i += 1
    if i == len(words) - 1:
        temp.append(words[-1])

    words = temp
    words = list(filter(None, words))
        
    return words


class PreTokenizer:
    def __init__(self, strip: bool = False) -> None:
        self.strip = strip
        
    def split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        words = split(str(normalized_string))
        if self.strip:
            words = [word.lstrip() for word in words if word.strip()]
        words = [NormalizedString(word) for word in words]

        return words
        
    def pre_tokenize(self, pretok: PreTokenizedString) -> None:
        pretok.split(self.split)