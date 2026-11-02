"""
Message type definitions for CodeMason
Equivalent to src/types/messages.ts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union, Any
from enum import Enum


# === Cache Control ===

@dataclass
class CacheControl:
    type: Literal['ephemeral'] = 'ephemeral'


# === Block Param Types (for sending to API) ===

@dataclass
class TextBlockParam:
    type: Literal['text'] = 'text'
    text: str = ''
    cache_control: Optional[CacheControl] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {'type': self.type, 'text': self.text}
        if self.cache_control:
            result['cache_control'] = {'type': self.cache_control.type}
        return result


@dataclass
class ImageBlockParam:
    type: Literal['image'] = 'image'
    source: Dict[str, Any] = field(default_factory=dict)
    cache_control: Optional[CacheControl] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {'type': self.type, 'source': self.source}
        if self.cache_control:
            result['cache_control'] = {'type': self.cache_control.type}
        return result


@dataclass
class ToolUseBlockParam:
    type: Literal['tool_use'] = 'tool_use'
    id: str = ''
    name: str = ''
    input: Dict[str, Any] = field(default_factory=dict)
    cache_control: Optional[CacheControl] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'type': self.type,
            'id': self.id,
            'name': self.name,
            'input': self.input
        }
        if self.cache_control:
            result['cache_control'] = {'type': self.cache_control.type}
        return result


@dataclass
class ToolResultBlockParam:
    type: Literal['tool_result'] = 'tool_result'
    tool_use_id: str = ''
    content: Union[str, List[Union['TextBlockParam', 'ImageBlockParam']]] = ''
    is_error: bool = False
    cache_control: Optional[CacheControl] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'type': self.type,
            'tool_use_id': self.tool_use_id,
            'content': self.content if isinstance(self.content, str) else [c.to_dict() for c in self.content]
        }
        if self.is_error:
            result['is_error'] = True
        if self.cache_control:
            result['cache_control'] = {'type': self.cache_control.type}
        return result


@dataclass
class DocumentBlockParam:
    type: Literal['document'] = 'document'
    source: Dict[str, Any] = field(default_factory=dict)
    cache_control: Optional[CacheControl] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {'type': self.type, 'source': self.source}
        if self.cache_control:
            result['cache_control'] = {'type': self.cache_control.type}
        return result


@dataclass
class ThinkingBlockParam:
    type: Literal['thinking'] = 'thinking'
    thinking: str = ''
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {'type': self.type, 'thinking': self.thinking}
        if self.signature:
            result['signature'] = self.signature
        return result


ContentBlockParam = Union[
    TextBlockParam,
    ImageBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
    DocumentBlockParam,
    ThinkingBlockParam
]


# === Block Types (received from API) ===

@dataclass
class TextBlock:
    type: Literal['text'] = 'text'
    text: str = ''

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextBlock':
        return cls(type='text', text=data.get('text', ''))


@dataclass
class ToolUseBlock:
    type: Literal['tool_use'] = 'tool_use'
    id: str = ''
    name: str = ''
    input: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolUseBlock':
        return cls(
            type='tool_use',
            id=data.get('id', ''),
            name=data.get('name', ''),
            input=data.get('input', {})
        )


@dataclass
class ThinkingBlock:
    type: Literal['thinking'] = 'thinking'
    thinking: str = ''

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThinkingBlock':
        return cls(type='thinking', thinking=data.get('thinking', ''))


ContentBlock = Union[TextBlock, ToolUseBlock, ThinkingBlock]


def parse_content_block(data: Dict[str, Any]) -> ContentBlock:
    """Parse a content block from API response"""
    block_type = data.get('type')
    if block_type == 'text':
        return TextBlock.from_dict(data)
    elif block_type == 'tool_use':
        return ToolUseBlock.from_dict(data)
    elif block_type == 'thinking':
        return ThinkingBlock.from_dict(data)
    else:
        # Default to text block
        return TextBlock(text=str(data))


# === Message Types ===

@dataclass
class MessageParam:
    role: Literal['user', 'assistant', 'system']
    content: Union[str, List[ContentBlockParam]]

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.content, str):
            return {'role': self.role, 'content': self.content}
        return {
            'role': self.role,
            'content': [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.content]
        }


@dataclass
class BetaUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BetaUsage':
        return cls(
            input_tokens=data.get('input_tokens', 0),
            output_tokens=data.get('output_tokens', 0),
            cache_read_input_tokens=data.get('cache_read_input_tokens', 0),
            cache_creation_input_tokens=data.get('cache_creation_input_tokens', 0)
        )


@dataclass
class Message:
    """Message type (API response)"""
    id: str
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    content: List[ContentBlock] = field(default_factory=list)
    model: str = ''
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: BetaUsage = field(default_factory=BetaUsage)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        content = [parse_content_block(c) for c in data.get('content', [])]
        usage = BetaUsage.from_dict(data.get('usage', {}))
        return cls(
            id=data.get('id', ''),
            type='message',
            role='assistant',
            content=content,
            model=data.get('model', ''),
            stop_reason=data.get('stop_reason'),
            stop_sequence=data.get('stop_sequence'),
            usage=usage
        )


# Alias for backward compatibility
APIMessage = Message


# === Error Classes ===

class APIError(Exception):
    """Base API Error"""
    def __init__(self, message: str, status: Optional[int] = None, headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.status = status
        self.headers = headers or {}


class APIConnectionError(APIError):
    """API Connection Error"""
    def __init__(self, message: str):
        super().__init__(message)
