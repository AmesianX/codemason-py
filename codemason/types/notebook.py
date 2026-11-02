"""
Notebook Types for CodeMason
Equivalent to src/types/notebook.ts
"""

from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass


# Cell types
NotebookCellType = Literal['code', 'markdown']


@dataclass
class NotebookOutputImage:
    """Image output from a notebook cell"""
    image_data: str
    media_type: Literal['image/png', 'image/jpeg']


@dataclass
class NotebookCellSourceOutput:
    """Processed output from a notebook cell"""
    output_type: str
    text: str = ''
    image: Optional[NotebookOutputImage] = None


@dataclass
class NotebookCellSource:
    """Processed notebook cell data"""
    cell: int
    cellType: NotebookCellType
    source: str
    language: str
    execution_count: Optional[int] = None
    outputs: Optional[List[NotebookCellSourceOutput]] = None


# Raw notebook structures (from JSON)

@dataclass
class NotebookCellOutput:
    """Raw cell output from notebook JSON"""
    output_type: str
    text: Optional[Union[str, List[str]]] = None
    data: Optional[Dict[str, Any]] = None
    ename: Optional[str] = None
    evalue: Optional[str] = None
    traceback: Optional[List[str]] = None


@dataclass
class NotebookCell:
    """Raw notebook cell from JSON"""
    cell_type: NotebookCellType
    source: Union[str, List[str]]
    metadata: Dict[str, Any]
    outputs: Optional[List[NotebookCellOutput]] = None
    execution_count: Optional[int] = None


@dataclass
class NotebookMetadata:
    """Notebook metadata"""
    language_info: Optional[Dict[str, Any]] = None
    kernelspec: Optional[Dict[str, Any]] = None


@dataclass
class NotebookContent:
    """Full notebook content"""
    cells: List[NotebookCell]
    metadata: NotebookMetadata
    nbformat: int
    nbformat_minor: int


def is_notebook_cell_type(value: Optional[str]) -> bool:
    """Check if value is a valid notebook cell type"""
    return value in ('code', 'markdown')
