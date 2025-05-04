from google.protobuf import struct_pb2 as _struct_pb2
from protoc_gen_openapiv2.options import annotations_pb2 as _annotations_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FilterOptions(_message.Message):
    __slots__ = ("limit", "offset", "sort_by", "sort_desc", "filters", "include_deleted")
    class FiltersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_FIELD_NUMBER: _ClassVar[int]
    SORT_DESC_FIELD_NUMBER: _ClassVar[int]
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_DELETED_FIELD_NUMBER: _ClassVar[int]
    limit: int
    offset: int
    sort_by: str
    sort_desc: bool
    filters: _containers.MessageMap[str, _struct_pb2.Value]
    include_deleted: bool
    def __init__(self, limit: _Optional[int] = ..., offset: _Optional[int] = ..., sort_by: _Optional[str] = ..., sort_desc: bool = ..., filters: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., include_deleted: bool = ...) -> None: ...

class PaginationInfo(_message.Message):
    __slots__ = ("total_items", "limit", "offset")
    TOTAL_ITEMS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    total_items: int
    limit: int
    offset: int
    def __init__(self, total_items: _Optional[int] = ..., limit: _Optional[int] = ..., offset: _Optional[int] = ...) -> None: ...
