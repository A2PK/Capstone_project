# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: proto/core/common.proto
# Protobuf Python Version: 6.30.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    30,
    0,
    '',
    'proto/core/common.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2
from protoc_gen_openapiv2.options import annotations_pb2 as protoc__gen__openapiv2_dot_options_dot_annotations__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17proto/core/common.proto\x12\x04\x63ore\x1a\x1cgoogle/protobuf/struct.proto\x1a.protoc-gen-openapiv2/options/annotations.proto\"\xff\x06\n\rFilterOptions\x12L\n\x05limit\x18\x01 \x01(\x05\x42\x38\x92\x41\x35\x32+Maximum number of items to return per page.:\x02\x35\x30J\x02\x35\x30H\x00\x88\x01\x01\x12s\n\x06offset\x18\x02 \x01(\x05\x42^\x92\x41[2SNumber of items to skip before starting to collect the result set (for pagination).:\x01\x30J\x01\x30H\x01\x88\x01\x01\x12v\n\x07sort_by\x18\x03 \x01(\tB`\x92\x41]2?Field name to sort the results by (e.g., \'created_at\', \'name\').:\x0c\"created_at\"J\x0c\"created_at\"H\x02\x88\x01\x01\x12Q\n\tsort_desc\x18\x04 \x01(\x08\x42\x39\x92\x41\x36\x32(Set to true to sort in descending order.:\x04trueJ\x04trueH\x03\x88\x01\x01\x12\xe6\x01\n\x07\x66ilters\x18\x05 \x03(\x0b\x32 .core.FilterOptions.FiltersEntryB\xb2\x01\x92\x41\xae\x01\x32\x8e\x01Key-value pairs for specific field filtering. Values should correspond to google.protobuf.Value structure (e.g., {\"email\": \"user@gmail.com\"}).J\x1b{\"email\": \"user@gmail.com\"}\x12l\n\x0finclude_deleted\x18\x08 \x01(\x08\x42N\x92\x41K2;Set to true to include soft-deleted records in the results.:\x05\x66\x61lseJ\x05\x66\x61lseH\x04\x88\x01\x01\x1a\x46\n\x0c\x46iltersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x16.google.protobuf.Value:\x02\x38\x01\x42\x08\n\x06_limitB\t\n\x07_offsetB\n\n\x08_sort_byB\x0c\n\n_sort_descB\x12\n\x10_include_deleted\"\xa0\x02\n\x0ePaginationInfo\x12\x63\n\x0btotal_items\x18\x01 \x01(\x03\x42N\x92\x41K2CTotal number of items matching the query criteria across all pages.J\x04\x31\x32\x33\x34\x12L\n\x05limit\x18\x02 \x01(\x05\x42=\x92\x41:24The limit (page size) used for the current response.J\x02\x35\x30\x12[\n\x06offset\x18\x03 \x01(\x05\x42K\x92\x41H2CThe offset (number of items skipped) used for the current response.J\x01\x30\x42\xba\x01Z+golang-microservices-boilerplate/proto/core\x92\x41\x89\x01\x12_\n\x17\x43ore Common Definitions\x12?Commonly used Protobuf messages for filtering, pagination, etc.2\x03\x31.0*\x02\x01\x02\x32\x10\x61pplication/json:\x10\x61pplication/jsonb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.core.common_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z+golang-microservices-boilerplate/proto/core\222A\211\001\022_\n\027Core Common Definitions\022?Commonly used Protobuf messages for filtering, pagination, etc.2\0031.0*\002\001\0022\020application/json:\020application/json'
  _globals['_FILTEROPTIONS_FILTERSENTRY']._loaded_options = None
  _globals['_FILTEROPTIONS_FILTERSENTRY']._serialized_options = b'8\001'
  _globals['_FILTEROPTIONS'].fields_by_name['limit']._loaded_options = None
  _globals['_FILTEROPTIONS'].fields_by_name['limit']._serialized_options = b'\222A52+Maximum number of items to return per page.:\00250J\00250'
  _globals['_FILTEROPTIONS'].fields_by_name['offset']._loaded_options = None
  _globals['_FILTEROPTIONS'].fields_by_name['offset']._serialized_options = b'\222A[2SNumber of items to skip before starting to collect the result set (for pagination).:\0010J\0010'
  _globals['_FILTEROPTIONS'].fields_by_name['sort_by']._loaded_options = None
  _globals['_FILTEROPTIONS'].fields_by_name['sort_by']._serialized_options = b'\222A]2?Field name to sort the results by (e.g., \'created_at\', \'name\').:\014\"created_at\"J\014\"created_at\"'
  _globals['_FILTEROPTIONS'].fields_by_name['sort_desc']._loaded_options = None
  _globals['_FILTEROPTIONS'].fields_by_name['sort_desc']._serialized_options = b'\222A62(Set to true to sort in descending order.:\004trueJ\004true'
  _globals['_FILTEROPTIONS'].fields_by_name['filters']._loaded_options = None
  _globals['_FILTEROPTIONS'].fields_by_name['filters']._serialized_options = b'\222A\256\0012\216\001Key-value pairs for specific field filtering. Values should correspond to google.protobuf.Value structure (e.g., {\"email\": \"user@gmail.com\"}).J\033{\"email\": \"user@gmail.com\"}'
  _globals['_FILTEROPTIONS'].fields_by_name['include_deleted']._loaded_options = None
  _globals['_FILTEROPTIONS'].fields_by_name['include_deleted']._serialized_options = b'\222AK2;Set to true to include soft-deleted records in the results.:\005falseJ\005false'
  _globals['_PAGINATIONINFO'].fields_by_name['total_items']._loaded_options = None
  _globals['_PAGINATIONINFO'].fields_by_name['total_items']._serialized_options = b'\222AK2CTotal number of items matching the query criteria across all pages.J\0041234'
  _globals['_PAGINATIONINFO'].fields_by_name['limit']._loaded_options = None
  _globals['_PAGINATIONINFO'].fields_by_name['limit']._serialized_options = b'\222A:24The limit (page size) used for the current response.J\00250'
  _globals['_PAGINATIONINFO'].fields_by_name['offset']._loaded_options = None
  _globals['_PAGINATIONINFO'].fields_by_name['offset']._serialized_options = b'\222AH2CThe offset (number of items skipped) used for the current response.J\0010'
  _globals['_FILTEROPTIONS']._serialized_start=112
  _globals['_FILTEROPTIONS']._serialized_end=1007
  _globals['_FILTEROPTIONS_FILTERSENTRY']._serialized_start=870
  _globals['_FILTEROPTIONS_FILTERSENTRY']._serialized_end=940
  _globals['_PAGINATIONINFO']._serialized_start=1010
  _globals['_PAGINATIONINFO']._serialized_end=1298
# @@protoc_insertion_point(module_scope)
