from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from proto.core import common_pb2 as _common_pb2
from google.api import annotations_pb2 as _annotations_pb2
from protoc_gen_openapiv2.options import annotations_pb2 as _annotations_pb2_1
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObservationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OBSERVATION_TYPE_UNSPECIFIED: _ClassVar[ObservationType]
    OBSERVATION_TYPE_ACTUAL: _ClassVar[ObservationType]
    OBSERVATION_TYPE_INTERPOLATION: _ClassVar[ObservationType]
    OBSERVATION_TYPE_PREDICTED: _ClassVar[ObservationType]
    OBSERVATION_TYPE_REALTIME_MONITORING: _ClassVar[ObservationType]

class IndicatorPurpose(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INDICATOR_PURPOSE_UNSPECIFIED: _ClassVar[IndicatorPurpose]
    INDICATOR_PURPOSE_PREDICTION: _ClassVar[IndicatorPurpose]
    INDICATOR_PURPOSE_DISPLAY: _ClassVar[IndicatorPurpose]
    INDICATOR_PURPOSE_ANALYSIS: _ClassVar[IndicatorPurpose]
OBSERVATION_TYPE_UNSPECIFIED: ObservationType
OBSERVATION_TYPE_ACTUAL: ObservationType
OBSERVATION_TYPE_INTERPOLATION: ObservationType
OBSERVATION_TYPE_PREDICTED: ObservationType
OBSERVATION_TYPE_REALTIME_MONITORING: ObservationType
INDICATOR_PURPOSE_UNSPECIFIED: IndicatorPurpose
INDICATOR_PURPOSE_PREDICTION: IndicatorPurpose
INDICATOR_PURPOSE_DISPLAY: IndicatorPurpose
INDICATOR_PURPOSE_ANALYSIS: IndicatorPurpose

class Station(_message.Message):
    __slots__ = ("id", "created_at", "updated_at", "deleted_at", "name", "latitude", "longitude", "country", "location")
    ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    DELETED_AT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    LATITUDE_FIELD_NUMBER: _ClassVar[int]
    LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    COUNTRY_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    id: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    deleted_at: _timestamp_pb2.Timestamp
    name: str
    latitude: float
    longitude: float
    country: str
    location: str
    def __init__(self, id: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., deleted_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., name: _Optional[str] = ..., latitude: _Optional[float] = ..., longitude: _Optional[float] = ..., country: _Optional[str] = ..., location: _Optional[str] = ...) -> None: ...

class StationInput(_message.Message):
    __slots__ = ("name", "latitude", "longitude", "country", "location")
    NAME_FIELD_NUMBER: _ClassVar[int]
    LATITUDE_FIELD_NUMBER: _ClassVar[int]
    LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    COUNTRY_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    name: str
    latitude: float
    longitude: float
    country: str
    location: str
    def __init__(self, name: _Optional[str] = ..., latitude: _Optional[float] = ..., longitude: _Optional[float] = ..., country: _Optional[str] = ..., location: _Optional[str] = ...) -> None: ...

class CreateStationsRequest(_message.Message):
    __slots__ = ("stations",)
    STATIONS_FIELD_NUMBER: _ClassVar[int]
    stations: _containers.RepeatedCompositeFieldContainer[StationInput]
    def __init__(self, stations: _Optional[_Iterable[_Union[StationInput, _Mapping]]] = ...) -> None: ...

class CreateStationsResponse(_message.Message):
    __slots__ = ("stations",)
    STATIONS_FIELD_NUMBER: _ClassVar[int]
    stations: _containers.RepeatedCompositeFieldContainer[Station]
    def __init__(self, stations: _Optional[_Iterable[_Union[Station, _Mapping]]] = ...) -> None: ...

class UpdateStationsRequest(_message.Message):
    __slots__ = ("stations",)
    STATIONS_FIELD_NUMBER: _ClassVar[int]
    stations: _containers.RepeatedCompositeFieldContainer[Station]
    def __init__(self, stations: _Optional[_Iterable[_Union[Station, _Mapping]]] = ...) -> None: ...

class UpdateStationsResponse(_message.Message):
    __slots__ = ("stations",)
    STATIONS_FIELD_NUMBER: _ClassVar[int]
    stations: _containers.RepeatedCompositeFieldContainer[Station]
    def __init__(self, stations: _Optional[_Iterable[_Union[Station, _Mapping]]] = ...) -> None: ...

class DeleteRequest(_message.Message):
    __slots__ = ("ids", "hard_delete")
    IDS_FIELD_NUMBER: _ClassVar[int]
    HARD_DELETE_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    hard_delete: bool
    def __init__(self, ids: _Optional[_Iterable[str]] = ..., hard_delete: bool = ...) -> None: ...

class DeleteResponse(_message.Message):
    __slots__ = ("affected_count",)
    AFFECTED_COUNT_FIELD_NUMBER: _ClassVar[int]
    affected_count: int
    def __init__(self, affected_count: _Optional[int] = ...) -> None: ...

class ListStationsRequest(_message.Message):
    __slots__ = ("options",)
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    options: _common_pb2.FilterOptions
    def __init__(self, options: _Optional[_Union[_common_pb2.FilterOptions, _Mapping]] = ...) -> None: ...

class ListStationsResponse(_message.Message):
    __slots__ = ("stations", "pagination")
    STATIONS_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_FIELD_NUMBER: _ClassVar[int]
    stations: _containers.RepeatedCompositeFieldContainer[Station]
    pagination: _common_pb2.PaginationInfo
    def __init__(self, stations: _Optional[_Iterable[_Union[Station, _Mapping]]] = ..., pagination: _Optional[_Union[_common_pb2.PaginationInfo, _Mapping]] = ...) -> None: ...

class DataPointFeature(_message.Message):
    __slots__ = ("name", "value", "textual_value", "purpose", "source")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TEXTUAL_VALUE_FIELD_NUMBER: _ClassVar[int]
    PURPOSE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: float
    textual_value: str
    purpose: IndicatorPurpose
    source: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[float] = ..., textual_value: _Optional[str] = ..., purpose: _Optional[_Union[IndicatorPurpose, str]] = ..., source: _Optional[str] = ...) -> None: ...

class DataPointFeatureInput(_message.Message):
    __slots__ = ("name", "value", "textual_value", "purpose", "source")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TEXTUAL_VALUE_FIELD_NUMBER: _ClassVar[int]
    PURPOSE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: float
    textual_value: str
    purpose: IndicatorPurpose
    source: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[float] = ..., textual_value: _Optional[str] = ..., purpose: _Optional[_Union[IndicatorPurpose, str]] = ..., source: _Optional[str] = ...) -> None: ...

class DataPoint(_message.Message):
    __slots__ = ("id", "created_at", "updated_at", "deleted_at", "monitoring_time", "wqi", "station_id", "source", "observation_type", "data_source_schema_id", "features")
    ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    DELETED_AT_FIELD_NUMBER: _ClassVar[int]
    MONITORING_TIME_FIELD_NUMBER: _ClassVar[int]
    WQI_FIELD_NUMBER: _ClassVar[int]
    STATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_SOURCE_SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    id: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    deleted_at: _timestamp_pb2.Timestamp
    monitoring_time: _timestamp_pb2.Timestamp
    wqi: float
    station_id: str
    source: str
    observation_type: ObservationType
    data_source_schema_id: str
    features: _containers.RepeatedCompositeFieldContainer[DataPointFeature]
    def __init__(self, id: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., deleted_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., monitoring_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., wqi: _Optional[float] = ..., station_id: _Optional[str] = ..., source: _Optional[str] = ..., observation_type: _Optional[_Union[ObservationType, str]] = ..., data_source_schema_id: _Optional[str] = ..., features: _Optional[_Iterable[_Union[DataPointFeature, _Mapping]]] = ...) -> None: ...

class DataPointInput(_message.Message):
    __slots__ = ("monitoring_time", "wqi", "station_id", "source", "observation_type", "data_source_schema_id", "features")
    MONITORING_TIME_FIELD_NUMBER: _ClassVar[int]
    WQI_FIELD_NUMBER: _ClassVar[int]
    STATION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_SOURCE_SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    monitoring_time: _timestamp_pb2.Timestamp
    wqi: float
    station_id: str
    source: str
    observation_type: ObservationType
    data_source_schema_id: str
    features: _containers.RepeatedCompositeFieldContainer[DataPointFeatureInput]
    def __init__(self, monitoring_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., wqi: _Optional[float] = ..., station_id: _Optional[str] = ..., source: _Optional[str] = ..., observation_type: _Optional[_Union[ObservationType, str]] = ..., data_source_schema_id: _Optional[str] = ..., features: _Optional[_Iterable[_Union[DataPointFeatureInput, _Mapping]]] = ...) -> None: ...

class CreateDataPointsRequest(_message.Message):
    __slots__ = ("data_points",)
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    data_points: _containers.RepeatedCompositeFieldContainer[DataPointInput]
    def __init__(self, data_points: _Optional[_Iterable[_Union[DataPointInput, _Mapping]]] = ...) -> None: ...

class CreateDataPointsResponse(_message.Message):
    __slots__ = ("data_points",)
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    data_points: _containers.RepeatedCompositeFieldContainer[DataPoint]
    def __init__(self, data_points: _Optional[_Iterable[_Union[DataPoint, _Mapping]]] = ...) -> None: ...

class UpdateDataPointsRequest(_message.Message):
    __slots__ = ("data_points",)
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    data_points: _containers.RepeatedCompositeFieldContainer[DataPoint]
    def __init__(self, data_points: _Optional[_Iterable[_Union[DataPoint, _Mapping]]] = ...) -> None: ...

class UpdateDataPointsResponse(_message.Message):
    __slots__ = ("data_points",)
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    data_points: _containers.RepeatedCompositeFieldContainer[DataPoint]
    def __init__(self, data_points: _Optional[_Iterable[_Union[DataPoint, _Mapping]]] = ...) -> None: ...

class ListDataPointsByStationRequest(_message.Message):
    __slots__ = ("station_id", "options")
    STATION_ID_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    station_id: str
    options: _common_pb2.FilterOptions
    def __init__(self, station_id: _Optional[str] = ..., options: _Optional[_Union[_common_pb2.FilterOptions, _Mapping]] = ...) -> None: ...

class ListDataPointsByStationResponse(_message.Message):
    __slots__ = ("data_points", "pagination")
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_FIELD_NUMBER: _ClassVar[int]
    data_points: _containers.RepeatedCompositeFieldContainer[DataPoint]
    pagination: _common_pb2.PaginationInfo
    def __init__(self, data_points: _Optional[_Iterable[_Union[DataPoint, _Mapping]]] = ..., pagination: _Optional[_Union[_common_pb2.PaginationInfo, _Mapping]] = ...) -> None: ...

class ListAllDataPointsRequest(_message.Message):
    __slots__ = ("options",)
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    options: _common_pb2.FilterOptions
    def __init__(self, options: _Optional[_Union[_common_pb2.FilterOptions, _Mapping]] = ...) -> None: ...

class ListAllDataPointsResponse(_message.Message):
    __slots__ = ("data_points", "pagination")
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_FIELD_NUMBER: _ClassVar[int]
    data_points: _containers.RepeatedCompositeFieldContainer[DataPoint]
    pagination: _common_pb2.PaginationInfo
    def __init__(self, data_points: _Optional[_Iterable[_Union[DataPoint, _Mapping]]] = ..., pagination: _Optional[_Union[_common_pb2.PaginationInfo, _Mapping]] = ...) -> None: ...

class UploadRequest(_message.Message):
    __slots__ = ("filename", "file_type", "data_chunk")
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    FILE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_CHUNK_FIELD_NUMBER: _ClassVar[int]
    filename: str
    file_type: str
    data_chunk: bytes
    def __init__(self, filename: _Optional[str] = ..., file_type: _Optional[str] = ..., data_chunk: _Optional[bytes] = ...) -> None: ...

class UploadDataResponse(_message.Message):
    __slots__ = ("message", "records_processed", "records_failed", "data_source_schema_id", "errors")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RECORDS_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    RECORDS_FAILED_FIELD_NUMBER: _ClassVar[int]
    DATA_SOURCE_SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    message: str
    records_processed: int
    records_failed: int
    data_source_schema_id: str
    errors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, message: _Optional[str] = ..., records_processed: _Optional[int] = ..., records_failed: _Optional[int] = ..., data_source_schema_id: _Optional[str] = ..., errors: _Optional[_Iterable[str]] = ...) -> None: ...

class DataSourceSchema(_message.Message):
    __slots__ = ("id", "created_at", "updated_at", "deleted_at", "name", "source_identifier", "source_type", "description", "schema_definition")
    ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    DELETED_AT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    id: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    deleted_at: _timestamp_pb2.Timestamp
    name: str
    source_identifier: str
    source_type: str
    description: str
    schema_definition: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., deleted_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., name: _Optional[str] = ..., source_identifier: _Optional[str] = ..., source_type: _Optional[str] = ..., description: _Optional[str] = ..., schema_definition: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class DataSourceSchemaInput(_message.Message):
    __slots__ = ("name", "source_identifier", "source_type", "description", "schema_definition")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    name: str
    source_identifier: str
    source_type: str
    description: str
    schema_definition: _struct_pb2.Struct
    def __init__(self, name: _Optional[str] = ..., source_identifier: _Optional[str] = ..., source_type: _Optional[str] = ..., description: _Optional[str] = ..., schema_definition: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class CreateDataSourceSchemaRequest(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: DataSourceSchemaInput
    def __init__(self, schema: _Optional[_Union[DataSourceSchemaInput, _Mapping]] = ...) -> None: ...

class CreateDataSourceSchemaResponse(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: DataSourceSchema
    def __init__(self, schema: _Optional[_Union[DataSourceSchema, _Mapping]] = ...) -> None: ...

class UpdateDataSourceSchemaRequest(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: DataSourceSchema
    def __init__(self, schema: _Optional[_Union[DataSourceSchema, _Mapping]] = ...) -> None: ...

class UpdateDataSourceSchemaResponse(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: DataSourceSchema
    def __init__(self, schema: _Optional[_Union[DataSourceSchema, _Mapping]] = ...) -> None: ...

class GetDataSourceSchemaRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class GetDataSourceSchemaResponse(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: DataSourceSchema
    def __init__(self, schema: _Optional[_Union[DataSourceSchema, _Mapping]] = ...) -> None: ...

class ListDataSourceSchemasRequest(_message.Message):
    __slots__ = ("options",)
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    options: _common_pb2.FilterOptions
    def __init__(self, options: _Optional[_Union[_common_pb2.FilterOptions, _Mapping]] = ...) -> None: ...

class ListDataSourceSchemasResponse(_message.Message):
    __slots__ = ("schemas", "pagination")
    SCHEMAS_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_FIELD_NUMBER: _ClassVar[int]
    schemas: _containers.RepeatedCompositeFieldContainer[DataSourceSchema]
    pagination: _common_pb2.PaginationInfo
    def __init__(self, schemas: _Optional[_Iterable[_Union[DataSourceSchema, _Mapping]]] = ..., pagination: _Optional[_Union[_common_pb2.PaginationInfo, _Mapping]] = ...) -> None: ...
