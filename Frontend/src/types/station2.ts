export interface Station {
    id: string;
    createdAt: string;
    updatedAt: string;
    name: string;
    latitude: number;
    longitude: number;
    country: string;
    location: string;
  }

export interface PaginParam {
    totalItems: string,
    limit: number,
    offset: number
}

export interface DataPoint {
    id: string;
    createdAt: string;
    updatedAt: string;
    monitoringTime: string;
    wqi: number;
    stationId: string;
    source: string;
    observationType: string;
    dataSourceSchemaId: string;
    features: Indicator[];
}

export interface Indicator {
    name: string;
    textualValue?: string;
    value?: number;
    purpose: string;
    source: string;
}

export interface QueryOptions {
    limit?: number;
    offset?: number;
    sortBy?: string;
    sortDesc?: boolean;
    // filters?: string;
    includeDeleted?: boolean;
  }
  export  interface FilterOptions {
    observation_type?: string;
    source?: string;
    monitoring_time?:string
  }

  export interface RequestOptions {
    filters?: FilterOptions;
    includeDeleted?: boolean;
    limit?: number;
    offset?: number;
    sortBy?: string;
    sortDesc?: boolean;
  }
  export interface ApiRequestPayload {
    options?: RequestOptions;
  }
  export interface ApiRequestDataPointsByStationId {
    options?: RequestOptions;
    stationId?:string;
  }