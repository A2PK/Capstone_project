export interface Thresholds {
    configs: ElementRange[]
}
export interface ElementRange {
    createdAt: string,
    deletedAt: string,
    elementName: string, //pH, COD, N-NO2, 
    id: string,
    maxValue: number | null,
    minValue: number | null,
    updatedAt: string
}

export interface CreateElementRangeDto {
    elementName: string;
    maxValue: number | null;
    minValue: number | null;
  }
  
  export interface CreateThresholdsDto {
    configs: CreateElementRangeDto[];
  }
  
  export interface DeleteThresholdsDto {
    hardDelete: boolean;
    ids: string[];
  }