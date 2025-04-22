import { DataPoint, Station,QueryOptions, ApiRequestDataPointsByStationId, ApiRequestPayload } from "@/types/station2";
import axios from "axios";

  
export async function getStations(options: QueryOptions = {}): Promise<Station[]> {
    try {
      const params = new URLSearchParams();
      if (options.limit !== undefined) {params.append("options.limit", options.limit.toString());}
      if (options.offset !== undefined) {params.append("options.offset", options.offset.toString());}
      if (options.sortBy) { params.append("options.sortBy", `"${options.sortBy}"`);} 
      if (options.sortDesc !== undefined) {params.append("options.sortDesc", options.sortDesc.toString());}
      if (options.includeDeleted !== undefined) {params.append("options.includeDeleted", options.includeDeleted.toString());}
      // if (options.filters) {params.append("options.filters", options.filters.toString()); }
      const baseUrl = `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/water-quality/stations`;
      const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;
      const res = await axios.get(url);
      return res.data.stations;
    } catch (error) {
      console.error(error);
      throw new Error("Failed to get stations");
    }
  }

export async function getDataPointsOfStationById(stationId:string ,options: QueryOptions = {}): Promise<DataPoint[]> {
    try {
      const params = new URLSearchParams();
      if (options.limit !== undefined) {params.append("options.limit", options.limit.toString());}
      if (options.offset !== undefined) {params.append("options.offset", options.offset.toString());}
      if (options.sortBy) { params.append("options.sortBy", `"${options.sortBy}"`);} 
      if (options.sortDesc !== undefined) {params.append("options.sortDesc", options.sortDesc.toString());}
      if (options.includeDeleted !== undefined) {params.append("options.includeDeleted", options.includeDeleted.toString());}
      // if (options.filters) {params.append("options.filters", options.filters.toString()); }
      const baseUrl = `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/water-quality/stations/${stationId}/data-points`;
      const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;
      const res = await axios.get(url);
      return res.data.dataPoints;
    } catch (error) {
      console.error(error);
      throw new Error("Failed to get data point");
    }
  }

export async function getAllDataPoints(options: ApiRequestPayload = {}): Promise<DataPoint[]> {
    try {
      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/water-quality/data-points/list-all`,
        options,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      return res.data.dataPoints;
    } catch (error) {
      console.error(error);
      throw new Error("Failed to get data points");
    }
  }

export async function getAllDataPointsByStationID(options: ApiRequestDataPointsByStationId = {}): Promise<DataPoint[]> {
    try {
      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/water-quality/data-points/list-by-station`,
        options,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      return res.data.dataPoints;
    } catch (error) {
      console.error(error);
      throw new Error("Failed to get data points");
    }
  }