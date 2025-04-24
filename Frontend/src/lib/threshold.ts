import { QueryOptions } from "@/types/station2"; 
import { ElementRange, Thresholds, CreateThresholdsDto, DeleteThresholdsDto, CreateElementRangeDto } from "@/types/threshold";
import axios from "axios";

const getBaseUrl = () => process.env.NEXT_PUBLIC_BACKEND_URL;
const getThresholdsEndpoint = () => `${getBaseUrl()}/api/v1/water-quality/threshold-configs`;

/**
 * Fetches all threshold configurations with optional query parameters.
 * @param options Optional query parameters (limit, offset, sortBy, etc.)
 * @returns Promise resolving to an array of ElementRange objects.
 */
export async function getAllThresholdConfigs(options: QueryOptions = {}): Promise<ElementRange[]> {
  try {
    const params = new URLSearchParams();
    if (options.limit !== undefined) { params.append("options.limit", options.limit.toString()); }
    if (options.offset !== undefined) { params.append("options.offset", options.offset.toString()); }
    if (options.sortBy) { params.append("options.sortBy", options.sortBy); } 
    if (options.sortDesc !== undefined) { params.append("options.sortDesc", options.sortDesc.toString()); }
    if (options.includeDeleted !== undefined) { params.append("options.includeDeleted", options.includeDeleted.toString()); }

    const baseUrl = getThresholdsEndpoint();
    const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;

    console.log(`GET Request URL: ${url}`); 

    const res = await axios.get<{ configs: ElementRange[] }>(url); 

    if (!res.data || !Array.isArray(res.data.configs)) {
        console.error("Invalid response structure from GET /threshold-configs:", res.data);
        throw new Error("Invalid response structure received from server.");
    }
    return res.data.configs;
  } catch (error) {
    console.error("Error fetching threshold configs:", error);
    if (axios.isAxiosError(error) && error.response) {
        throw new Error(`Failed to get threshold configs: ${error.response.status} ${error.response.statusText}`);
    }
    throw new Error("Failed to get threshold configs due to a network or server error.");
  }
}

/**
 * Creates new threshold configurations.
 * @param payload The threshold configurations to create.
 * @returns Promise resolving to the created Thresholds object containing full ElementRange details.
 */
export async function createThresholdConfigs(payload: CreateThresholdsDto): Promise<Thresholds> {
  const url = getThresholdsEndpoint();
  console.log(`POST Request URL: ${url}`);
  console.log(`POST Request Body:`, payload);
  try {
    const res = await axios.post<Thresholds>(url, payload);
    if (!res.data || !Array.isArray(res.data.configs)) {
        console.error("Invalid response structure from POST /threshold-configs:", res.data);
        throw new Error("Invalid response structure received after creation.");
    }
    return res.data;
  } catch (error) {
    console.error("Error creating threshold configs:", error);
    if (axios.isAxiosError(error) && error.response) {
        // Log backend error details if available
        console.error("Backend Error:", error.response.data);
        throw new Error(`Failed to create threshold configs: ${error.response.status} ${error.response.statusText}`);
    }
    throw new Error("Failed to create threshold configs due to a network or server error.");
  }
}

export async function updateThresholdConfigs(payload: Thresholds): Promise<Thresholds> {
  const url = getThresholdsEndpoint(); 
   console.log(`PUT Request URL: ${url}`);
   console.log(`PUT Request Body:`, payload);
  try {
    if (!payload.configs.every(config => config.id)) {
        throw new Error("Cannot update thresholds: One or more config items are missing an ID.");
    }

    const res = await axios.put<Thresholds>(url, payload);

     if (!res.data || !Array.isArray(res.data.configs)) {
        console.error("Invalid response structure from PUT /threshold-configs:", res.data);
        throw new Error("Invalid response structure received after update.");
    }

    return res.data;
  } catch (error) {
    console.error("Error updating threshold configs:", error);
     if (axios.isAxiosError(error) && error.response) {
        console.error("Backend Error:", error.response.data);
        throw new Error(`Failed to update threshold configs: ${error.response.status} ${error.response.statusText}`);
    }
    throw new Error("Failed to update threshold configs due to a network or server error.");
  }
}

/**
 * Deletes threshold configurations (soft or hard delete).
 * Uses a POST request to a /delete sub-endpoint as specified.
 * @param payload Object containing the IDs to delete and the hardDelete flag.
 * @returns Promise resolving when the deletion is successful. Throws an error on failure.
 */
export async function deleteThresholdConfigs(payload: DeleteThresholdsDto): Promise<void> {
  const url = `${getThresholdsEndpoint()}/delete`;
  console.log(`POST Request URL (for delete): ${url}`);
  console.log(`POST Request Body (for delete):`, payload);
  try {
    // Expecting a 2xx status code for success, maybe 204 No Content or 200 OK
    await axios.post(url, payload);
    // No return value needed if backend just sends status
  } catch (error) {
    console.error("Error deleting threshold configs:", error);
    if (axios.isAxiosError(error) && error.response) {
        console.error("Backend Error:", error.response.data);
        throw new Error(`Failed to delete threshold configs: ${error.response.status} ${error.response.statusText}`);
    }
    throw new Error("Failed to delete threshold configs due to a network or server error.");
  }
}