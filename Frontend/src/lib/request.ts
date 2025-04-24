import axios from "axios"; 
import { Request } from "@/types/requestofficial"; 
import { QueryOptions } from "@/types/station2"; 

const getBaseUrl = () => process.env.NEXT_PUBLIC_BACKEND_URL;
export async function getAllRequests(
    token: string | null | undefined,
    options: QueryOptions = {}
): Promise<Request[]> {
    if (!token) {
        console.error("getAllRequests: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    try {
        const params = new URLSearchParams();
        if (options.limit !== undefined) { params.append("options.limit", String(options.limit)); }
        if (options.offset !== undefined) { params.append("options.offset", String(options.offset)); }
        if (options.sortBy) { params.append("options.sortBy", options.sortBy); }
        if (options.sortDesc !== undefined) { params.append("options.sortDesc", String(options.sortDesc)); }
        if (options.includeDeleted !== undefined) { params.append("options.includeDeleted", String(options.includeDeleted)); }
        const baseUrl = `${getBaseUrl()}/api/v1/requests`;
        const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;
        console.log(`GET Request URL: ${url}`); 
        const res = await axios.get(url, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        return res.data.requests;

    } catch (error) {
        console.error("Error fetching requests:", error);
         if (axios.isAxiosError(error)) {
             console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) {
                throw new Error("Unauthorized: Invalid or expired token.");
            }
         }
        throw new Error("Failed to get requests");
    }
}
export type CreateRequestData = Omit<Request, 'id' | 'createdAt' | 'updatedAt' | 'deletedAt'> & {};
export async function createRequest(
    token: string | null | undefined,
    data: CreateRequestData 
): Promise<Request> { 
    if (!token) {
        console.error("createRequest: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/requests`;
        console.log(`POST Request URL: ${url}`);
        console.log(`POST Request Data:`, data); 

        const res = await axios.post(url, data, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return res.data.request;

    } catch (error) {
        console.error(`Error creating request:`, error);
        if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) {
                throw new Error("Unauthorized: Invalid or expired token.");
            }
            if (error.response?.status === 400) {
                const message = error.response.data?.message || "Bad Request: Invalid data provided.";
                throw new Error(message);
            }
        }
        throw new Error("Failed to create request");
    }
}
export async function deleteRequest(
    token: string | null | undefined,
    id: string ,
    hardDelete: boolean = true
): Promise<number> { 
    if (!token) {
        console.error("deleteRequest: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    if (!id) {
        console.error("deleteRequest: Request ID is missing.");
        throw new Error("Request ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/requests/${id}?hardDelete=${hardDelete}`;
        console.log(`DELETE Request URL: ${url}`);

        const res = await axios.delete(url, {
            headers: {
                'Authorization': `Bearer ${token}`,
            }
        });
        return res.status

    } catch (error) {
        console.error(`Error deleting request ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to delete request");
    }
}

export async function getRequestById(
    token: string | null | undefined,
    id: string 
): Promise<Request> {
    if (!token) {
        console.error("getRequestById: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
     if (!id) {
        console.error("getRequestById: Request ID is missing.");
        throw new Error("Request ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/requests/${id}`;
        console.log(`GET Request URL: ${url}`);

        const res = await axios.get(url, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        return res.data.request; 
    } catch (error) {
        console.error(`Error fetching request ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to get request by ID");
    }
}

export async function updateRequest(
    token: string | null | undefined,
    id: string, 
    data: Partial<Omit<Request, 'id' | 'createdAt' | 'updatedAt' | 'deletedAt'>> 
): Promise<Request> { 
    if (!token) {
        console.error("updateRequest: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
     if (!id) {
        console.error("updateRequest: Request ID is missing.");
        throw new Error("Request ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/requests/${id}`;
        console.log(`PATCH Request URL: ${url}`);

        const res = await axios.patch(url, data, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return res.data.request; 

    } catch (error) {
        console.error(`Error updating request ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to update request");
    }
}

export async function findRequestsByUserParticipant(
    token: string | null | undefined,
    userId: string, 
    options: QueryOptions = {}
): Promise<Request[]> {
    if (!token) {
        console.error("findRequestsByUserParticipant: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    if (!userId) {
        console.error("findRequestsByUserParticipant: User ID is missing.");
        throw new Error("User ID is required.");
    }
    try {
        const params = new URLSearchParams();
        if (options.limit !== undefined) { params.append("options.limit", String(options.limit)); }
        if (options.offset !== undefined) { params.append("options.offset", String(options.offset)); }
        if (options.sortBy) { params.append("options.sortBy", options.sortBy); }
        if (options.sortDesc !== undefined) { params.append("options.sortDesc", String(options.sortDesc)); }
        if (options.includeDeleted !== undefined) { params.append("options.includeDeleted", String(options.includeDeleted)); }
        const baseUrl = `${getBaseUrl()}/api/v1/users/${userId}/requests`;
        const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;

        console.log(`GET Request URL: ${url}`);

        const res = await axios.get(url, { // Expect array or object containing array
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
         return res.data.requests;
    } catch (error) {
        console.error(`Error fetching requests for user ${userId}:`, error);
        if (axios.isAxiosError(error)) {
             console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("User or requests not found."); }
        }
        throw new Error("Failed to find requests by user participant");
    }
}
