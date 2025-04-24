import axios from "axios"; 
import { FileInfo, postFileParam } from "@/types/file";
import { QueryOptions } from "@/types/station2";

const getBaseUrl = () => process.env.NEXT_PUBLIC_BACKEND_URL;
export async function uploadFile(
    token: string | null | undefined,
    data: postFileParam
): Promise<FileInfo> {
    // 1. Validate the authentication token
    if (!token) {
        console.error("uploadFile Error: Authentication token is missing.");
        throw new Error("Authentication token is required for file upload.");
    }

    const formData = new FormData();

    formData.append('type', data.type);
    formData.append('name', data.name); 
    formData.append('file', data.file, data.file.name); 
    const url = `${getBaseUrl()}/api/v1/files`;

    try {
        console.log(`Attempting to POST file to: ${url}`);
        const response = await axios.post< { file: FileInfo } >( 
            url,
            formData, 
            {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    // **IMPORTANT**: Do NOT set 'Content-Type': 'multipart/form-data' manually.
                    'Accept': 'application/json',
                }
            }
        );

        // 6. Log success and return the relevant data from the response
        console.log("File uploaded successfully. Server response:", response.data);
        // Assuming the server responds with an object containing a 'file' property
        if (response.data && response.data.file) {
            return response.data.file;
        } else {
            // Handle cases where the response structure is unexpected
            console.error("Unexpected server response structure:", response.data);
            throw new Error("Failed to upload file: Unexpected server response.");
        }

    } catch (error) {
        console.error(`Error uploading file to ${url}:`, error);

        if (axios.isAxiosError(error)) {
            console.error("Axios error details:", {
                message: error.message,
                code: error.code,
                status: error.response?.status,
                responseData: error.response?.data
            });
            if (error.response?.status === 401) {
                throw new Error("Unauthorized: Invalid or expired token. Please log in again.");
            }
            if (error.response?.status === 400) {
                const serverMessage = error.response.data?.message || error.response.data?.error || "Bad Request: Invalid data provided.";
                throw new Error(`Upload failed: ${serverMessage}`);
            }
             if (error.response?.status === 413) {
                throw new Error("Upload failed: File is too large.");
            }
            throw new Error(`Failed to upload file: Server responded with status ${error.response?.status || 'unknown'}`);
        } else {
            throw new Error("Failed to upload file due to an unexpected network or client-side error.");
        }
    }
}

export async function deleteFile(
    token: string | null | undefined,
    id: string 
): Promise<number> { 
    if (!token) {
        console.error("deleteFile: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    if (!id) {
        console.error("deleteFile: File ID is missing.");
        throw new Error("File ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/files/${id}`;
        console.log(`DELETE File URL: ${url}`);

        const res = await axios.delete(url, {
            headers: {
                'Authorization': `Bearer ${token}`,
            }
        });
        return res.status

    } catch (error) {
        console.error(`Error deleting File ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to delete file");
    }
}

export async function getAllFilesUploadedByUser(
    token: string | null | undefined,
    id:string,
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
        const baseUrl = `${getBaseUrl()}/api/v1/users/${id}/files`;
        const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;
        console.log(`GET Request URL: ${url}`); 
        const res = await axios.get(url, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        return res.data.files;

    } catch (error) {
        console.error("Error fetching requests:", error);
         if (axios.isAxiosError(error)) {
             console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) {
                throw new Error("Unauthorized: Invalid or expired token.");
            }
         }
        throw new Error("Failed to get files");
    }
}

interface getfilesParambyFileIds {
    ids: string[]
  }
  // get multiple files by file Ids
  export async function getMultipleFilesByFileIds(token: string | null | undefined, param: getfilesParambyFileIds): Promise<FileInfo[]> { 
    if (!token) {
        console.error("getMultipleFilesByFileIds: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    try {
      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/files/batch`, 
        param,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );
      return res.data.files;
    } catch (error) {
      console.error("Failed to get files:", error);
      throw new Error("Failed to get files");
    }
  }

  export async function importFileToDataBase(token: string | null | undefined, url: {fileUrl: string}): Promise<number> { 
    if (!token) {
        console.error("importFileToDataBase: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    try {
      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/water-quality/upload`, 
        url,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );
      return res.status;
    } catch (error) {
      console.error("Failed to import file:", error);
      throw new Error("Failed to import file");
    }
  }
