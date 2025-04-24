import axios from "axios"; 
import { QueryOptions } from "@/types/station2"; 
import { Notification } from "@/types/notification";

const getBaseUrl = () => process.env.NEXT_PUBLIC_BACKEND_URL;
export async function getAllNotifications(
    token: string | null | undefined,
    options: QueryOptions = {}
): Promise<Notification[]> {
    if (!token) {
        console.error("getAllNotifications: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    try {
        const params = new URLSearchParams();
        if (options.limit !== undefined) { params.append("options.limit", String(options.limit)); }
        if (options.offset !== undefined) { params.append("options.offset", String(options.offset)); }
        if (options.sortBy) { params.append("options.sortBy", options.sortBy); }
        if (options.sortDesc !== undefined) { params.append("options.sortDesc", String(options.sortDesc)); }
        if (options.includeDeleted !== undefined) { params.append("options.includeDeleted", String(options.includeDeleted)); }
        const baseUrl = `${getBaseUrl()}/api/v1/notifications`;
        const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;
        console.log(`GET Request URL: ${url}`); 
        const res = await axios.get(url, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        return res.data.notifications;

    } catch (error) {
        console.error("Error fetching notifications:", error);
         if (axios.isAxiosError(error)) {
             console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) {
                throw new Error("Unauthorized: Invalid or expired token.");
            }
         }
        throw new Error("Failed to get notifications");
    }
}
export type CreateNotificationData = Omit<Notification, 'id' | 'createdAt' | 'updatedAt' | 'deletedAt' | 'read'> & {};
export async function createNotification(
    token: string | null | undefined,
    data: CreateNotificationData 
): Promise<Notification> { 
    if (!token) {
        console.error("createNotification: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/notifications`;
        console.log(`POST Request URL: ${url}`);
        console.log(`POST Request Data:`, data); 

        const res = await axios.post(url, data, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return res.data.notification;

    } catch (error) {
        console.error(`Error creating notification:`, error);
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
        throw new Error("Failed to create notification");
    }
}

export async function markAsReadNotifications(
    token: string | null | undefined,
    Ids: string[] 
): Promise<number> { 
    if (!token) {
        console.error("markAsReadNotification: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/notifications`;
        console.log(`POST Request URL: ${url}`);
        console.log(`POST Request Data:`, Ids); 

        const res = await axios.post(url, {notificationIds: Ids}, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return res.status;

    } catch (error) {
        console.error(`Error mark as read notification:`, error);
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
        throw new Error("Failed to mark as read notification");
    }
}

export async function deleteNotification(
    token: string | null | undefined,
    id: string ,
    hardDelete: boolean = true
): Promise<number> { 
    if (!token) {
        console.error("deleteNotification: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    if (!id) {
        console.error("deleteNotification: Notification ID is missing.");
        throw new Error("Notification ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/notifications/${id}?hardDelete=${hardDelete}`;
        console.log(`DELETE Request URL: ${url}`);

        const res = await axios.delete(url, {
            headers: {
                'Authorization': `Bearer ${token}`,
            }
        });
        return res.status

    } catch (error) {
        console.error(`Error deleting notification ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to delete notification");
    }
}

export async function getNotificationById(
    token: string | null | undefined,
    id: string 
): Promise<Notification> {
    if (!token) {
        console.error("getNotificationById: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
     if (!id) {
        console.error("getNotificationById: Notification ID is missing.");
        throw new Error("Notification ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/notifications/${id}`;
        console.log(`GET Request URL: ${url}`);

        const res = await axios.get(url, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        return res.data.notification; 
    } catch (error) {
        console.error(`Error fetching request ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to get Notification by ID");
    }
}

export async function updateNotification(
    token: string | null | undefined,
    id: string, 
    data: Partial<Omit<Notification, 'id' | 'createdAt' | 'updatedAt' | 'deletedAt' | 'userId'>> 
): Promise<Notification> { 
    if (!token) {
        console.error("updateNotification: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
     if (!id) {
        console.error("updateNotification: Notification ID is missing.");
        throw new Error("Notification ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/notifications/${id}`;
        console.log(`PATCH Request URL: ${url}`);

        const res = await axios.patch(url, data, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return res.data.notification; 

    } catch (error) {
        console.error(`Error updating Notification ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to update Notification");
    }
}

export async function findNotificationsByUserId(
    token: string | null | undefined,
    userId: string, 
    options: QueryOptions = {}
): Promise<Notification[]> {
    if (!token) {
        console.error("findNotificationsByUserId: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    if (!userId) {
        console.error("findNotificationsByUserId: User ID is missing.");
        throw new Error("User ID is required.");
    }
    try {
        const params = new URLSearchParams();
        if (options.limit !== undefined) { params.append("options.limit", String(options.limit)); }
        if (options.offset !== undefined) { params.append("options.offset", String(options.offset)); }
        if (options.sortBy) { params.append("options.sortBy", options.sortBy); }
        if (options.sortDesc !== undefined) { params.append("options.sortDesc", String(options.sortDesc)); }
        if (options.includeDeleted !== undefined) { params.append("options.includeDeleted", String(options.includeDeleted)); }
        const baseUrl = `${getBaseUrl()}/api/v1/users/${userId}/notifications`;
        const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;

        console.log(`GET Request URL: ${url}`);

        const res = await axios.get(url, { // Expect array or object containing array
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
         return res.data.notifications;
    } catch (error) {
        console.error(`Error fetching notifications for user ${userId}:`, error);
        if (axios.isAxiosError(error)) {
             console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("User or requests not found."); }
        }
        throw new Error("Failed to find notifications by user participant");
    }
}
