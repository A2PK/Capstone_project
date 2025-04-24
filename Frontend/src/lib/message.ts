import { Message, messgeParam } from "@/types/message";
import { QueryOptions } from "@/types/station2";
import axios from "axios";

const getBaseUrl = () => process.env.NEXT_PUBLIC_BACKEND_URL;

export async function sendDirectMessage(
    token: string | null | undefined,
    data: messgeParam 
): Promise<Message> { 
    if (!token) {
        console.error("sendDirectMessage: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/messages`;
        console.log(`POST file URL: ${url}`);
        console.log(`POST file Data:`, data); 

        const res = await axios.post(url, data, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return res.data.message;

    } catch (error) {
        console.error(`Error send direct message:`, error);
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
        throw new Error("Failed to send direct message");
    }
}
interface paramAsRead {
    messageIds: string[] 
}
export async function markAsReadMessages(
    token: string | null | undefined,
    messageIds: paramAsRead
): Promise<number> { 
    if (!token) {
        console.error("markAsReadNotification: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/messages/read`;
        console.log(`POST Request URL: ${url}`);
        console.log(`POST Request Data:`, messageIds); 

        const res = await axios.post(url, messageIds, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return res.status;

    } catch (error) {
        console.error(`Error mark as read messages:`, error);
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
        throw new Error("Failed to mark as read messages");
    }
}

export async function getAllMessagesBetweenUsers(
    token: string | null | undefined,
    userId1: string,
    userId2: string,
    options: QueryOptions = {}
): Promise<Message[]> {
    if (!token) {
        console.error("getAllMessagesBetweenUsers: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    try {
        const params = new URLSearchParams();
        if (options.limit !== undefined) { params.append("options.limit", String(options.limit)); }
        if (options.offset !== undefined) { params.append("options.offset", String(options.offset)); }
        if (options.sortBy) { params.append("options.sortBy", options.sortBy); }
        if (options.sortDesc !== undefined) { params.append("options.sortDesc", String(options.sortDesc)); }
        if (options.includeDeleted !== undefined) { params.append("options.includeDeleted", String(options.includeDeleted)); }
        const baseUrl = `${getBaseUrl()}/api/v1/messages/conversation?userId1=${userId1}&userId2=${userId2}`;
        const url = params.toString() ? `${baseUrl}&${params.toString()}` : baseUrl;
        console.log(`GET Request URL: ${url}`); 
        const res = await axios.get(url, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        return res.data.messages;

    } catch (error) {
        console.error("Error fetching conversation:", error);
         if (axios.isAxiosError(error)) {
             console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) {
                throw new Error("Unauthorized: Invalid or expired token.");
            }
         }
        throw new Error("Failed to get conversation");
    }
}

export async function getUnseenMessagesForCurrentUser( token: string | null | undefined): Promise<Message[]> {
    if (!token) {
        console.error("getUnseenMessagesForCurrentUser: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    try {
        const params = new URLSearchParams();
        const baseUrl = `${getBaseUrl()}/api/v1/messages/unseen`;
        console.log(`GET Request URL: ${baseUrl}`); 
        const res = await axios.get<{ messages: Message[] }>(baseUrl, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });
        return res.data?.messages || [];

    } catch (error) {
        console.error("Error unseen messages of current user:", error);
         if (axios.isAxiosError(error)) {
             console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) {
                throw new Error("Unauthorized: Invalid or expired token.");
            }
         }
        throw new Error("Failed to get unseen messages");
    }
}