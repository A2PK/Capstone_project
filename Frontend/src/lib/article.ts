import { Article, CreateArticleData } from "@/types/article";
import { QueryOptions } from "@/types/station2";
import axios from "axios";

const getBaseUrl = () => process.env.NEXT_PUBLIC_BACKEND_URL;
export async function getAllArticles(options: QueryOptions = {}): Promise<Article[]> {
    try {
      const params = new URLSearchParams();
      if (options.limit !== undefined) {params.append("options.limit", options.limit.toString());}
      if (options.offset !== undefined) {params.append("options.offset", options.offset.toString());}
      if (options.sortBy) { params.append("options.sortBy", `"${options.sortBy}"`);} 
      if (options.sortDesc !== undefined) {params.append("options.sortDesc", options.sortDesc.toString());}
      if (options.includeDeleted !== undefined) {params.append("options.includeDeleted", options.includeDeleted.toString());}
      const baseUrl = `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/articles`;
      const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;
      const res = await axios.get(url);
      return res.data.articles;
    } catch (error) {
      console.error(error);
      throw new Error("Failed to get articles");
    }
  }

 
  export async function createArticle(
      token: string | null | undefined,
      data: CreateArticleData 
  ): Promise<Article> { 
      if (!token) {
          console.error("createArticle: Auth token is missing.");
          throw new Error("Authentication token is required.");
      }
  
      try {
          const url = `${getBaseUrl()}/api/v1/articles`;
          console.log(`POST Request URL: ${url}`);
          console.log(`POST Request Data:`, data); 
  
          const res = await axios.post(url, data, { 
              headers: {
                  'Authorization': `Bearer ${token}`,
                  'Content-Type': 'application/json',
                  'Accept': 'application/json'
              }
          });
          return res.data.article;
  
      } catch (error) {
          console.error(`Error creating article:`, error);
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
          throw new Error("Failed to create article");
      }
  }

export async function getArticleById(id: string): Promise<Article> {
    try {
      const res = await axios.get(
        `${getBaseUrl()}/api/v1/articles/${id}`
      );
      return res.data.article;
    } catch (error) {
      console.log(error);
      throw new Error("Failed to get user by id");
    }
  }

export async function deleteArticle(
    token: string | null | undefined,
    id: string ,
    hardDelete: boolean = true
): Promise<number> { 
    if (!token) {
        console.error("deleteArticle: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
    if (!id) {
        console.error("deleteArticle: Request ID is missing.");
        throw new Error("Request ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/articles/${id}?hardDelete=${hardDelete}`;
        console.log(`DELETE Request URL: ${url}`);

        const res = await axios.delete(url, {
            headers: {
                'Authorization': `Bearer ${token}`,
            }
        });
        return res.status

    } catch (error) {
        console.error(`Error deleting article ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to delete article");
    }
}
interface EditArticle {
    badge:string,
content: string,
fileIds: {
    ids: string[]
},
pictureUrl: string,
title: string
}
//Omit<Article, 'id' | 'createdAt' | 'updatedAt' | 'deletedAt'>
export async function updateArticle(
    token: string | null | undefined,
    id: string, 
    data: Partial<EditArticle> 
): Promise<Article> { 
    if (!token) {
        console.error("updateArticle: Auth token is missing.");
        throw new Error("Authentication token is required.");
    }
     if (!id) {
        console.error("updateArticle: Request ID is missing.");
        throw new Error("Request ID is required.");
    }

    try {
        const url = `${getBaseUrl()}/api/v1/articles/${id}`;
        console.log(`PATCH Request URL: ${url}`);

        const res = await axios.patch(url, data, { 
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return res.data.article; 

    } catch (error) {
        console.error(`Error updating article ${id}:`, error);
         if (axios.isAxiosError(error)) {
            console.error("Axios error details:", { status: error.response?.status, data: error.response?.data });
            if (error.response?.status === 401) { throw new Error("Unauthorized: Invalid or expired token."); }
            if (error.response?.status === 404) { throw new Error("Request not found."); }
         }
        throw new Error("Failed to update article");
    }
}

export const generateProxyUrl = (fileUrl: string) => {
    if (!fileUrl || typeof fileUrl !== 'string') {
      console.warn('generateProxyUrl nhận được fileUrl không hợp lệ:', fileUrl);
      return '';
    }
    console.log(fileUrl)
    const apiRouteBasePath = '/api/auth/proxy-file';
    const encodedFileUrl = encodeURIComponent(fileUrl);
    return `${apiRouteBasePath}?url=${encodedFileUrl}`;
  };