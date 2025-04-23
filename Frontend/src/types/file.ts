export interface FileInfo {
    id: string;
    name: string;
    type: string; 
    size: string; 
    url: string; 
    serviceInternalId: string; 
    createdAt: string; 
    updatedAt: string; 
    deletedAt: string | null; 
  }

 export interface postFileParam {
    type: string;
    name: string;
    file: File;
}