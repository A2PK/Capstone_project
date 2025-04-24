export interface Notification {
    id: string;
    userId: string;
    title: string;
    description: string;
    read: boolean;
    createdAt: string; 
    updatedAt: string; 
    deletedAt: string | null; 
  }
