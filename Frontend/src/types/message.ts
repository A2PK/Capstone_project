export interface Message {
    id: string;
    senderId: string;
    receiverId: string;
    message: string; 
    read: boolean;
    createdAt: string; 
    updatedAt: string; 
    deletedAt: string | null; 
  }
export interface messgeParam {
    message: string;
    receiverId: string;
}