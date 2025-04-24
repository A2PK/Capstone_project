export interface Request {
    id: string
    createdAt: string
    deletedAt: string
    receiverId: string
    senderId: string
    description?: string;
    status: "pending" | "approved" | "rejected" | "cancelled"
    date: string
    title: string
    updatedAt: string
    fileIds: string[]
  }