export interface Request {
    id: string
    subject: string
    message: string
    status: "pending" | "approved" | "rejected" | "cancelled"
    date: string
    files: string[]
  }