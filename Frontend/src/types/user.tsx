import { PaginParam } from "./station2";

export interface User {
  id: string;
  createdAt: string;
  updatedAt: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  role: string; // Hoặc cụ thể hơn: 'admin' | 'officer' | ...
  isActive: boolean;
  phone?: string | null; // Thêm ? hoặc | null nếu có thể thiếu
  address?: string | null;
  age?: number | null;
  profilePic?: string | null;
}

export interface getUsers {
  users: User[],
  paginationInfo: PaginParam
}