import { PaginParam } from "./station2";

export interface User {
  id: string;
  createdAt: string;
  updatedAt: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  role: string; // 'admin' | 'officer' 
  isActive: boolean;
  phone?: string | null; 
  address?: string | null;
  age?: number | null;
  profilePic?: string | null;
  password?:string
}

export interface getUsers {
  users: User[],
  paginationInfo: PaginParam
}