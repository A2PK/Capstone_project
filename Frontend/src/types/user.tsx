export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: string; // You might consider using a more specific type like 'user' | 'admin' | 'officer' if the roles are fixed
  createdAt: string; // Or potentially number or Date if you plan to convert it immediately
  updatedAt: string; // Or potentially number or Date
  username: string;
  isActive: boolean;
  lastLoginAt: string; // Or potentially number or Date
  phone: string;
  address: string;
  age: number;
  profilePic: string;
}