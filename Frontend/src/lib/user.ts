import { getUsers, User} from "@/types/user";
import { QueryOptions } from "@/types/station2";
import axios from "axios";

export async function getAllUsers(options: QueryOptions = {}): Promise<getUsers> {
    try {
      const res = await axios.get(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/users` 
      );
      console.log(res);
      return res.data;
    } catch (error) {
      console.log(error);
      throw new Error("Failed to get users");
    }
  }

export async function getUserById(id: string): Promise<User> {
  try {
    const res = await axios.get(
      `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/users/${id}`
    );
    return res.data;
  } catch (error) {
    console.log(error);
    throw new Error("Failed to get user by id");
  }
}

export async function createUser(user: User): Promise<number> {
    try {
      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/users`,
        user,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      return res.status;
    } catch (error) {
      console.error("Failed to create user:", error);
      throw new Error("Failed to create user");
    }
  }
export async function updateUser(update: Partial<User> , id: string): Promise<number> {
  try {
    const res = await axios.patch( 
      `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/users/${id}`,
      update, 
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    return res.status;
  } catch (error) {
    console.error("Failed to update user:", error);
    throw new Error("Failed to update user");
  }
}

export async function deleteUser(id: string, hardDelete: boolean = true): Promise<number> {
  try {
    const url = `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/users/${id}?hardDelete=${hardDelete}`;
    const res = await axios.delete(url);
    return res.status;
  } catch (error) {
    console.error("Failed to delete user:", error);
    throw new Error("Failed to delete user");
  }
}

interface searchParamsUser {
  options: {
    filters: {
      email: string
    },
    includeDeleted: boolean,
    limit: number,
    offset: number,
    sortBy: string,
    sortDesc: boolean
  }
}
export async function searchUser(param: searchParamsUser): Promise<getUsers> { 
  try {
    const res = await axios.post(
      `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/users/search`, 
      param,
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    return res.data;
  } catch (error) {
    console.error("Failed to search users:", error);
    throw new Error("Failed to search users");
  }
}