import { User } from "@/types/user";
import axios from "axios";
export async function getAllStaffs(): Promise<User> {
    try {
      const res = await axios.get(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/staffs-api/api/staffs`
      );
      console.log(res);
      return res.data;
    } catch (error) {
      console.log(error);
      throw new Error("Failed to get doctors");
    }
  }