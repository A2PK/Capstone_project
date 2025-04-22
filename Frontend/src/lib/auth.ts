import { User } from "@/types/user";
import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
  
  export function isExpired(exp: number) {
    const currentTimestamp = Math.floor(Date.now() / 1000);
  
    return currentTimestamp > exp;
  }
  
  export function getSession() {
    const accessToken = cookies().get("access_token")?.value;
    if (!accessToken) return null;
    return accessToken;
  }

  export function getCurrentUser(): User | null {
    const userCookie = cookies().get('user_data');
    if (!userCookie?.value) {
      return null;
    }
    try {
      const userData: User = JSON.parse(userCookie.value);
      return userData; 
    } catch (error) {
      console.error("getCurrentUser (from cookie): Failed to parse user data from cookie:", error);
      return null;
    }
  }

  export function getUserId(): string | null {
    const user = getCurrentUser(); // Gọi hàm đọc từ cookie
    return user?.id ?? null;
  }
  export function getUserRole(): string | null {
    const user = getCurrentUser(); // Gọi hàm đọc từ cookie
    return user?.role ?? null;
  }
  export async function refresh(request: NextRequest) {
    const refreshToken = request.cookies.get("refresh_token")?.value;
    if (!refreshToken) {
      return NextResponse.redirect(new URL("auth/login", request.url));
    }
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/auth/refresh`,
      {
        method: "POST",
        headers: {
          "Set-Cookie": `refreshToken=${refreshToken}`,
        } as HeadersInit,
      }
    );
    const responseData: any = await res.json();
    console.log(responseData);
  
    if (res.ok) {
      const accessToken = responseData.accessToken;
  
      const accessExpires = new Date(
        Date.now() + responseData.expiresAt * 60 * 60 * 1000
      );
      const refreshExpires = new Date(
        Date.now() + responseData.expiresAt * 60 * 60 * 1000
      );
      const res = NextResponse.next();
      res.cookies.set("access_token", accessToken, {
        expires: accessExpires,
        httpOnly: true,
      });
      return res;
    }
    return NextResponse.redirect(new URL("auth/login", request.url));
  }
  export async function logout() {
    cookies().set("access_token", "", { expires: new Date(0), path: "/" });
    cookies().set("refresh_token", "", { expires: new Date(0), path: "/" });
    cookies().set("user_data", "", { expires: new Date(0), path: "/" }); // <-- Xóa cookie user_data
    console.log("Logged out, cleared auth cookies.");
  }