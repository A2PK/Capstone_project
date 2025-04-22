"use server";
import { User } from "@/types/user";
import { cookies } from "next/headers";

type LoginFormState = {
  type: "success" | "fail" | "";
  value: string[];
  key?: string;
  user?: User | null;
};

export async function login(
  prevState: LoginFormState,
  formData: FormData
): Promise<LoginFormState> {
  const data = {
    email: formData.get("email")?.toString()!,
    password: formData.get("password")?.toString()!,
  };

  if (!data.email || !data.password) {
    return { type: "fail", value: ["Email and password are required."] };
  }
  console.log("Sending login data:", data);

  try {
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/auth/login`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      }
    );

    let responseData: any;
    try {
        responseData = await res.json();
        console.log("Parsed Response Data:", responseData);
    } catch (parseError) {
        console.error("Failed to parse response JSON:", parseError);
        const responseText = await res.text().catch(() => "Could not read response text");
        return { type: "fail", value: [`Server returned non-JSON response (Status: ${res.status}). Body: ${responseText.substring(0,100)}...`] };
    }


    if (res.ok) {
      const accessToken = responseData.accessToken;
      const refreshToken = responseData.refreshToken;
      const user = responseData.user as User; // Lấy user object

      // Kiểm tra token và user
      if (typeof accessToken !== 'string' || !accessToken || typeof refreshToken !== 'string' || !refreshToken || !user) {
        console.error("Missing or invalid tokens/user in successful login response");
        return { type: "fail", value: ["Login successful but failed to retrieve valid authentication data."] };
      }

      // Tính toán thời gian hết hạn
      const accessExpires = responseData.expiresAt
        ? new Date(parseInt(responseData.expiresAt, 10) * 1000)
        : new Date(Date.now() + 15 * 60 * 1000);
      const refreshExpires = responseData.refreshTokenExpiresIn
        ? new Date(Date.now() + parseInt(responseData.refreshTokenExpiresIn, 10) * 1000)
        : new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);

      // --- LƯU USER VÀO COOKIE ---
      let userCookieSet = false;
      try {
        const userString = JSON.stringify(user); // Chuyển thành chuỗi JSON
        // Kiểm tra kích thước (ước lượng, nên kiểm tra kỹ hơn nếu user object lớn)
        if (userString.length < 3800) { // Giới hạn an toàn ~3.8KB
            cookies().set("user_data", userString, {
              expires: refreshExpires, // Cho hết hạn cùng refresh token
              httpOnly: true,         // *** Rất quan trọng: Đặt là true vì getCurrentUser chạy server-side ***
              secure: process.env.NODE_ENV === "production",
              path: "/",
              sameSite: "lax",
            });
            userCookieSet = true;
            console.log("User data saved to 'user_data' cookie.");
        } else {
            console.warn("User data is too large to fit in a cookie securely.");
            // Có thể chọn không lưu, hoặc lưu một phần dữ liệu user nếu muốn
        }
      } catch (stringifyError) {
         console.error("Failed to stringify user data for cookie:", stringifyError);
         // Bỏ qua nếu không thể stringify
      }
      // --------------------------

      // Đặt cookies cho token
      cookies().set("access_token", accessToken, {
        expires: accessExpires,
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        path: "/",
        sameSite: "lax",
      });
      cookies().set("refresh_token", refreshToken, {
        expires: refreshExpires,
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        path: "/",
        sameSite: "lax",
      });

      return {
        type: "success",
        value: ["Login successfully"],
        user: user, // Vẫn trả về user trong state
      };
    } else {
       const errorMessage = responseData.message || responseData.error || `Login failed (Status: ${res.status})`;
       console.error("Login API returned error:", errorMessage);
       return { type: "fail", value: [errorMessage] };
    }
  } catch (error: any) {
    console.error("Login action network/unexpected error:", error);
    return { type: "fail", value: [`An unexpected error occurred: ${error.message}`] };
  }
}

// Bước 2: Cập nhật hàm logout để xóa cookie user_data
export async function logout() {
  cookies().set("access_token", "", { expires: new Date(0), path: "/" });
  cookies().set("refresh_token", "", { expires: new Date(0), path: "/" });
  cookies().set("user_data", "", { expires: new Date(0), path: "/" }); // <-- Xóa cookie user_data
  console.log("Logged out, cleared auth cookies.");
}