// Trong LoginPage.tsx
"use client";

import { useFormState, useFormStatus } from "react-dom"; // Import useFormState và useFormStatus
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import Link from "next/link";
import Image from "next/image";
import { login } from "../actions"; // Import Server Action
import water from "/public/songnuoc.jpg";
import logowqm from "/public/wqm.jpg";
import { Eye, EyeOff } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { User } from "@/types/user";
// Định nghĩa lại kiểu state cho phù hợp với Server Action
type LoginFormState = {
  type: "success" | "fail" | "";
  value: string[];
  key?: string; // Thêm key để reset form state nếu cần
  user?: User | null;
};

const initialState: LoginFormState = { type: "", value: [], user: null };

// Component nút Submit riêng để dùng useFormStatus
function SubmitButton() {
  const { pending } = useFormStatus(); // Hook để lấy trạng thái pending của form
  return (
    <Button
      type="submit"
      className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600"
      disabled={pending} // Disable nút khi form đang được gửi
    >
      {pending ? "Đang đăng nhập..." : "Đăng nhập"}
    </Button>
  );
}


export default function LoginPage() {
  // Sử dụng useFormState
  const [state, formAction] = useFormState<LoginFormState, FormData>(login, initialState);

  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false); // Giữ lại nếu muốn dùng cho cookie expiry

  const router = useRouter();
  const { toast } = useToast();

  // Xử lý kết quả trả về từ Server Action
  useEffect(() => {
    if (state.type === "success" && state.user) {
      toast({
        variant: "success",
        title: "Đăng nhập thành công !",
        description: state.value[0] || `Chào mừng, ${state.user.firstName}!`, // Có thể dùng tên user
      });

      try {
        // Lưu user data vào localStorage
        localStorage.setItem('userData', JSON.stringify(state.user));
        console.log('User data saved to localStorage:', state.user);
      } catch (error) {
        console.error("Failed to save user data to localStorage:", error);
        // Có thể hiển thị toast lỗi nếu cần
      }

      // Chuyển hướng sau khi lưu
      router.refresh(); // Quan trọng: refresh để cập nhật trạng thái server (ví dụ: layout đọc cookie mới)
      router.push("/"); // Chuyển đến trang chính
    } else if (state.type === "fail") {
      toast({
        title: "Login Failed",
        description: state.value[0] || "Please check your credentials.",
        variant: "destructive",
      });
    }
  }, [state, router, toast]);


  return (
    <div className="flex flex-col md:flex-row h-screen">
      {/* Left Section */}
      <div className="hidden md:block md:w-2/3 relative h-full">
         <Image src={water} alt="Waterplace" layout="fill" objectFit="cover" priority />
      </div>

      {/* Right Section */}
      <div className="flex flex-1 flex-col justify-center items-center px-6 sm:px-8 md:px-16 bg-white">
        <div className="w-full max-w-xs sm:max-w-sm md:max-w-lg">
          {/* Logo */}
           <div className="flex justify-center mb-8">
              <Link href={"/"}><Image src={logowqm} alt="WQM Logo" width={150} height={60} className="h-16" priority /></Link>
            </div>
          <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-center">Đăng nhập</h2>
          {/* Sử dụng formAction từ useFormState */}
          <form action={formAction}>
             {/* Input Email */}
            <div className="mb-4">
               <label htmlFor="email" className="block mb-2 text-sm font-medium">Tài khoản</label>
               <Input id="email" name="email" type="email" placeholder="Nhập Email hoặc SĐT" className="w-full" required />
             </div>
             {/* Input Password */}
            <div className="mb-4">
              <label htmlFor="password" className="block mb-2 text-sm font-medium">Mật khẩu</label>
              <div className="relative">
                <Input id="password" name="password" type={showPassword ? "text" : "password"} placeholder="Nhập mật khẩu" className="w-full pr-10" required />
                <Button type="button" variant="ghost" size="icon" className="absolute right-2 top-1/2 transform -translate-y-1/2" onClick={() => setShowPassword(!showPassword)}>
                  {showPassword ? <EyeOff className="h-4 w-4 text-gray-500" /> : <Eye className="h-4 w-4 text-gray-500" />}
                </Button>
              </div>
            </div>
             {/* Remember Me & Forgot Password */}
            <div className="flex items-center justify-between mb-6">
              <label className="flex items-center space-x-2">
                <input type="checkbox" name="rememberMe" className="form-checkbox" checked={rememberMe} onChange={(e) => setRememberMe(e.target.checked)} />
                <span className="text-sm">Ghi nhớ đăng nhập</span>
              </label>
              <Link href="/forgot-password" className="text-sm text-blue-500">Quên mật khẩu?</Link>
            </div>
             {/* Nút Submit */}
            <SubmitButton />
            {/* Hiển thị lỗi chung từ state nếu có */}
            {/* {state.type === 'fail' && <p className="text-red-500 text-sm mt-2">{state.value.join(', ')}</p>} */}
          </form>
        </div>
      </div>
    </div>
  );
}