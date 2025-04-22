import type { Metadata } from "next";
import { redirect } from 'next/navigation'; // Import redirect
import { Suspense } from "react";

import AppFooter from "@/components/footer";
import Navbar from "@/components/navbar"; 
import Sidebar from "@/components/sidebar"; 
import SidebarAdmin from "@/components/sidebaradmin"; 
import { TooltipProvider } from "@/components/ui/tooltip";
import PageLoader from "@/components/pageloader";
import { getUserRole } from "@/lib/auth"; // Import hàm getUserRole

export const metadata: Metadata = {
  title: "Water quality monitoring system",
  description: "The system which helps officers and admins collect and research the data inside", // Cập nhật description nếu cần
};

// Component đơn giản để hiển thị khi không có quyền truy cập
// (Bạn có thể đặt component này ở một file riêng và import nếu dùng ở nhiều nơi)
function Unauthorized() {
  return (
    <div className="flex items-center justify-center h-screen text-center p-4">
      <div>
        <h1 className="text-3xl font-bold text-red-600 mb-4">Unauthorized Access</h1>
        <p className="text-lg text-gray-700">You do not have permission to view this page.</p>
        <p className="text-sm text-gray-500 mt-2">Only users with 'admin' or 'officer' roles can access this area.</p>
        {/* Optional: Link về trang chủ hoặc trang login */}
        <a href="/" className="text-blue-600 hover:underline mt-4 inline-block">Go to Homepage</a>
      </div>
    </div>
  );
}

// Chuyển component thành async function
export default async function DashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const role = await getUserRole(); // Lấy role người dùng

  // 1. Kiểm tra nếu chưa đăng nhập
  if (role === null) {
    console.log("DashboardLayout (General): User not logged in. Redirecting to login.");
    redirect('/auth/login'); // Chuyển hướng về trang login (đảm bảo đường dẫn đúng)
  }

  // 2. Kiểm tra nếu role không hợp lệ (không phải admin VÀ không phải officer)
  if (role !== 'admin' && role !== 'officer') {
    console.warn(`DashboardLayout (General): Unauthorized access attempt by user with role: ${role}`);
    return <Unauthorized />; // Hiển thị trang báo lỗi không có quyền
  }

  // Nếu đã đăng nhập và có role là admin hoặc officer
  console.log(`DashboardLayout (General): Access granted for role: ${role}.`);
  return (
    <>
      <Navbar />
      <div className="flex">
        <TooltipProvider>
          {/* Giả sử Sidebar này phù hợp cho cả admin và officer */}
          {/* Nếu cần Sidebar khác nhau tùy role, bạn có thể thêm logic ở đây */}
          {role === 'admin' ? (
            <SidebarAdmin /> // Hiển thị Sidebar cho Admin
          ) : (
            <Sidebar/> // Hiển thị Sidebar cho Officer (và các role hợp lệ khác nếu có)
          )}
        </TooltipProvider>
        {/* Đảm bảo main có chiều cao tối thiểu để footer không bị đẩy lên */}
        <main className="flex-1 p-6 bg-gray-50 min-h-[calc(100vh-var(--navbar-height,64px)-var(--footer-height,50px))]">
          <Suspense fallback={<PageLoader />}>
            {children}
          </Suspense>
        </main>
      </div>
      <AppFooter />
    </>
  );
}