import type { Metadata } from "next";
import { redirect } from 'next/navigation'; // Import redirect từ next/navigation
import { Suspense } from "react";

import AppFooter from "@/components/footer";
import Navbar from "@/components/navbar";
import SidebarAdmin from "@/components/sidebaradmin";
import { TooltipProvider } from "@/components/ui/tooltip";
import PageLoader from "@/components/pageloader";
import { getUserRole } from "@/lib/auth"; // Import hàm getUserRole server-side

export const metadata: Metadata = {
  // Cập nhật title/description nếu layout này chỉ dành cho Admin
  title: "Admin Dashboard - WQM System",
  description: "Admin management area for the Water Quality Monitoring system.",
};

// Component đơn giản để hiển thị khi không có quyền truy cập
function Unauthorized() {
  return (
    <div className="flex items-center justify-center h-screen text-center p-4">
      <div>
        <h1 className="text-3xl font-bold text-red-600 mb-4">Unauthorized Access</h1>
        <p className="text-lg text-gray-700">You do not have permission to view this page.</p>
        <p className="text-sm text-gray-500 mt-2">Please contact the administrator if you believe this is an error.</p>
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
  const role = await getUserRole();
  if (role === null) {
    console.log("DashboardLayout (Admin): User not logged in. Redirecting to login.");
    redirect('/auth/login'); // Chuyển hướng về trang login
  }
  if (role !== 'admin') {
    console.warn(`DashboardLayout (Admin): Unauthorized access attempt by user with role: ${role}`);
    return <Unauthorized />;
  }

  console.log("DashboardLayout (Admin): Access granted.");
  return (
    <>
      <Navbar />
      <div className="flex">
        <TooltipProvider>
          <SidebarAdmin />
        </TooltipProvider>
        <main className="flex-1 p-6 bg-gray-50 min-h-[calc(100vh-var(--navbar-height)-var(--footer-height))]"> {/* Đảm bảo main có chiều cao tối thiểu */}
          <Suspense fallback={<PageLoader />}>
            {children} 
          </Suspense>
        </main>
      </div>
      <AppFooter />
    </>
  );
}