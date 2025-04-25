import type { Metadata } from "next";
import { redirect } from 'next/navigation'; 
import { Suspense } from "react";

import AppFooter from "@/components/footer";
import Navbar from "@/components/navbar"; 
import Sidebar from "@/components/sidebar"; 
import SidebarAdmin from "@/components/sidebaradmin"; 
import { TooltipProvider } from "@/components/ui/tooltip";
import PageLoader from "@/components/pageloader";
import { getUserRole } from "@/lib/auth"; 

export const metadata: Metadata = {
  title: "Water quality monitoring system",
  description: "The system which helps officers and admins collect and research the data inside", 
};

function Unauthorized() {
  return (
    <div className="flex items-center justify-center h-screen text-center p-4">
      <div>
        <h1 className="text-3xl font-bold text-red-600 mb-4">Truy cập Không được chấp nhận</h1>
        <p className="text-lg text-gray-700">Bạn không có quyền xem trang này.</p>
        <p className="text-sm text-gray-500 mt-2">Chỉ người dùng {'"'} có vai trò quản trị viên{'"'} hoặc {'"'}viên chức{'"'} mới có thể truy cập.</p>
        <a href="/" className="text-blue-600 hover:underline mt-4 inline-block">Quay lại trang chủ</a>
      </div>
    </div>
  );
}

export default async function DashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const role = await getUserRole(); 
  if (role === null) {
    redirect('/auth/login'); 
  }
  if (role !== "manager" && role !== 'admin' && role !== 'officer') {
    return <Unauthorized />; 
  }
  return (
    <>
      <Navbar />
      <div className="flex">
        <TooltipProvider>
          {((role === 'admin') || (role === 'manager')) ? (
            <SidebarAdmin /> 
          ) : (
            <Sidebar/> 
          )}
        </TooltipProvider>
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