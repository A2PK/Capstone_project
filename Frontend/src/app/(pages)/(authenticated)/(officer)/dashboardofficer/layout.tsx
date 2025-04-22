import type { Metadata } from "next";
import { redirect } from 'next/navigation'; // Import redirect
import { Suspense } from "react";

import AppFooter from "@/components/footer";
import Navbar from "@/components/navbar"; // Navbar cho officer (khác với admin?)
import Sidebar from "@/components/sidebar"; // Sidebar cho officer (khác với admin?)
import { TooltipProvider } from "@/components/ui/tooltip";
import PageLoader from "@/components/pageloader";
import { getUserRole } from "@/lib/auth"; // Import hàm server-side

export const metadata: Metadata = {
  title: "Officer Dashboard - WQM System", // Đặt title cụ thể hơn
  description: "The system which helps officers collect and research the data inside",
};

// Có thể dùng lại component Unauthorized từ layout admin hoặc tạo riêng
function UnauthorizedAccess() {
  return (
    <div className="flex items-center justify-center h-screen text-center p-4">
      <div>
        <h1 className="text-3xl font-bold text-red-600 mb-4">Unauthorized Access</h1>
        <a href="/" className="text-blue-600 hover:underline mt-4 inline-block">Go to Homepage</a>
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
    console.log("DashboardLayout (Officer): User not logged in. Redirecting to login.");
    redirect('/auth/login'); 
  }
  if (role !== 'officer') {
    console.warn(`DashboardLayout (Officer): Unauthorized access attempt by user with role: ${role}`);
    return <UnauthorizedAccess />;
  }
  console.log("DashboardLayout (Officer): Access granted.");
  return (
    <>
      {/* Sử dụng Navbar/Sidebar của officer */}
      <Navbar />
      <div className="flex">
        <TooltipProvider>
          <Sidebar />
        </TooltipProvider>
        <main className="flex-1 p-6 bg-gray-50 min-h-[calc(100vh-var(--navbar-height)-var(--footer-height))]">
          <Suspense fallback={<PageLoader />}>
            {children} {/* Nội dung các trang của officer */}
          </Suspense>
        </main>
      </div>
      <AppFooter />
    </>
  );
}