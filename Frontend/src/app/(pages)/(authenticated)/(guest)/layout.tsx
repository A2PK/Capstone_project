import type { Metadata } from "next";
import AppFooter from "@/components/footer";
import Navbar from "@/components/navbarguest";
import Sidebar from "@/components/sidebarguest";

export const metadata: Metadata = {
  title: "Water quality monitoring system",
  description: "The system which helps officers collect and research the data inside",
};

export default function DashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <Navbar />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6 bg-gray-50">
          {children}
        </main>
      </div>
      <AppFooter />
    </>
  );
}