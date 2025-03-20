import type { Metadata } from "next";
import AppFooter from "@/components/footer";
import Navbar from "@/components/navbar";
import Sidebar from "@/components/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Suspense } from "react";
import PageLoader from "@/components/pageloader";
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
        <TooltipProvider>
          <Sidebar />
        </TooltipProvider>
        <main className="flex-1 p-6 bg-gray-50">
        <Suspense fallback={<PageLoader />}>
  {children}
</Suspense>
        </main>
      </div>
      <AppFooter />
    </>
  );
}
