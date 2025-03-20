import type { Metadata } from "next";
import { Toaster } from "@/components/ui/toaster";
import { Skeleton } from "@/components/ui/skeleton";
import { Loader2 } from "lucide-react";
import "./globals.css";
import { Suspense } from "react";

export const metadata: Metadata = {
  title: "Water Quality Monitoring System",
  description: "The system which helps officers collect and research the data inside.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/atlogo.png" type="image/jpeg" sizes="16x16" />
      </head>
      <body className="bg-gray-50">
        <Toaster />
        <Suspense
          fallback={
            <div className="flex flex-col items-center justify-center h-screen space-y-4">
              <Loader2 className="h-10 w-10 animate-spin text-blue-500" />
              <p className="text-gray-600">Đang tải dữ liệu...</p>
            </div>
          }
        >
          {children}
        </Suspense>
      </body>
    </html>
  );
}
