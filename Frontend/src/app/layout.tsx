import type { Metadata } from "next";

import { Toaster } from "@/components/ui/toaster";

import "./globals.css";

export const metadata: Metadata = {
  title: "Ecommerce Analysis system",
  description: "Hotel booking system for many gorgeous rooms inside",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
      <link rel="icon" href="https://www.svgrepo.com/show/324142/dashboard-graph-analytics-report.svg" type="image/svg+xml" sizes="16x16" />
      </head>
      <body>
        <Toaster />
        {children}
       
      </body>
    </html>
  );
}
