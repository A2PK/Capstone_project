import type { Metadata } from "next";

import { Toaster } from "@/components/ui/toaster";

import "./globals.css";
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
      <body>
        <Toaster />
        {children}
      </body>
    </html>
  );
}
