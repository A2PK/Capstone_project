import type { Metadata } from "next";

import AppFooter from "@/components/footer";
import Navbar from "@/components/navbar";

export const metadata: Metadata = {
  title: "Ecommerce Analysis system",
  description: "Hotel booking system for many gorgeous rooms inside",
};

export default function DashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <Navbar />
      {children}
      <AppFooter />
    </>
  );
}
