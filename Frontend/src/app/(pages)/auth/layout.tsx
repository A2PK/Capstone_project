import type { Metadata } from "next";

// import AppFooter from "@/components/footer";
// import Navbar from "@/components/navbar";

export const metadata: Metadata = {
  title: "Water quality monitoring system",
  description: "The system which helps officiers collect and research the data inside",
};

export default function DashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      {children}
    </>
  );
}
