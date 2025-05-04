"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";

export default function RootPage() {
  const router = useRouter();

  useEffect(() => {
    let isMounted = true;

    async function fetchRoleAndRedirect() {
      let fetchedRole: string | null = null;

      try {
        console.log("Fetching role from /api/auth/role...");
        const res = await fetch('/api/auth/role'); // Gọi API route
        if (!isMounted) return;

        if (res.ok) {
          const data = await res.json();
          fetchedRole = data.role; // Lấy role từ response API
          console.log("Fetched role:", fetchedRole);
        } else {
          // API trả về lỗi (ví dụ 401 - không có session hợp lệ)
          console.log("Fetching role failed, status:", res.status);
          // fetchedRole vẫn là null
        }
      } catch (error) {
        if (!isMounted) return;
        console.error("Error fetching role:", error);
        // Lỗi mạng, fetchedRole vẫn là null
      }
      let intendedRoute = "/homeguest"; 
      if (fetchedRole) {
        if (fetchedRole === "officer") {
          intendedRoute = "/dashboardofficer/homepage";
        } else if ((fetchedRole === "admin")||(fetchedRole === "manager")) {
          intendedRoute = "/homead";
        }
      }

      console.log(`Redirecting to: ${intendedRoute}`);
      router.replace(intendedRoute);
    }

    fetchRoleAndRedirect();

    return () => {
      isMounted = false;
    };
  }, [router]);

  return (
    <div className="flex flex-col items-center justify-center h-screen space-y-4 bg-gray-50">
      <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
      <p className="text-sm text-gray-500">Checking authentication...</p>
    </div>
  );
}