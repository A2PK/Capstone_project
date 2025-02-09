"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";

export default function Error() {
  const router = useRouter();
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-3">
      <h2>Something went wrong!</h2>
      <Button onClick={() => router.refresh()}>Try again</Button>
    </div>
  );
}
