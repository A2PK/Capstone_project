import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
export const getStatusTextColor = (status: string) => {
  switch (status) {
    case "Rất tốt":
      return "text-emerald-900";
    case "Tốt":
      return "text-green-800";
    case "Bình thường":
      return "text-amber-800";
    case "Không tốt":
      return "text-orange-800";
    case "Nguy hiểm":
      return "text-red-800";
    default:
      return "text-gray-800";
  }
};
