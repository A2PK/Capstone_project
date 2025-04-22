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
export const getDonvi = (featureName: string): string => { // <<< THÊM export const
  if (!featureName) {
    return "";
  }

  const normalizedFeature = featureName.toUpperCase().trim();

  switch (normalizedFeature) {
    case "DO":
      return "mg/l";
    case "EC":
      // Sử dụng LaTeX: $\mu$S/cm
      return "µS/cm";
    case "N-NO2":
    case "N-NH4":
    case "P-PO4":
    case "TSS":
    case "COD":
      return "mg/l";
    case "AH":
      return "CFU/ml";
    case "PH":
    case "WQI":
      return "";
    // Thêm các case khác nếu cần
    default:
      // console.warn(`Không tìm thấy đơn vị cho chỉ số: ${featureName}`);
      return "";
  }
};