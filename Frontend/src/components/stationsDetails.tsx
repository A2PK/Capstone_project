// File: src/components/stationsDetails.tsx
import {
  Card,
  CardHeader,
  CardContent,
  CardFooter
} from "@/components/ui/card"; // Điều chỉnh đường dẫn nếu cần
import { Badge } from "@/components/ui/badge"; // Điều chỉnh đường dẫn nếu cần
import {
  AlertTriangle, Activity, Clock, MapPin, Thermometer, Droplet, Zap, Wind,
  Cloudy, FlaskConical, Beaker, TestTube, HelpCircle, Waves, Gauge, Info, // Thêm các icon cần thiết
  FlaskRound
} from "lucide-react";
import { Station as BaseStation, DataPoint, Indicator } from "@/types/station2"; // Điều chỉnh đường dẫn nếu cần
// --- THÊM IMPORT ---
import { ElementRange } from '@/types/threshold'; // Import kiểu ElementRange // Điều chỉnh đường dẫn nếu cần
import { useCallback } from 'react'; // Import useCallback nếu cần
import { cn, getDonvi } from "@/lib/utils"; // Import cn để kết hợp class // Điều chỉnh đường dẫn nếu cần
// -------------------
import { format, parseISO } from 'date-fns';

// Interface cho prop `selectedStation` đã được kết hợp thông tin
interface CombinedStationInfo extends BaseStation {
  wqi: number | string | null;
  status: string;
  time: string; // Thời gian của latestDataPoint
  recommendation: string;
  createdAt: string; // Thời gian tạo/cập nhật của bản ghi station (dùng cho footer)
}

// Cập nhật interface Props để nhận thêm thresholds
interface StationDetailsProps {
  selectedStation: CombinedStationInfo | null;
  latestDataPoint: DataPoint | null;
  availableFeatures: string[];
  thresholds: ElementRange[] | null; // <-- THÊM PROP THRESHOLDS
}

// --- Helper Function: Get Icon for Feature ---
function getIconForFeature(featureName: string) {
   switch (featureName.toUpperCase()) {
      case 'PH': return FlaskRound;
      case 'DO': return Droplet;
      case 'TSS': return Waves;
      case 'COD': return Activity;
      case 'EC': return Zap;
      case 'TEMP': return Thermometer;
      case 'N-NO2': return Beaker;
      case 'NO2': return Beaker; // Thêm nếu có
      case 'N-NH4': return FlaskConical;
      case 'NH4': return FlaskConical; // Thêm nếu có
      case 'P-PO4': return TestTube;
      case 'PO4': return TestTube; // Thêm nếu có
      case 'AH': return HelpCircle;
      case 'WQI': return Gauge;
      default: return HelpCircle;
  }
}
// -----------------------------------------

export default function StationDetails({
  selectedStation,
  latestDataPoint,
  availableFeatures,
  thresholds // <-- Nhận prop thresholds
}: StationDetailsProps) {

  // --- HÀM KIỂM TRA VƯỢT NGƯỠNG ---
  const isValueOutOfRange = useCallback((
      featureName: string,
      value: string | number | null,
      thresholdsToCheck: ElementRange[] | null
  ): boolean => {
      if (value === null || typeof value === 'string' || !thresholdsToCheck || thresholdsToCheck.length === 0) {
          return false;
      }
      const numericValue = value;
      const keyLower = featureName.toLowerCase();
      const config = thresholdsToCheck.find(c => c.elementName.toLowerCase() === keyLower);
      if (!config) return false;
      const { minValue, maxValue } = config;
      if (minValue !== null && numericValue < minValue) return true;
      if (maxValue !== null && numericValue > maxValue) return true;
      return false;
  }, []);
  // --------------------------------------

  if (!selectedStation) {
      return (
          <div className="text-center text-gray-500 p-4 mt-4 bg-gray-50 border rounded-lg shadow-sm">
              Chọn một trạm trên bản đồ hoặc từ bảng để xem chi tiết.
          </div>
      );
  }

  const getStatusTailwindClasses = (status: string): string => {
       switch (status) {
           case "Rất Tốt": return "bg-emerald-100 text-emerald-900 border-emerald-200";
           case "Tốt": return "bg-green-100 text-green-800 border-green-200";
           case "Trung Bình": return "bg-yellow-100 text-yellow-800 border-yellow-200";
           case "Kém": return "bg-orange-100 text-orange-800 border-orange-200";
           case "Rất Kém": return "bg-red-100 text-red-800 border-red-200";
           default: return "bg-gray-100 text-gray-800 border-gray-200";
      }
  };

   const getFeatureValue = (featureName: string): string | number | null => {
       if (!latestDataPoint || !latestDataPoint.features) return null;
       // Tìm chính xác tên trước
       let feature = latestDataPoint.features.find(f => f.name.toLowerCase() === featureName.toLowerCase());
       // Nếu không thấy, thử tìm tên có chứa (ví dụ 'N-NH4' vs 'NH4') - Cẩn thận nếu có tên trùng lặp không mong muốn
       if (!feature) {
          feature = latestDataPoint.features.find(f => f.name.toLowerCase().includes(featureName.toLowerCase()));
       }

       if (!feature) return null;

       // Ưu tiên value số nếu có và hợp lệ
       if (typeof feature.value === 'number' && !isNaN(feature.value)) {
            return feature.value;
       }
       // Sau đó dùng textualValue nếu là số hợp lệ
       if (typeof feature.textualValue === 'string' && feature.textualValue.trim() !== '') {
            const parsed = parseFloat(feature.textualValue);
            if (!isNaN(parsed)) return parsed;
       }
       // Nếu không có giá trị số, trả về textualValue
       if(feature.textualValue) return feature.textualValue;

       return null; // Hoặc 'N/A'
   }

  // Hàm định dạng thời gian ở Footer
  const formatFooterTime = (timeString: string | undefined | null): string => {
      if (!timeString) return 'Chưa có';
      try {
          const dateObject = parseISO(timeString);
          // Kiểm tra xem dateObject có hợp lệ không
           if (isNaN(dateObject.getTime())) {
               console.error("Invalid date format for footer:", timeString);
               return timeString; // Trả về chuỗi gốc nếu không parse được
           }
          return format(dateObject, 'HH:mm dd/MM/yyyy');
      } catch (error) {
          console.error("Lỗi định dạng ngày footer:", timeString, error);
          return timeString; // Trả về chuỗi gốc nếu lỗi
      }
  };


  return (
      <Card className="w-full bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
          <CardHeader className="pb-3 pt-4 px-4 bg-gray-50 border-b border-gray-200">
               <div className="flex flex-wrap justify-between items-start gap-2">
                 <div>
                     <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                         <MapPin className="w-5 h-5 text-blue-600 inline-block flex-shrink-0"/>
                         <span className="break-words">Trạm: {selectedStation.name}</span>
                     </h2>
                     {selectedStation.location && (
                         <p className="text-sm text-gray-600 mt-1" style={{ paddingLeft: '28px' }}>{selectedStation.location}</p>
                     )}
                 </div>
                 <Badge variant="outline" className={`${getStatusTailwindClasses(selectedStation.status)} px-3 py-1 text-sm font-medium rounded-md whitespace-nowrap self-center`}>
                     {selectedStation.status}
                 </Badge>
               </div>
          </CardHeader>

          <CardContent className="p-4">
              {/* Phần WQI và Khuyến cáo */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                   <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 flex flex-col justify-between">
                     <div className="flex items-center gap-2 mb-2">
                         <Gauge className="w-5 h-5 text-blue-700 flex-shrink-0" />
                         <h3 className="font-semibold text-blue-800">Chỉ số chất lượng nước (WQI)</h3>
                     </div>
                     <div className="text-right">
                         <span className="text-4xl font-bold text-blue-700">{selectedStation.wqi ?? "--"}</span>
                     </div>
                   </div>
                   <div className="bg-amber-50 p-4 rounded-lg border border-amber-100">
                     <div className="flex items-center gap-2 mb-2">
                         <Info className="w-5 h-5 text-amber-700 flex-shrink-0" />
                         <h3 className="font-semibold text-amber-800">Khuyến cáo</h3>
                     </div>
                     <p className="text-sm text-gray-800 leading-relaxed">{selectedStation.recommendation}</p>
                   </div>
              </div>

              {/* Phần hiển thị các chỉ số chi tiết */}
              {latestDataPoint && availableFeatures.length > 0 && (
                  <>
                      {/* Sử dụng selectedStation.time để hiển thị thời gian của dữ liệu */}
                      <h4 className="text-md font-semibold text-gray-700 mb-3 mt-5 border-t pt-4">
                          Các chỉ số thành phần (Lúc: {selectedStation.time || 'N/A'})
                      </h4>
                      <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                          {availableFeatures.map(featureName => {
                              const value = getFeatureValue(featureName);
                              const IconComponent = getIconForFeature(featureName);
                              const unit = getDonvi(featureName);
                              const outOfRange = isValueOutOfRange(featureName, value, thresholds); // Kiểm tra vượt ngưỡng

                              return (
                                  <div key={featureName} className="bg-gray-50 p-3 rounded-md border border-gray-200 hover:shadow-sm transition-shadow duration-150">
                                      <div className="flex items-center justify-center text-gray-600 mb-1">
                                          <IconComponent className="w-4 h-4 mr-1.5 flex-shrink-0" />
                                          <p className="text-xs font-semibold uppercase tracking-wide">{featureName} </p>
                                      </div>
                                      <p className={cn( // Áp dụng class tô đỏ nếu vượt ngưỡng
                                          "text-xl font-bold text-gray-900 text-center",
                                          outOfRange && "text-red-600 font-bold"
                                      )}>
                                          {value !== null ? (typeof value === 'number' ? value.toFixed(2) : value) : 'N/A'}
                                          {unit ? <span className="text-sm font-normal ml-1">({unit})</span> : ''}
                                      </p>
                                  </div>
                              );
                          })}
                      </div>
                  </>
              )}
               {latestDataPoint && availableFeatures.length === 0 && (
                   <div className="text-sm text-center text-gray-500 mt-4 border-t pt-4">Không có dữ liệu chi tiết cho các chỉ số thành phần.</div>
               )}
          </CardContent>

          <CardFooter className="bg-gray-50 border-t border-gray-200 py-2 px-4 text-xs text-gray-600 flex items-center justify-end">
               <Clock className="w-4 h-4 mr-1 flex-shrink-0" />
               {/* Sử dụng hàm formatFooterTime cho createdAt */}
               <span>Cập nhật lần cuối: {formatFooterTime(selectedStation.createdAt)}</span>
          </CardFooter>
      </Card>
  );
}