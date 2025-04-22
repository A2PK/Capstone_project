import {
  Card,
  CardHeader,
  CardContent,
  CardFooter
} from "@/components/ui/card"; // Adjust path
import { Badge } from "@/components/ui/badge"; // Adjust path
import {
  AlertTriangle, Activity, Clock, MapPin, Thermometer, Droplet, Zap, Wind,
  Cloudy, FlaskConical, Beaker, TestTube, HelpCircle, Waves, Gauge, Info // Thêm các icon cần thiết
} from "lucide-react";
import { Station as BaseStation, DataPoint, Indicator } from "@/types/station2"; // Import thêm DataPoint, Indicator - Adjust path
import { getDonvi } from "@/lib/utils";

// Interface cho prop `selectedStation` đã được kết hợp thông tin
interface CombinedStationInfo extends BaseStation {
  wqi: number | string | null; // Có thể là số, string "N/A", hoặc null
  status: string;
  time: string;
  recommendation: string;
}

// Cập nhật interface Props để nhận thêm latestDataPoint và availableFeatures
interface StationDetailsProps {
  selectedStation: CombinedStationInfo | null;
  latestDataPoint: DataPoint | null;
  availableFeatures: string[];
}

// --- Helper Function: Get Icon for Feature ---
// Trả về component Icon từ Lucide dựa trên tên chỉ số
function getIconForFeature(featureName: string) {
    switch (featureName.toUpperCase()) {
        case 'PH': return Droplet;       // Biểu tượng cho pH (độ axit/bazơ)
        case 'DO': return Waves;         // Biểu tượng cho Oxy hòa tan
        case 'TSS': return Cloudy;       // Biểu tượng cho Tổng chất rắn lơ lửng (độ đục)
        case 'COD': return FlaskConical; // Biểu tượng cho Nhu cầu Oxy hóa học
        case 'EC': return Zap;           // Biểu tượng cho Độ dẫn điện
        case 'TEMP': return Thermometer; // Biểu tượng cho Nhiệt độ (nếu có)
        case 'N-NO2': return Beaker;      // Biểu tượng cho Nitrit
        case 'N-NH4': return FlaskConical; // Biểu tượng cho Amoni (dùng icon khác Beaker)
        case 'P-PO4': return TestTube;    // Biểu tượng cho Phosphat (dùng icon khác nữa)
        case 'AH': return HelpCircle;    // Biểu tượng cho chỉ số chưa rõ (ví dụ: Available Head?)
        case 'WQI': return Gauge;         // Biểu tượng cho chỉ số tổng hợp WQI
        // Thêm các case khác cho các chỉ số của bạn ở đây
        // Ví dụ: case 'SALINITY': return SaltIcon; (nếu có icon phù hợp)
        default: return Activity;      // Icon mặc định nếu không khớp
    }
}
// -----------------------------------------

export default function StationDetails({
  selectedStation,
  latestDataPoint,
  availableFeatures
}: StationDetailsProps) {

  if (!selectedStation) {
    return (
      <div className="text-center text-gray-500 p-4 mt-4 bg-gray-50 border rounded-lg shadow-sm">
        Chọn một trạm trên bản đồ hoặc từ bảng để xem chi tiết.
      </div>
    );
  }

  // Hàm lấy màu Tailwind dựa trên trạng thái (giữ nguyên)
  const getStatusTailwindClasses = (status: string): string => {
    switch (status) {
      case "Rất Tốt":
        return "bg-emerald-100 text-emerald-900 border-emerald-200";
      case "Tốt":
        return "bg-green-100 text-green-800 border-green-200";
      case "Trung Bình":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "Kém":
        return "bg-orange-100 text-orange-800 border-orange-200";
      case "Rất Kém":
        return "bg-red-100 text-red-800 border-red-200";
      case "Không xác định":
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  // Helper để lấy giá trị của một chỉ số cụ thể từ latestDataPoint (giữ nguyên)
  const getFeatureValue = (featureName: string): string | number | null => {
      if (!latestDataPoint || !latestDataPoint.features) return null;
      const feature = latestDataPoint.features.find(f => f.name === featureName);
      if (!feature) return null;
      // Ưu tiên textualValue nếu là số hợp lệ
      if (typeof feature.textualValue === 'string' && feature.textualValue.trim() !== '') {
          const parsed = parseFloat(feature.textualValue);
          if (!isNaN(parsed)) return parsed;
      }
      // Sau đó dùng value nếu là số hợp lệ
      if (typeof feature.value === 'number' && !isNaN(feature.value)) {
          return feature.value;
      }
       // Nếu không có giá trị số, trả về textualValue (có thể là text như "Bình thường")
      if(feature.textualValue) return feature.textualValue;

      return null; // Hoặc 'N/A'
  }

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
          <Badge
            variant="outline"
            className={`${getStatusTailwindClasses(
              selectedStation.status
            )} px-3 py-1 text-sm font-medium rounded-md whitespace-nowrap self-center`}
          >
            {selectedStation.status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="p-4">
        {/* Phần WQI và Khuyến cáo */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 flex flex-col justify-between">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-5 h-5 text-blue-700 flex-shrink-0" />
              <h3 className="font-semibold text-blue-800">
                Chỉ số chất lượng nước (WQI)
              </h3>
            </div>
            <div className="text-right">
              <span className="text-4xl font-bold text-blue-700">
                {selectedStation.wqi ?? "--"}
              </span>
            </div>
          </div>

          <div className="bg-amber-50 p-4 rounded-lg border border-amber-100">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-amber-700 flex-shrink-0" />
              <h3 className="font-semibold text-amber-800">Khuyến cáo</h3>
            </div>
            <p className="text-sm text-gray-800 leading-relaxed">
              {selectedStation.recommendation}
            </p>
          </div>
        </div>
            
        {/* Phần hiển thị các chỉ số chi tiết (lúc {selectedStation.time})*/}
        {latestDataPoint && availableFeatures.length > 0 && (
          <>
            <h4 className="text-md font-semibold text-gray-700 mb-3 mt-5 border-t pt-4">Các chỉ số thành phần </h4> 
            <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {availableFeatures.map(featureName => {
                 const value = getFeatureValue(featureName);
                 // Lấy component Icon tương ứng
                 const IconComponent = getIconForFeature(featureName);
                 const unit = getDonvi(featureName);
                 return (
                   <div key={featureName} className="bg-gray-100 p-3 rounded-md border border-gray-200 hover:shadow-sm transition-shadow duration-150">
                      <div className="flex items-center justify-center text-gray-600 mb-1">
                         {/* Render component Icon */}
                         <IconComponent className="w-4 h-4 mr-1.5 flex-shrink-0" />
                         <p className="text-xs font-semibold uppercase tracking-wide">{featureName} </p> 
                      </div>
                      <p className="text-xl font-bold text-gray-900 text-center">{value !== null ? value : 'N/A'} {unit ? `  (${unit})` : ''}</p> 
                   </div>
                 );
              })}
            </div>
          </>
        )}
        {/* Thông báo nếu không có chỉ số thành phần nào */}
         {latestDataPoint && availableFeatures.length === 0 && (
             <div className="text-sm text-center text-gray-500 mt-4 border-t pt-4">Không có dữ liệu chi tiết cho các chỉ số thành phần tại thời điểm này.</div>
         )}
      </CardContent>

      <CardFooter className="bg-gray-50 border-t border-gray-200 py-2 px-4 text-xs text-gray-600 flex items-center justify-end">
          <Clock className="w-4 h-4 mr-1 flex-shrink-0" />
          <span>Dữ liệu lúc: {selectedStation.time !== 'N/A' ? selectedStation.time : 'Chưa có'}</span>
      </CardFooter>
    </Card>
  );
}