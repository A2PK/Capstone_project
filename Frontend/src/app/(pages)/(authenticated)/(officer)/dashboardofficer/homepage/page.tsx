"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import "leaflet/dist/leaflet.css";
import { MapPin, Droplet, AlertTriangle, ArrowRight, ServerCrash, CheckCircle, TrendingUp, TrendingDown } from "lucide-react"; // Thêm icon xu hướng nếu muốn
import PageLoader from "@/components/pageloader";
import { cn } from "@/lib/utils";

// --- Types (Nên định nghĩa rõ ràng hơn) ---
interface FeatureData {
  name: string;
  trend: number[];
  prediction: number[];
}

interface Station {
  name: string;
  lat: number;
  lng: number;
  wqi: number;
  status: "Nguy hiểm" | "Kém" | "Trung bình" | "Tốt" | "Rất tốt" | string; // Mở rộng Status
  recommendation: string;
  time: string;
  trend: number[];
  prediction: number[];
  feature: FeatureData[];
}

// --- Dynamic Imports cho Map ---
const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(mod => mod.Popup), { ssr: false });
const MarkerClusterGroup = dynamic(() => import("react-leaflet-cluster"), { ssr: false });

// --- Dữ liệu mẫu mới ---
const stations: Station[] = [
 {
   name: "Tân An",
   lat: 10.535,
   lng: 106.413,
   wqi: 37,
   status: "Nguy hiểm",
   recommendation: "WQI nguy hiểm, cần xử lý nguồn nước khẩn cấp",
   time: "10:20, Thg 3, 2025",
   trend: [25, 30, 34, 37, 33, 39, 42, 40, 45, 47, 44, 49, 46, 50, 48],
   prediction: [50, 53, 55, 58, 60, 62, 65, 68, 70, 72, 74, 75, 78, 80, 82],
   feature: [ { name: "pH", trend: [6.0, 6.1, 6.3, 6.2, 6.4, 6.5, 6.3, 6.6, 6.7, 6.8, 6.5, 6.9, 7.0, 6.8, 7.1], prediction: [7.0, 7.1, 7.2, 7.3, 7.2, 7.4, 7.5, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.0, 8.1], }, { name: "NH4", trend: [1.2, 1.5, 1.8, 2.0, 1.7, 2.2, 2.4, 2.1, 2.5, 2.7, 2.6, 2.9, 3.0, 2.8, 3.1], prediction: [3.0, 3.2, 3.3, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.8], }, { name: "DO", trend: [4.5, 4.3, 4.0, 3.8, 4.1, 3.9, 3.7, 4.0, 4.2, 4.3, 4.1, 4.4, 4.5, 4.6, 4.7], prediction: [4.8, 4.9, 5.0, 5.1, 5.0, 5.2, 5.3, 5.4, 5.4, 5.5, 5.6, 5.6, 5.7, 5.8, 6.0], }, ],
 },
 {
   name: "Mỹ Tho",
   lat: 10.36,
   lng: 106.365,
   wqi: 78,
   status: "Tốt",
   recommendation: "WQI tốt, cần theo dõi thường xuyên",
   time: "10:20, Thg 3, 2025",
   trend: [68, 70, 72, 74, 73, 76, 78, 77, 80, 82, 81, 84, 85, 87, 89],
   prediction: [90, 91, 92, 93, 94, 95, 96, 97, 97, 98, 99, 99, 100, 100, 100],
   feature: [ { name: "pH", trend: [6.8, 6.9, 7.0, 7.1, 7.0, 7.2, 7.3, 7.2, 7.4, 7.5, 7.4, 7.6, 7.7, 7.8, 7.9], prediction: [8.0, 8.0, 8.1, 8.2, 8.2, 8.3, 8.4, 8.4, 8.5, 8.6, 8.6, 8.7, 8.8, 8.8, 8.9], }, { name: "NH4", trend: [0.8, 0.9, 1.0, 1.1, 1.0, 1.2, 1.3, 1.2, 1.4, 1.5, 1.4, 1.6, 1.7, 1.8, 1.9], prediction: [1.9, 2.0, 2.1, 2.2, 2.2, 2.3, 2.4, 2.5, 2.6, 2.6, 2.7, 2.8, 2.9, 3.0, 3.0], }, { name: "DO", trend: [6.0, 6.2, 6.3, 6.4, 6.3, 6.5, 6.6, 6.7, 6.8, 6.8, 6.9, 7.0, 7.1, 7.1, 7.2], prediction: [7.2, 7.3, 7.4, 7.5, 7.5, 7.6, 7.7, 7.7, 7.8, 7.9, 7.9, 8.0, 8.1, 8.2, 8.2], }, ],
 },
 // ... (các trạm khác giữ nguyên cấu trúc tương tự) ...
 {
   name: "Trà Vinh",
   lat: 9.934,
   lng: 106.342,
   wqi: 62,
   status: "Trung bình",
   recommendation: "WQI trung bình, cần giám sát thêm",
   time: "10:18, Thg 3, 2025",
   trend: [50, 52, 55, 57, 60, 58, 62, 63, 65, 67, 68, 70, 72, 73, 75],
   prediction: [75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
   feature: [ /* ... feature data ... */ ],
 },
  {
    name: "Long Xuyên",
    lat: 10.371,
    lng: 105.432,
    wqi: 45,
    status: "Kém", // Thêm status mới
    recommendation: "WQI thấp, cần xử lý trước khi sử dụng",
    time: "10:18, Thg 3, 2025",
    trend: [40, 42, 43, 44, 45, 46, 47, 47, 48, 49, 50, 51, 51, 52, 53],
    prediction: [53, 54, 55, 55, 56, 57, 58, 58, 59, 60, 61, 62, 63, 64, 65],
    feature: [ /* ... feature data ... */ ],
  },
];


// --- Code cho Map Icons (giữ nguyên) ---
let redIcon: any, blueIcon: any, orangeIcon: any, greenIcon: any, createClusterCustomIcon: any; // Thêm orange, green

if (typeof window !== "undefined") {
  const { Icon, divIcon, point } = require("leaflet");

  // Định nghĩa các màu icon khác nhau nếu muốn map trực quan hơn
  redIcon = new Icon({ iconUrl: "/red_one.png", iconSize: [38, 38], }); // Nguy hiểm
  orangeIcon = new Icon({ iconUrl: "/orange_one.png", iconSize: [38, 38], }); // Kém (cần tạo file /orange_one.png)
  blueIcon = new Icon({ iconUrl: "/blue_one.png", iconSize: [38, 38], }); // Trung bình/Tốt (ví dụ)
  greenIcon = new Icon({ iconUrl: "/green_one.png", iconSize: [38, 38], }); // Rất tốt (cần tạo file /green_one.png)


  createClusterCustomIcon = (cluster: any) => { /* ... giữ nguyên ... */
    return divIcon({
     html: `<span style="background-color: #333; height: 2em; width: 2em; color: #fff; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 1.2rem; box-shadow: 0 0 0px 5px #fff;">${cluster.getChildCount()}</span>`,
     className: "custom-marker-cluster",
     iconSize: point(33, 33, true),
   });
  };
}

// --- Component chính ---
export default function StationsPage() {
  const [isLoading, setIsLoading] = useState(true);
  useEffect(() => {
    const timeout = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    return () => clearTimeout(timeout);
  }, []);

  const [selectedStation, setSelectedStation] = useState<Station | null>(null); // Có thể null ban đầu
  const [selectedMarker, setSelectedMarker] = useState<string | null>(null);

  // --- Lọc các trạm cần cảnh báo (Thêm "Kém") ---
  const warningStations = stations.filter(
    station => station.status === "Nguy hiểm" || station.status === "Trung bình" || station.status === "Kém"
  );

  // --- Hàm lấy màu và icon dựa trên status ---
  const getStatusStyle = (status: string): { color: string; border: string; bg: string; icon?: any } => {
    switch (status) {
      case "Nguy hiểm": return { color: "text-red-600", border: "border-red-500", bg: "bg-red-50", icon: redIcon };
      case "Kém": return { color: "text-orange-600", border: "border-orange-500", bg: "bg-orange-50", icon: orangeIcon }; // Thêm Kém
      case "Trung bình": return { color: "text-yellow-600", border: "border-yellow-500", bg: "bg-yellow-50", icon: blueIcon }; // Ví dụ icon
      case "Tốt": return { color: "text-green-600", border: "border-green-500", bg: "bg-green-50", icon: blueIcon }; // Ví dụ icon
      case "Rất tốt": return { color: "text-blue-600", border: "border-blue-500", bg: "bg-blue-50", icon: greenIcon };
      default: return { color: "text-gray-600", border: "border-gray-500", bg: "bg-gray-50", icon: blueIcon };
    }
  };


  if (isLoading) return <PageLoader message="Đang tải trang chủ..." />;

  return (
    <div className="container mx-auto px-4 py-6 space-y-8">

      {/* === Dashboard Cảnh báo (Cập nhật) === */}
      <section>
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <AlertTriangle className="w-6 h-6 mr-2 text-yellow-500" />
          Dashboard Cảnh báo
        </h2>
        {warningStations.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"> {/* Thêm xl breakpoint nếu cần */}
            {warningStations.map((station, index) => {
              const statusStyle = getStatusStyle(station.status);
              // Lấy WQI trước đó để xem xu hướng (nếu có đủ dữ liệu)
              const previousWQI = station.trend.length > 1 ? station.trend[station.trend.length - 2] : null;
              const trendIcon = previousWQI !== null ? (station.wqi > previousWQI ? <TrendingUp className="w-4 h-4 text-green-500" /> : (station.wqi < previousWQI ? <TrendingDown className="w-4 h-4 text-red-500" /> : null) ) : null;

              return (
                <Card key={index} className={cn("hover:shadow-md transition-shadow border-l-4 flex flex-col h-full", statusStyle.border)}> {/* Đảm bảo card có chiều cao bằng nhau */}
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-start gap-2">
                        <CardTitle className="text-lg font-semibold flex items-center">
                          <MapPin className="w-5 h-5 mr-2 text-gray-600 flex-shrink-0" />
                          {station.name}
                        </CardTitle>
                        <span className={cn("text-xs font-bold px-2 py-0.5 rounded-full whitespace-nowrap", statusStyle.color, statusStyle.bg, statusStyle.border.replace('border-l-4','border'))}>
                            {station.status}
                        </span>
                    </div>
                    <CardDescription className="text-xs pt-1">{station.time}</CardDescription>
                  </CardHeader>
                  <CardContent className="flex-grow space-y-2 pb-3"> {/* flex-grow để nội dung chiếm không gian */}
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Chỉ số WQI:</span>
                        <div className="flex items-center gap-1">
                           {/* Hiển thị icon xu hướng nếu có */}
                           {/* {trendIcon} */}
                           <strong className={cn("text-3xl font-bold", statusStyle.color)}>{station.wqi}</strong>
                        </div>
                    </div>
                    <p className="text-sm text-gray-700 italic">
                       <span className="font-medium text-gray-800">Khuyến nghị:</span> "{station.recommendation}"
                    </p>
                  </CardContent>
                  <CardFooter className="pt-2 pb-3 border-t border-gray-200 mt-auto"> {/* Đẩy footer xuống dưới */}
                     <Button asChild variant="link" size="sm" className="p-0 h-auto text-blue-600 hover:text-blue-800 font-medium">
                          <Link href={`/dashboardofficer/stations`}> {/* Cần trang chi tiết để xem trend/prediction */}
                            Xem chi tiết
                            <ArrowRight className="w-4 h-4 ml-1" />
                          </Link>
                     </Button>
                  </CardFooter>
                </Card>
              );
            })}
          </div>
        ) : (
          <Card className="border-green-500 bg-green-50 border-l-4">
             {/* ... Nội dung không có cảnh báo giữ nguyên ... */}
             <CardHeader className="flex flex-row items-center space-x-3 pb-4">
                <CheckCircle className="w-6 h-6 text-green-600"/>
                <CardTitle className="text-green-700">Không có cảnh báo</CardTitle>
            </CardHeader>
            <CardContent>
                <p className="text-sm text-green-800">Tất cả các trạm hiện đang có chỉ số ổn định.</p>
            </CardContent>
          </Card>
        )}
      </section>

      {/* === Bản đồ địa lý (Cập nhật icon) === */}
      <section>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Bản đồ Trạm quan trắc</h2>
        <div className="relative w-full h-[500px] border border-gray-300 rounded-lg overflow-hidden shadow-md">
          <MapContainer center={[10.3, 106.1]} zoom={9} style={{ height: "100%", width: "100%" }}>
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {createClusterCustomIcon && (
              <MarkerClusterGroup chunkedLoading iconCreateFunction={createClusterCustomIcon}>
                {stations.map((station, index) => {
                   const statusStyle = getStatusStyle(station.status); // Lấy style (bao gồm cả icon)
                   return (
                      <Marker
                        key={index}
                        position={[station.lat, station.lng]}
                        icon={statusStyle.icon || blueIcon} // Sử dụng icon từ style, fallback về blueIcon
                        eventHandlers={{
                          click: () => {
                            setSelectedStation(station);
                            setSelectedMarker(station.name);
                          }
                        }}
                      >
                        <Popup>
                          {/* Popup content giữ nguyên hoặc tùy chỉnh */}
                           <div className="p-2 rounded-lg bg-white max-w-xs min-w-[200px] border border-gray-200 shadow-sm">
                              {/* ... nội dung popup như cũ ... */}
                              <div className="flex items-center mb-1">
                                <MapPin className="w-4 h-4 text-blue-500 mr-1 flex-shrink-0"/>
                                <h3 className="text-base font-bold text-gray-800">{station.name}</h3>
                              </div>
                              <div className="text-xs space-y-0.5">
                                  <div className="flex items-center">
                                    <Droplet className="w-3 h-3 text-blue-500 mr-1.5 flex-shrink-0"/>
                                    <span>WQI: <strong className={cn(statusStyle.color)}>{station.wqi}</strong></span>
                                  </div>
                                  <div className="flex items-center">
                                      <AlertTriangle className={cn("w-3 h-3 mr-1.5 flex-shrink-0", statusStyle.color)} />
                                      <span>Trạng thái:{" "}
                                          <span className={cn("font-semibold", statusStyle.color)}>
                                              {station.status}
                                          </span>
                                      </span>
                                  </div>
                                  <p className="text-gray-500 pt-0.5">{station.time}</p>
                              </div>
                              <Button asChild size="sm" className="mt-2 w-full h-8 text-white bg-blue-600 hover:bg-blue-700">
                                <Link href={`/dashboardofficer/stations`}>
                                  Xem chi tiết
                                  <ArrowRight className="w-4 h-4 ml-1"/>
                                </Link>
                              </Button>
                           </div>
                        </Popup>
                      </Marker>
                   );
                })}
              </MarkerClusterGroup>
            )}
          </MapContainer>
        </div>
      </section>

    </div>
  );
}