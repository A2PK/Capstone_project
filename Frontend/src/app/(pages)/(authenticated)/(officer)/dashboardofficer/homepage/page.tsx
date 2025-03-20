"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import "leaflet/dist/leaflet.css";
import { MapPin, Droplet, AlertTriangle, ArrowRight } from "lucide-react";
import PageLoader from "@/components/pageloader";

const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(mod => mod.Popup), { ssr: false });
const MarkerClusterGroup = dynamic(() => import("react-leaflet-cluster"), { ssr: false });

const stations = [
  { name: "Tân An", lat: 10.535, lng: 106.413, wqi: 35, status: "Nguy hiểm", recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp", time: "12:32, Thg 9, 2024" },
  { name: "Mỹ Tho", lat: 10.36, lng: 106.365, wqi: 80, status: "Tốt", recommendation: "WQI tốt, chú ý lọc nước", time: "12:32, Thg 9, 2024" },
  { name: "Bến Tre", lat: 10.241, lng: 106.375, wqi: 90, status: "Rất tốt", recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm", time: "12:31, Thg 9, 2024" },
  { name: "Vĩnh Long", lat: 10.253, lng: 105.972, wqi: 75, status: "Tốt", recommendation: "WQI tốt, có thể sử dụng nước sinh hoạt", time: "12:29, Thg 9, 2024" },
  { name: "Cần Thơ",
    lat: 10.033,
    lng: 105.783,
    wqi: 88,
    status: "Rất tốt",
    recommendation: "WQI rất tốt, đảm bảo an toàn sử dụng",
    time: "10:18, Thg 3, 2025" },
    { name: "Trà Vinh",
      lat: 9.934,
      lng: 106.342,
      wqi: 62,
      status: "Trung bình",
      recommendation: "WQI trung bình, cần giám sát thêm",
      time: "10:18, Thg 3, 2025" }
];

const articles = [
  {
    title: "Bản tin khẩn chất lượng nước sông Tiền !!",
    date: "4 Nov 2024",
    description: "Dữ liệu quan trắc được từ trạm sông Tiền cho thấy rằng chất lượng nước nơi này đang có sự ô nhiễm nhẹ...",
    imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
  },
  {
    title: "Báo cáo hàng tuần về chất lượng nước sông Hậu!!",
    date: "4 Nov 2024",
    description: "Dữ liệu quan trắc được từ trạm sông Hồng cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
    imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
  },
  {
    title: "Báo cáo hàng tuần về chất lượng nước sông Cần!!",
    date: "4 Nov 2024",
    description: "Dữ liệu quan trắc được từ trạm sông Cần cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
    imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
  },
];

let redIcon: any, blueIcon: any, createClusterCustomIcon: any;

if (typeof window !== "undefined") {
  const { Icon, divIcon, point } = require("leaflet");

  redIcon = new Icon({
    iconUrl: "/red_one.png",
    iconSize: [38, 38],
  });

  blueIcon = new Icon({
    iconUrl: "/blue_one.png",
    iconSize: [38, 38],
  });

  createClusterCustomIcon = (cluster: any) => {
    return divIcon({
      html: `<span style="background-color: #333; height: 2em; width: 2em; color: #fff; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 1.2rem; box-shadow: 0 0 0px 5px #fff;">${cluster.getChildCount()}</span>`,
      className: "custom-marker-cluster",
      iconSize: point(33, 33, true),
    });
  };
}

export default function StationsPage() {
  const [isLoading, setIsLoading] = useState(true);
      useEffect(() => {
              // Simulate loading delay (e.g., fetching data)
              const timeout = setTimeout(() => {
                setIsLoading(false);
              }, 1000); // 1.5s delay
              return () => clearTimeout(timeout);
            }, []);
          
  const [selectedStation, setSelectedStation] = useState(stations[0]);
  const [selectedMarker, setSelectedMarker] = useState<string | null>(null);
if (isLoading) return <PageLoader message="Đang tải trang chủ..." />;
  return (
    <div className="container mx-auto px-4 py-6 space-y-6">
      <section>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Bài viết liên quan</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {articles.map((article, index) => (
            <Link key={index} href="/dashboardofficer/newsdetail" className="block h-full">
              <Card className="hover:shadow-lg transition-shadow flex flex-col h-full">
                <img
                  src={article.imageUrl}
                  alt={article.title}
                  className="w-full h-64 object-cover rounded-t-md"
                />
                <CardHeader className="flex-grow">
                  <CardTitle className="text-xl font-bold">{article.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-2">{article.description}</p>
                  <p className="text-xs text-gray-500">{article.date}</p>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
        <div className="flex justify-center mt-6">
          <Link href="/dashboardofficer/newspaper">
            <Button className="px-6 py-2 text-white bg-black hover:bg-gray-800">
              Xem thêm
            </Button>
          </Link>
        </div>
      </section>
      <section>
        <h2 className="text-3xl font-bold text-orange-500 mb-6">Bản đồ địa lý</h2>
        <div className="relative w-full h-[500px] border border-gray-300 rounded-md overflow-hidden">
          <MapContainer center={[10.535, 106.413]} zoom={10} style={{ height: "100%", width: "100%" }}>
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {createClusterCustomIcon && (
              <MarkerClusterGroup iconCreateFunction={createClusterCustomIcon}>
                {stations.map((station, index) => (
                  <Marker
                    key={index}
                    position={[station.lat, station.lng]}
                    icon={selectedMarker === station.name ? blueIcon : redIcon} // Chỉ đổi màu khi click
                    eventHandlers={{
                      click: () => {
                        setSelectedStation(station);
                        setSelectedMarker(station.name); // Giữ màu xanh sau khi click
                      }
                    }}
                  >
                    <Popup>
                      <div className="p-2 rounded-lg bg-white max-w-xs min-w-[200px] md:min-w-[200px] border border-gray-200 transition-all duration-300">
                        <div className="flex items-center mb-1">
                          <MapPin className="w-4 h-4 text-blue-500 mr-1" />
                          <h3 className="text-lg font-bold text-gray-800">{station.name}</h3>
                        </div>
                        <div className="flex flex-col gap-y-0 leading-tight">
                          <div className="flex items-center -mb-1">
                            <Droplet className="w-3 h-3 text-blue-500 mr-1" />
                            <p className="text-sm text-gray-600">
                              WQI: <span className="font-bold text-blue-500">{station.wqi}</span>
                            </p>
                          </div>
                          <div className="flex items-center -mt-1">
                            <AlertTriangle className="w-3 h-3 text-yellow-500 mr-1" />
                            <p className="text-sm text-gray-600">
                              Trạng thái:{" "}
                              <span
                                className={`font-semibold ${station.status === "Nguy hiểm"
                                  ? "text-red-500"
                                  : station.status === "Tốt"
                                    ? "text-green-500"
                                    : "text-blue-500"
                                  }`}
                              >
                                {station.status}
                              </span>
                            </p>
                          </div>
                          <p className="text-[10px] text-gray-500 -mt-1">{station.time}</p>
                        </div>
                        <Button
                          asChild
                          className="mt-1 px-2 py-1 text-white bg-blue-700 hover:bg-blue-600 border border-black-600"
                        >
                          <Link href={`/dashboardofficer/stations`}>
                            <span className="text-white flex items-center justify-center">
                              Xem chi tiết
                              <ArrowRight className="w-3 h-3 ml-1 transition-transform duration-300" />
                            </span>
                          </Link>
                        </Button>
                      </div>
                    </Popup>
                  </Marker>
                ))}
              </MarkerClusterGroup>
            )}
          </MapContainer>
        </div>
      </section>
    </div>
  );
}

