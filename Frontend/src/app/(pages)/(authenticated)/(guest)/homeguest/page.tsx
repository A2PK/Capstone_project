// Assuming this file is at: src/app/(pages)/(authenticated)/(guest)/homeguest/page.tsx
"use client";

import React, { useState, useEffect, useMemo } from "react";
import dynamic from "next/dynamic"; // For client-side only components like Leaflet
import Link from "next/link";
import Image from "next/image"; // Import Next Image
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
// REMOVE Leaflet CSS import from top: import "leaflet/dist/leaflet.css";
// REMOVE Leaflet core import from top: import L, { Icon, DivIcon, PointTuple, point as leafletPoint, LatLngExpression } from 'leaflet';
import { MapPin, AlertTriangle, ArrowRight, ServerCrash, CheckCircle, Droplet } from "lucide-react";
import PageLoader from "@/components/pageloader"; // Your loader component
import { cn } from "@/lib/utils"; // Utility for class names
import {
    Station,
    DataPoint,
    ApiRequestDataPointsByStationId, // Use this for filtering
} from "@/types/station2";
import { getStations, getAllDataPointsByStationID } from "@/lib/station"; // Use the correct API function
import { format, parseISO, isValid as isValidDate } from "date-fns"; // For date formatting

// --- Dynamic Imports for React-Leaflet Components (Keep these) ---
const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(mod => mod.Popup), { ssr: false });
const MarkerClusterGroup = dynamic(() => import("react-leaflet-cluster"), { ssr: false });

// Type for Leaflet assets state (using 'any' for simplicity with dynamic import)
type LeafletAssets = {
    redIcon: any;
    blueIcon: any;
    createClusterCustomIcon: (cluster: any) => any; // Leaflet's DivIcon type comes from L
};

// Type for map center/position
type LeafletLatLngExpression = [number, number];

// --- Helper Functions (Keep as before) ---
function deriveStatusFromWqi(wqi: number | null | undefined): string {
    if (wqi === null || wqi === undefined || isNaN(wqi)) return "Không xác định";
    if (wqi > 91) return "Rất Tốt";
    if (wqi > 76) return "Tốt";
    if (wqi > 51) return "Trung Bình";
    if (wqi > 26) return "Kém";
    if (wqi >= 0) return "Rất Kém";
    return "Không xác định";
}
const getStatusTailwindClasses = (status: string): { bg: string; text: string; border: string; iconColor: string } => {
    switch (status) {
        case "Rất Tốt": return { bg: "bg-emerald-50", text: "text-emerald-800", border: "border-emerald-400", iconColor: "text-emerald-500" };
        case "Tốt": return { bg: "bg-green-50", text: "text-green-800", border: "border-green-400", iconColor: "text-green-500" };
        case "Trung Bình": return { bg: "bg-yellow-50", text: "text-yellow-800", border: "border-yellow-400", iconColor: "text-yellow-500" };
        case "Kém": return { bg: "bg-orange-50", text: "text-orange-800", border: "border-orange-400", iconColor: "text-orange-500" };
        case "Rất Kém": return { bg: "bg-red-50", text: "text-red-800", border: "border-red-400", iconColor: "text-red-500" };
        default: return { bg: "bg-gray-100", text: "text-gray-800", border: "border-gray-400", iconColor: "text-gray-500" };
    }
};
function formatMonitoringTime(timeString: string | undefined | null): string {
    if (!timeString) return "N/A";
    try {
        const date = parseISO(timeString);
        if (!isValidDate(date)) return timeString;
        return format(date, 'HH:mm, dd/MM/yyyy');
    } catch (e) { return timeString; }
}

interface StationDisplayData extends Station {
    latestData: (DataPoint & { status: string }) | null;
    isLoadingData: boolean;
    errorData: string | null;
}

// --- Mock Data for News Section (Keep as before) ---
const articles = [
     { title: "Bản tin khẩn chất lượng nước sông Tiền !!", date: "4 Nov 2024", description: "Dữ liệu quan trắc được từ trạm sông Tiền cho thấy rằng chất lượng nước nơi này đang có sự ô nhiễm nhẹ...", imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg" },
     { title: "Báo cáo hàng tuần về chất lượng nước sông Hậu!!", date: "4 Nov 2024", description: "Dữ liệu quan trắc được từ trạm sông Hồng cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...", imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg" },
     { title: "Báo cáo hàng tuần về chất lượng nước sông Cần!!", date: "4 Nov 2024", description: "Dữ liệu quan trắc được từ trạm sông Cần cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...", imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg" },
];

// --- Main Component ---
export default function HomeGuestPage() {
    // --- State ---
    const [stationDisplayData, setStationDisplayData] = useState<StationDisplayData[]>([]);
    const [selectedStationId, setSelectedStationId] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isClient, setIsClient] = useState(false); // Track client-side
    const [leafletAssets, setLeafletAssets] = useState<LeafletAssets | null>(null); // Hold Leaflet assets

    // --- Constants ---
    const initialMapCenter: LeafletLatLngExpression = [10.8231, 106.6297];
    const initialMapZoom = 10;

    // Client-side detection effect
    useEffect(() => {
        setIsClient(true);
    }, []);

    // Leaflet initialization effect - ONLY runs client-side and dynamically imports L and CSS
     useEffect(() => {
        // Only run this effect if we are on the client
        if (isClient) {
            // Use Promise.all to load both Leaflet JS and CSS dynamically
            Promise.all([
                import('leaflet'),
                import('leaflet/dist/leaflet.css' as any) // Dynamically import CSS (may cause FOUC)
            ]).then(([L]) => { // Destructure L from the resolved promise array
                // Now L (Leaflet library) is available

                // Create icons using the dynamically imported L
                const redIcon = new L.Icon({
                    iconUrl: "/red_one.png",
                    iconSize: [38, 38],
                    iconAnchor: [19, 38],
                    popupAnchor: [0, -38],
                });

                const blueIcon = new L.Icon({
                    iconUrl: "/blue_one.png",
                    iconSize: [38, 38],
                    iconAnchor: [19, 38],
                    popupAnchor: [0, -38],
                });

                const createClusterCustomIcon = (cluster: any) => {
                    const count = cluster.getChildCount();
                    const style = `
                        background-color: rgba(51, 51, 51, 0.8); height: 2.5em; width: 2.5em;
                        color: #fff; display: flex; align-items: center; justify-content: center;
                        border-radius: 50%; font-size: 1rem; font-weight: bold;
                        box-shadow: 0 0 5px rgba(0,0,0,0.5); border: 2px solid #fff;
                    `;
                    // Use L.point from the dynamically imported L
                    return L.divIcon({
                        html: `<span style="${style}">${count}</span>`,
                        className: "custom-marker-cluster",
                        iconSize: L.point(40, 40), // Use L.point here
                    });
                };

                // Set the created assets into state
                setLeafletAssets({ redIcon, blueIcon, createClusterCustomIcon });

            }).catch(err => {
                console.error("Failed to load Leaflet dynamically:", err);
                // Optionally set an error state specific to the map
            });
        }
    }, [isClient]); // Rerun if isClient changes (runs once client-side)


    // --- Data Fetching Effect (Keep as before) ---
    useEffect(() => {
        const fetchInitialData = async () => {
            setIsLoading(true);
            setError(null);
            setStationDisplayData([]);
            try {
                const stations = await getStations({ limit: 1000 });
                const initialDisplayData: StationDisplayData[] = stations.map(station => ({
                    ...station, latestData: null, isLoadingData: true, errorData: null,
                }));
                setStationDisplayData(initialDisplayData);

                const dataPointPromises = stations.map(async (station) => {
                    try {
                        const apiRequestOptions: ApiRequestDataPointsByStationId = {
                            stationId: station.id,
                            options: { limit: 1, sortBy: 'monitoring_time', sortDesc: true, filters: { observation_type: 'actual' } }
                        };
                        const dataPoints = await getAllDataPointsByStationID(apiRequestOptions);
                        if (dataPoints.length > 0) {
                            const latestPoint = dataPoints[0];
                            const status = deriveStatusFromWqi(latestPoint.wqi);
                            return { stationId: station.id, latestData: { ...latestPoint, status }, errorData: null };
                        } else {
                            return { stationId: station.id, latestData: null, errorData: null };
                        }
                    } catch (err) {
                        console.error(`Failed fetch data for station ${station.id}:`, err);
                        return { stationId: station.id, latestData: null, errorData: "Lỗi tải dữ liệu điểm đo" };
                    }
                });
                const results = await Promise.allSettled(dataPointPromises);
                setStationDisplayData(prevData => {
                    const newDataMap = new Map(prevData.map(s => [s.id, s]));
                    results.forEach(result => {
                        if (result.status === 'fulfilled' && result.value) {
                            const { stationId, latestData, errorData } = result.value;
                            const station = newDataMap.get(stationId);
                            if (station) {
                                station.latestData = latestData;
                                station.isLoadingData = false;
                                station.errorData = errorData;
                            }
                        }
                    });
                    return Array.from(newDataMap.values());
                });
            } catch (err) {
                console.error("Failed to fetch stations or process data:", err);
                setError(err instanceof Error ? err.message : "Lỗi không xác định khi tải trang.");
            } finally {
                setIsLoading(false);
            }
        };
        fetchInitialData();
    }, []);

    // --- Memoized Data Derivations (Keep as before) ---
    const uniqueStationsForMap = useMemo(() => {
        const coordMap = new Map<string, StationDisplayData>();
        stationDisplayData.forEach((station) => {
            if (typeof station.latitude === 'number' && typeof station.longitude === 'number' && !isNaN(station.latitude) && !isNaN(station.longitude)) {
                const coordKey = `${station.latitude.toFixed(5)},${station.longitude.toFixed(5)}`;
                if (!coordMap.has(coordKey)) {
                    coordMap.set(coordKey, station);
                }
            } else {
                 console.warn(`Station ${station.id} (${station.name}) has invalid coordinates.`);
            }
        });
        return Array.from(coordMap.values());
    }, [stationDisplayData]);

    // --- Event Handlers (Keep as before) ---
    const handleSelectStation = (station: StationDisplayData) => {
        setSelectedStationId(station.id === selectedStationId ? null : station.id);
    };

    // --- Render Logic ---
    if (isLoading && stationDisplayData.length === 0) {
        return <PageLoader message="Đang tải dữ liệu trang chủ..." />;
    }

    if (error && stationDisplayData.length === 0) {
        // ... (Error Card display) ...
         return (
            <div className="container mx-auto px-4 py-6 flex justify-center items-center h-[calc(100vh-200px)]">
                <Card className="border-red-500 bg-red-50 border-l-4 max-w-md">
                   <CardHeader className="flex flex-row items-center space-x-3 pb-4"> <ServerCrash className="w-6 h-6 text-red-600" /> <CardTitle className="text-red-700">Lỗi Tải Dữ Liệu</CardTitle> </CardHeader>
                   <CardContent> <p className="text-sm text-red-800">{error}</p> <p className="text-sm text-red-800 mt-2">Vui lòng thử làm mới trang hoặc kiểm tra kết nối mạng.</p> </CardContent>
                </Card>
            </div>
        );
    }

    return (
        <div className="container mx-auto px-4 py-6 space-y-8">

             {/* === News Section (Keep as before) === */}
            <section>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Bài viết liên quan</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {articles.map((article, index) => (
                       <Link key={index} href="/newsguestdetail" className="block h-full">
                           <Card className="hover:shadow-lg transition-shadow flex flex-col h-full">
                               <div className="relative w-full h-64 rounded-t-md overflow-hidden">
                                 <Image src={article.imageUrl} alt={article.title} fill style={{ objectFit: "cover" }} sizes="(max-width: 768px) 100vw, 33vw" />
                               </div>
                               <CardHeader className="flex-grow pb-2"> <CardTitle className="text-lg font-semibold">{article.title}</CardTitle> </CardHeader>
                               <CardContent> <p className="text-sm text-gray-600 mb-2 line-clamp-3">{article.description}</p> <p className="text-xs text-gray-500">{article.date}</p> </CardContent>
                           </Card>
                       </Link>
                    ))}
                </div>
                <div className="flex justify-center mt-6"> <Link href="/newsguest"> <Button className="px-6 py-2 text-white bg-black hover:bg-gray-800">Xem thêm</Button> </Link> </div>
            </section>

            {/* NO Warning Dashboard Section */}

            {/* === Geographic Map Section === */}
            <section>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Bản đồ Trạm quan trắc</h2>
                <div className="relative w-full h-[500px] border border-gray-300 rounded-lg overflow-hidden shadow-md z-0">
                    {/* Render only on client AND after leafletAssets are created */}
                    {isClient && leafletAssets ? (
                        <MapContainer key="leaflet-map-guest-client" center={initialMapCenter} zoom={initialMapZoom} style={{ height: "100%", width: "100%" }} scrollWheelZoom={true}>
                            <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                            {uniqueStationsForMap.length > 0 && MapContainer && TileLayer && Marker && Popup && MarkerClusterGroup && (
                                <MarkerClusterGroup chunkedLoading iconCreateFunction={leafletAssets.createClusterCustomIcon} maxClusterRadius={60}>
                                    {uniqueStationsForMap.map((station) => {
                                        const isValidPosition = typeof station.latitude === 'number' && typeof station.longitude === 'number';
                                        if (!isValidPosition) return null;

                                        const isSelected = selectedStationId === station.id;
                                        const displayIcon = isSelected ? leafletAssets.blueIcon : leafletAssets.redIcon;
                                        const status = station.latestData?.status || "Không xác định";
                                        const wqi = station.latestData?.wqi;
                                        const time = formatMonitoringTime(station.latestData?.monitoringTime);
                                        const statusStyle = getStatusTailwindClasses(status);

                                        return (
                                            <Marker key={station.id} position={[station.latitude, station.longitude]} icon={displayIcon} eventHandlers={{ click: () => handleSelectStation(station) }} zIndexOffset={isSelected ? 1000 : 0} >
                                                <Popup minWidth={200}>
                                                    {/* Popup Content */}
                                                    <div className="text-sm space-y-1"> <h3 className="text-md font-bold mb-1">{station.name}</h3> <p className="text-xs text-gray-600">{station.location || `(${station.latitude?.toFixed(4)}, ${station.longitude?.toFixed(4)})`}</p> <hr className="my-1" /> <p className="text-xs font-semibold text-gray-700">Dữ liệu mới nhất:</p> {station.isLoadingData ? (<p className="text-xs text-gray-500 italic">Đang tải...</p>) : station.errorData ? (<p className="text-xs text-red-500 italic">{station.errorData}</p>) : station.latestData ? (<> <p>WQI: <strong className={cn("font-bold", statusStyle.text)}>{wqi?.toFixed(1) ?? "N/A"}</strong></p> <p className={cn(statusStyle.text)}>Trạng thái: {status}</p> <p className="text-xs text-gray-500 mt-1">Lúc: {time}</p> </>) : (<p className="text-xs text-gray-500 italic">Không có dữ liệu.</p>)} {/* Link removed for guest */} </div>
                                                </Popup>
                                            </Marker>
                                        );
                                    })}
                                </MarkerClusterGroup>
                            )}
                             {/* Message if no stations */}
                             {!isLoading && uniqueStationsForMap.length === 0 && (
                                <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-50 z-10">
                                    <p className="text-gray-500">Không có trạm nào để hiển thị trên bản đồ.</p>
                                </div>
                             )}
                        </MapContainer>
                    ) : (
                         // Fallback while waiting for client or leafletAssets
                         <div className="flex items-center justify-center h-full bg-gray-50">
                             <PageLoader message="Đang tải bản đồ..." /> {/* Use PageLoader here */}
                         </div>
                    )}
                </div>
            </section>

            {/* Reminder about potential other errors */}
             {/* If 'window is not defined' persists, check PageLoader or other imports */}

        </div>
    );
}