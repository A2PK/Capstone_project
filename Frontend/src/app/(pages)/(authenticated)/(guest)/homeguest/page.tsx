// src/app/(pages)/(authenticated)/(guest)/homeguest/page.tsx

"use client";

import React, { useState, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { MapPin, AlertTriangle, ArrowRight, ServerCrash, CheckCircle, Droplet, Newspaper, Loader2, AlertCircle as ErrorIcon } from "lucide-react";
import PageLoader from "@/components/pageloader";
import { cn } from "@/lib/utils";
// Types
import {
    Station,
    DataPoint,
    ApiRequestDataPointsByStationId, // Assuming this type includes options like limit, sortBy etc.
    QueryOptions as StationQueryOptions // Rename if QueryOptions is used differently for articles
} from "@/types/station2";
import { Article } from "@/types/article";
import { QueryOptions as ArticleQueryOptions } from "@/types/station2"; // Assuming article also has QueryOptions type

// API Functions
import { getStations, getAllDataPointsByStationID } from "@/lib/station";
import { getAllArticles, generateProxyUrl } from "@/lib/article"; // Re-added generateProxyUrl

// Date Formatting
import { format, parseISO, isValid as isValidDate } from "date-fns";

// --- Dynamic Imports for React-Leaflet Components ---
const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(mod => mod.Popup), { ssr: false });
const MarkerClusterGroup = dynamic(() => import("react-leaflet-cluster"), { ssr: false });

// Type definition for Leaflet library (will be populated after dynamic import)
// Using 'any' for simplicity here, but you could use @types/leaflet if strictly needed server-side
let L: any = null;

// Type for Leaflet assets state
type LeafletAssets = {
    redIcon: any; // L.Icon
    blueIcon: any; // L.Icon
    createClusterCustomIcon: (cluster: any) => any; // L.DivIcon
};

// Type for map center/position (LatLngExpression from leaflet)
type LeafletLatLngExpression = [number, number];

// --- Helper Functions ---
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

// Interface for Station Display Data
interface StationDisplayData extends Station {
    latestData: (DataPoint & { status: string }) | null;
    isLoadingData: boolean;
    errorData: string | null;
}

// Ảnh mặc định
const DEFAULT_IMAGE_URL = "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg";

// --- Main Component ---
export default function HomeGuestPage() {
    // --- State ---
    const [stationDisplayData, setStationDisplayData] = useState<StationDisplayData[]>([]);
    const [selectedStationId, setSelectedStationId] = useState<string | null>(null);
    const [isLoadingStations, setIsLoadingStations] = useState(true);
    const [errorStations, setErrorStations] = useState<string | null>(null);
    const [isClient, setIsClient] = useState(false);
    const [leafletAssets, setLeafletAssets] = useState<LeafletAssets | null>(null);

    // State cho bài viết
    const [newsArticles, setNewsArticles] = useState<Article[]>([]);
    const [isLoadingNews, setIsLoadingNews] = useState<boolean>(true);
    const [errorNews, setErrorNews] = useState<string | null>(null);

    // --- Constants ---
    const initialMapCenter: LeafletLatLngExpression = [10.8231, 106.6297]; // Ho Chi Minh City center
    const initialMapZoom = 10;

    // Client-side detection effect
    useEffect(() => { setIsClient(true); }, []);

    // Leaflet initialization effect
    useEffect(() => {
        if (isClient && !leafletAssets) { // Chỉ chạy nếu ở client và assets chưa được tạo
            Promise.all([
                import('leaflet'),
                import('leaflet/dist/leaflet.css' as any)
            ]).then(([Leaflet]) => {
                L = Leaflet; // Gán L đã import vào biến toàn cục (hoặc cục bộ nếu thích)
                const redIcon = new L.Icon({
                    iconUrl: "/red_one.png",
                    iconSize: [38, 38] as L.PointTuple,
                    iconAnchor: [19, 38] as L.PointTuple,
                    popupAnchor: [0, -38] as L.PointTuple,
                });
                const blueIcon = new L.Icon({
                    iconUrl: "/blue_one.png",
                    iconSize: [38, 38] as L.PointTuple,
                    iconAnchor: [19, 38] as L.PointTuple,
                    popupAnchor: [0, -38] as L.PointTuple,
                });
                const createClusterCustomIcon = (cluster: any): L.DivIcon => {
                    const count = cluster.getChildCount();
                    const style = ` background-color: rgba(51, 51, 51, 0.8); height: 2.5em; width: 2.5em; color: #fff; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 1rem; font-weight: bold; box-shadow: 0 0 5px rgba(0,0,0,0.5); border: 2px solid #fff; `;
                    return L.divIcon({
                        html: `<span style="${style}">${count}</span>`,
                        className: "custom-marker-cluster",
                        iconSize: L.point(40, 40),
                    });
                };
                setLeafletAssets({ redIcon, blueIcon, createClusterCustomIcon });
            }).catch(err => {
                console.error("Failed to load Leaflet dynamically:", err);
                setErrorStations("Không thể tải thư viện bản đồ."); // Gán lỗi vào state chính nếu map là cốt lõi
            });
        }
    }, [isClient, leafletAssets]); // Thêm leafletAssets để tránh chạy lại không cần thiết

    // --- Data Fetching Effect cho Stations ---
    useEffect(() => {
        const fetchInitialData = async () => {
            setIsLoadingStations(true);
            setErrorStations(null);
            setStationDisplayData([]);
            try {
                // Fetch stations (Limit to 1000 or adjust as needed)
                const stations = await getStations({ limit: 1000 });
                const initialDisplayData: StationDisplayData[] = stations.map(station => ({
                    ...station, latestData: null, isLoadingData: true, errorData: null,
                }));
                setStationDisplayData(initialDisplayData); // Set stations first for map to render pins

                // Fetch latest data point for each station
                const dataPointPromises = stations.map(async (station) => {
                    try {
                        const apiRequestOptions: ApiRequestDataPointsByStationId = {
                            stationId: station.id,
                            options: { limit: 1, sortBy: 'monitoring_time', sortDesc: true, filters: { observation_type: 'actual' } }
                        };
                        const dataPoints = await getAllDataPointsByStationID(apiRequestOptions);
                        if (dataPoints && dataPoints.length > 0) {
                            const latestPoint = dataPoints[0];
                            const status = deriveStatusFromWqi(latestPoint.wqi);
                            return { stationId: station.id, latestData: { ...latestPoint, status }, errorData: null };
                        } else {
                            return { stationId: station.id, latestData: null, errorData: null }; // No data found
                        }
                    } catch (err) {
                        console.error(`Failed fetch data for station ${station.id}:`, err);
                        return { stationId: station.id, latestData: null, errorData: "Lỗi tải dữ liệu điểm đo" };
                    }
                });

                // Wait for all data points fetches and update state
                const results = await Promise.allSettled(dataPointPromises);

                setStationDisplayData(prevData => {
                    const newDataMap = new Map(prevData.map(s => [s.id, { ...s }])); // Create a new map copy
                    results.forEach(result => {
                        if (result.status === 'fulfilled' && result.value) {
                            const { stationId, latestData, errorData } = result.value;
                            const station = newDataMap.get(stationId);
                            if (station) {
                                station.latestData = latestData;
                                station.isLoadingData = false; // Mark as loaded
                                station.errorData = errorData; // Set specific error or null
                            }
                        } else if (result.status === 'rejected') {
                            // Attempt to find which station failed if possible, or handle generically
                            console.error("A datapoint promise was rejected:", result.reason);
                            // You might need a way to map the rejected promise back to a station ID if needed
                        }
                    });
                    return Array.from(newDataMap.values()); // Return new array from updated map
                });

            } catch (err) {
                console.error("Failed to fetch stations or process data:", err);
                setErrorStations(err instanceof Error ? err.message : "Lỗi không xác định khi tải trạm.");
            } finally {
                setIsLoadingStations(false);
            }
        };
        fetchInitialData();
    }, []); // Run once on mount

    // --- useEffect CHO FETCH BÀI VIẾT ---
    useEffect(() => {
        const fetchNews = async () => {
            setIsLoadingNews(true);
            setErrorNews(null);
            try {
                const options: ArticleQueryOptions = { limit: 100, sortBy: 'created_at', sortDesc: true };
                const fetchedNews = await getAllArticles(options);
                const allArticles = fetchedNews || [];
                const dangerArticles = allArticles.filter(article => article.badge === 'danger');

            // 3. Lấy 3 bài đầu tiên từ danh sách đã lọc (vì đã sắp xếp sẵn)
                const latestThreeDangerArticles = dangerArticles.slice(0, 3);
                setNewsArticles(latestThreeDangerArticles);
            } catch (err) {
                console.error("Failed to fetch news articles:", err);
                setErrorNews("Không thể tải các bài viết mới nhất. Vui lòng thử lại sau.");
                setNewsArticles([]);
            } finally {
                setIsLoadingNews(false);
            }
        };
         if (isClient) { fetchNews(); }
    }, [isClient]);

    // --- Memoized Data ---
     const uniqueStationsForMap = useMemo(() => {
         const coordMap = new Map<string, StationDisplayData>();
         stationDisplayData.forEach((station) => {
             if (typeof station.latitude === 'number' && typeof station.longitude === 'number' && !isNaN(station.latitude) && !isNaN(station.longitude)) {
                 const coordKey = `${station.latitude.toFixed(5)},${station.longitude.toFixed(5)}`;
                 // Keep the first station found at a specific coordinate
                 if (!coordMap.has(coordKey)) {
                     coordMap.set(coordKey, station);
                 }
                 // Optional: logic to handle multiple stations at the exact same spot if needed
             } else {
                 console.warn(`Station ${station.id} (${station.name}) has invalid coordinates.`);
             }
         });
         return Array.from(coordMap.values());
     }, [stationDisplayData]); // Recalculate when station data changes

    // --- Event Handlers ---
    const handleSelectStation = (station: StationDisplayData) => {
        setSelectedStationId(station.id === selectedStationId ? null : station.id);
        // Add logic here if you want the map to pan/zoom to the selected station
    };

    // --- Render Logic ---
    // Show main loader only if both initial station and news loads are pending
     if (isLoadingStations && isLoadingNews && stationDisplayData.length === 0 && newsArticles.length === 0) {
        return <PageLoader message="Đang tải dữ liệu trang chủ..." />;
    }
    // Show main error only if station loading failed significantly
    if (errorStations && stationDisplayData.length === 0) {
        return (
             <div className="container mx-auto px-4 py-6 flex justify-center items-center h-[calc(100vh-200px)]">
                <Card className="border-red-500 bg-red-50 border-l-4 max-w-md">
                   <CardHeader className="flex flex-row items-center space-x-3 pb-4"> <ServerCrash className="w-6 h-6 text-red-600" /> <CardTitle className="text-red-700">Lỗi Tải Dữ Liệu Trạm</CardTitle> </CardHeader>
                   <CardContent> <p className="text-sm text-red-800">{errorStations}</p> <p className="text-sm text-red-800 mt-2">Vui lòng thử làm mới trang.</p> </CardContent>
                </Card>
            </div>
        );
    }

    return (
        <div className="container mx-auto px-4 py-6 space-y-8">

            {/* === News Section === */}
            <section>
                <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Newspaper className="h-6 w-6 text-orange-600"/> Bài viết liên quan
                </h2>

                {/* Loading News */}
                {isLoadingNews && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                         {[...Array(3)].map((_, index) => (
                             <Card key={index} className="animate-pulse flex flex-col h-full overflow-hidden">
                                 <div className="bg-gray-300 h-64 rounded-t-md"></div>
                                 <CardHeader className="pb-2"><div className="h-6 bg-gray-300 rounded w-3/4 mb-2"></div></CardHeader>
                                 <CardContent className="flex-grow pt-0"><div className="h-4 bg-gray-300 rounded mb-2"></div><div className="h-4 bg-gray-300 rounded mb-2 w-5/6"></div><div className="h-4 bg-gray-300 rounded w-1/2 mt-3"></div></CardContent>
                             </Card>
                         ))}
                    </div>
                )}
                {/* Error News */}
                {!isLoadingNews && errorNews && (
                    <div className="text-center text-red-600 bg-red-50 border border-red-200 p-6 rounded-md flex flex-col items-center justify-center min-h-[200px]">
                         <ErrorIcon className="h-10 w-10 mb-3 text-red-500" />
                         <p className="font-semibold">Đã xảy ra lỗi</p>
                         <p className="text-sm mt-1">{errorNews}</p>
                    </div>
                )}
                {/* News List */}
                {!isLoadingNews && !errorNews && newsArticles.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {newsArticles.map((article) => (
                            <Link key={article.id} href={`/newsguest/${article.id}`} className="block h-full">
                                <Card className="hover:shadow-lg transition-shadow flex flex-col h-full">
                                    <div className="relative w-full h-64 rounded-t-md overflow-hidden">
                                        <Image
                                            src={generateProxyUrl(article.pictureUrl) || DEFAULT_IMAGE_URL} // Dùng lại proxy
                                            alt={article.title}
                                            fill
                                            style={{ objectFit: "cover" }}
                                            sizes="(max-width: 768px) 100vw, 33vw"
                                            onError={(e) => { (e.target as HTMLImageElement).src = DEFAULT_IMAGE_URL; }}
                                        />
                                        <span className={`absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white ${ article.badge === 'danger' ? 'bg-red-500' : article.badge === 'good' ? 'bg-green-500' : 'bg-gray-500' }`}>
                                            {article.badge === 'danger' ? 'Khẩn cấp' : article.badge === 'good' ? 'Tốt' : 'Thông thường'}
                                        </span>
                                    </div>
                                    <CardHeader className="flex-grow pb-2"><CardTitle className="text-lg font-semibold line-clamp-2">{article.title}</CardTitle></CardHeader>
                                    <CardContent><p className="text-sm text-gray-600 mb-2 line-clamp-3">{article.content}</p><p className="text-xs text-gray-500">{format(parseISO(article.createdAt), 'dd/MM/yyyy HH:mm')}</p></CardContent>
                                </Card>
                            </Link>
                        ))}
                    </div>
                )}
                {/* Empty News */}
                {!isLoadingNews && !errorNews && newsArticles.length === 0 && ( <div className="text-center text-gray-500 mt-6 border rounded-lg p-8 bg-gray-50">Không có bài viết nào để hiển thị.</div> )}
                {/* View More Button */}
                <div className="flex justify-center mt-6"><Link href="/newsguest"><Button className="px-6 py-2 text-white bg-black hover:bg-gray-800">Xem tất cả bản tin</Button></Link></div>
            </section>
            {/* === End News Section === */}


            {/* === Geographic Map Section === */}
            <section>
                 <h2 className="text-2xl font-bold text-gray-800 mb-4">Bản đồ Trạm quan trắc</h2>
                 <div className="relative w-full h-[500px] border border-gray-300 rounded-lg overflow-hidden shadow-md z-0">
                     {isClient && leafletAssets && L ? ( // Ensure L is also loaded
                         <MapContainer key="leaflet-map-guest-client" center={initialMapCenter} zoom={initialMapZoom} style={{ height: "100%", width: "100%" }} scrollWheelZoom={true}>
                             <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                             {uniqueStationsForMap.length > 0 && MapContainer && TileLayer && Marker && Popup && MarkerClusterGroup && (
                                 <MarkerClusterGroup chunkedLoading iconCreateFunction={leafletAssets.createClusterCustomIcon} maxClusterRadius={60}>
                                     {uniqueStationsForMap.map((station) => {
                                         const isValidPosition = typeof station.latitude === 'number' && typeof station.longitude === 'number';
                                         if (!isValidPosition) return null; // Skip stations without valid coordinates

                                         const isSelected = selectedStationId === station.id;
                                         const displayIcon = isSelected ? leafletAssets.blueIcon : leafletAssets.redIcon;
                                         const status = station.latestData?.status || "Không xác định";
                                         const wqi = station.latestData?.wqi;
                                         const time = formatMonitoringTime(station.latestData?.monitoringTime);
                                         const statusStyle = getStatusTailwindClasses(status);

                                         return (
                                             <Marker
                                                 key={station.id}
                                                 position={[station.latitude, station.longitude] as LeafletLatLngExpression}
                                                 icon={displayIcon}
                                                 eventHandlers={{ click: () => handleSelectStation(station) }}
                                                 zIndexOffset={isSelected ? 1000 : 0}
                                             >
                                                 <Popup minWidth={220}>
                                                      {/* Popup Content - Đầy đủ */}
                                                     <div className="text-sm space-y-1">
                                                         <h3 className="text-md font-bold mb-1">{station.name}</h3>
                                                         <p className="text-xs text-gray-600">{station.location || `(${station.latitude?.toFixed(4)}, ${station.longitude?.toFixed(4)})`}</p>
                                                         <hr className="my-1.5" />
                                                         <p className="text-xs font-semibold text-gray-700">Dữ liệu mới nhất:</p>
                                                         {station.isLoadingData ? (
                                                             <p className="text-xs text-gray-500 italic flex items-center gap-1"><Loader2 className="h-3 w-3 animate-spin"/> Đang tải...</p>
                                                         ) : station.errorData ? (
                                                             <p className="text-xs text-red-500 italic">{station.errorData}</p>
                                                         ) : station.latestData ? (
                                                             <>
                                                                 <p>WQI: <strong className={cn("font-bold", statusStyle.text)}>{wqi?.toFixed(1) ?? "N/A"}</strong></p>
                                                                 <p className={cn("flex items-center gap-1", statusStyle.text)}>
                                                                     Trạng thái: {status}
                                                                     {status === "Rất Tốt" && <CheckCircle className="h-3.5 w-3.5 inline-block"/>}
                                                                     {status === "Tốt" && <CheckCircle className="h-3.5 w-3.5 inline-block"/>}
                                                                     {status === "Trung Bình" && <Droplet className="h-3.5 w-3.5 inline-block"/>}
                                                                     {status === "Kém" && <AlertTriangle className="h-3.5 w-3.5 inline-block"/>}
                                                                     {status === "Rất Kém" && <AlertTriangle className="h-3.5 w-3.5 inline-block"/>}
                                                                 </p>
                                                                 <p className="text-xs text-gray-500 mt-1">Lúc: {time}</p>
                                                             </>
                                                         ) : (
                                                             <p className="text-xs text-gray-500 italic">Không có dữ liệu.</p>
                                                         )}
                                                          {/* Removed station detail link for guests */}
                                                     </div>
                                                 </Popup>
                                             </Marker>
                                         );
                                     })}
                                 </MarkerClusterGroup>
                             )}
                              {/* Message if no stations */}
                              {!isLoadingStations && uniqueStationsForMap.length === 0 && (
                                 <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-50 z-10">
                                     <p className="text-gray-500">Không có trạm nào để hiển thị trên bản đồ.</p>
                                 </div>
                             )}
                         </MapContainer>
                     ) : (
                         // Fallback while waiting for client or leafletAssets
                         <div className="flex items-center justify-center h-full bg-gray-50">
                            <PageLoader message="Đang tải bản đồ..." />
                         </div>
                     )}
                 </div>
            </section>

        </div>
    );
}