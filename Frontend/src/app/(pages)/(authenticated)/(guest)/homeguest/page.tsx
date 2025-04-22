"use client";

import React, { useState, useEffect, useMemo } from "react";
import dynamic from "next/dynamic"; // For client-side only components like Leaflet
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"; // Added CardDescription, CardFooter
import "leaflet/dist/leaflet.css"; // Leaflet CSS
import L, { Icon, DivIcon, PointTuple, point as leafletPoint, LatLngExpression } from 'leaflet'; // Import Leaflet types directly
import { MapPin, AlertTriangle, ArrowRight, ServerCrash, CheckCircle, Droplet } from "lucide-react"; // Added more icons
import PageLoader from "@/components/pageloader"; // Your loader component
import { cn } from "@/lib/utils"; // Utility for class names

// --- Make sure these paths and types are correct in your project ---
import {
    Station,
    DataPoint,
    // QueryOptions, // Uncomment if getStations needs it
    ApiRequestDataPointsByStationId,
    // RequestOptions // Uncomment if needed by ApiRequestDataPointsByStationId
} from "@/types/station2"; // Assuming this path is correct
import { getStations, getAllDataPointsByStationID } from "@/lib/station"; // Assuming this path is correct
// --- End of dependency check ---

import { format, parseISO, isValid as isValidDate } from "date-fns"; // For date formatting

// --- Dynamic Imports for Leaflet (Client-side only) ---
const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(mod => mod.Popup), { ssr: false });
const MarkerClusterGroup = dynamic(() => import("react-leaflet-cluster"), { ssr: false });

// --- Leaflet Icons (Client-side setup) ---
// Ensure /red_one.png and /blue_one.png exist in your public folder
let redIcon: Icon | undefined, blueIcon: Icon | undefined, createClusterCustomIcon: ((cluster: any) => DivIcon) | undefined;
if (typeof window !== "undefined") {
    // Use the imported L directly
    redIcon = new L.Icon({
        iconUrl: "/red_one.png",
        iconSize: [38, 38] as PointTuple,
        iconAnchor: [19, 38] as PointTuple,
        popupAnchor: [0, -38] as PointTuple,
    });

    blueIcon = new L.Icon({
        iconUrl: "/blue_one.png",
        iconSize: [38, 38] as PointTuple,
        iconAnchor: [19, 38] as PointTuple,
        popupAnchor: [0, -38] as PointTuple,
    });

    createClusterCustomIcon = (cluster: any): DivIcon => {
        const count = cluster.getChildCount();
        const style = `
            background-color: rgba(51, 51, 51, 0.8);
            height: 2.5em;
            width: 2.5em;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            font-size: 1rem;
            font-weight: bold;
            box-shadow: 0 0 5px rgba(0,0,0,0.5);
            border: 2px solid #fff;
        `;
        return L.divIcon({
            html: `<span style="${style}">${count}</span>`,
            className: "custom-marker-cluster",
            iconSize: leafletPoint(40, 40, true),
        });
    };
}

// --- Helper Functions ---

// Function to derive status text from WQI
function deriveStatusFromWqi(wqi: number | null | undefined): string {
    if (wqi === null || wqi === undefined || isNaN(wqi)) {
        return "Không xác định";
    }
    if (wqi > 91) return "Rất Tốt";
    if (wqi > 76) return "Tốt";
    if (wqi > 51) return "Trung Bình";
    if (wqi > 26) return "Kém";
    if (wqi >= 0) return "Rất Kém";
    return "Không xác định";
}

// Function to get Tailwind CSS classes based on status for styling cards/elements
const getStatusTailwindClasses = (status: string): { bg: string; text: string; border: string; iconColor: string } => {
    switch (status) {
        case "Rất Tốt":
            return { bg: "bg-emerald-50", text: "text-emerald-800", border: "border-emerald-400", iconColor: "text-emerald-500" };
        case "Tốt":
            return { bg: "bg-green-50", text: "text-green-800", border: "border-green-400", iconColor: "text-green-500" };
        case "Trung Bình":
            return { bg: "bg-yellow-50", text: "text-yellow-800", border: "border-yellow-400", iconColor: "text-yellow-500" };
        case "Kém":
            return { bg: "bg-orange-50", text: "text-orange-800", border: "border-orange-400", iconColor: "text-orange-500" };
        case "Rất Kém":
            return { bg: "bg-red-50", text: "text-red-800", border: "border-red-400", iconColor: "text-red-500" };
        case "Không xác định":
        default:
            return { bg: "bg-gray-100", text: "text-gray-800", border: "border-gray-400", iconColor: "text-gray-500" };
    }
};

// Function to format the monitoring time string
function formatMonitoringTime(timeString: string | undefined | null): string {
    if (!timeString) return "N/A";
    try {
        const date = parseISO(timeString);
        if (!isValidDate(date)) {
            console.warn("Invalid date format received:", timeString);
            return timeString; // Return original if invalid
        }
        return format(date, 'HH:mm, dd/MM/yyyy'); // Format as HH:mm, dd/MM/yyyy
    } catch (e) {
        console.error("Failed to format time:", timeString, e);
        return timeString; // Return original on error
    }
}

// Interface for combined station and latest data display
interface StationDisplayData extends Station {
    latestData: (DataPoint & { status: string }) | null; // Include derived status
    isLoadingData: boolean; // Track loading specific to this station's data
    errorData: string | null; // Track error specific to this station's data
}

// --- Mock Data for News Section ---
const articles = [
    {
        title: "Bản tin khẩn chất lượng nước sông Tiền !!",
        date: "4 Nov 2024", // Adjust date if needed
        description: "Dữ liệu quan trắc được từ trạm sông Tiền cho thấy rằng chất lượng nước nơi này đang có sự ô nhiễm nhẹ...",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
    },
    {
        title: "Báo cáo hàng tuần về chất lượng nước sông Hậu!!",
        date: "4 Nov 2024", // Adjust date if needed
        description: "Dữ liệu quan trắc được từ trạm sông Hồng cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
    },
    {
        title: "Báo cáo hàng tuần về chất lượng nước sông Cần!!",
        date: "4 Nov 2024", // Adjust date if needed
        description: "Dữ liệu quan trắc được từ trạm sông Cần cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
    },
];


// --- Main Component ---
export default function StationsPage() {
    // --- State ---
    const [stationDisplayData, setStationDisplayData] = useState<StationDisplayData[]>([]); // Holds stations merged with latest data
    const [selectedStationId, setSelectedStationId] = useState<string | null>(null); // Track selected station on map
    const [isLoading, setIsLoading] = useState(true); // Overall loading state for map/station data
    const [error, setError] = useState<string | null>(null); // Overall error state for map/station data

    // --- Constants ---
    const initialMapCenter: LatLngExpression = [10.8231, 106.6297]; // HCMC default center
    const initialMapZoom = 10; // Default map zoom

    // --- Data Fetching Effect (For Map Data) ---
    useEffect(() => {
        const fetchMapData = async () => {
            setIsLoading(true);
            setError(null);
            setStationDisplayData([]); // Clear previous data

            try {
                // 1. Fetch all stations
                const stations = await getStations({ limit: 1000 }); // Adjust limit if needed

                // Initialize display data with loading state for data points
                const initialDisplayData: StationDisplayData[] = stations.map(station => ({
                    ...station,
                    latestData: null,
                    isLoadingData: true, // Start loading data for each
                    errorData: null,
                }));
                setStationDisplayData(initialDisplayData); // Show stations on map/list immediately

                // 2. Fetch latest 'actual' data point for each station concurrently
                const dataPointPromises = stations.map(async (station) => {
                    try {
                        // Define the payload for the POST request
                        const apiRequestOptions: ApiRequestDataPointsByStationId = {
                            stationId: station.id,
                            options: {
                                limit: 1,
                                sortBy: 'monitoring_time', // Ensure backend uses this field name
                                sortDesc: true,
                                filters: {
                                    observation_type: 'actual' // Filter for actual data points
                                }
                            }
                        };

                        const dataPoints = await getAllDataPointsByStationID(apiRequestOptions);

                        if (dataPoints.length > 0) {
                            const latestPoint = dataPoints[0];
                            const status = deriveStatusFromWqi(latestPoint.wqi);
                            return { stationId: station.id, latestData: { ...latestPoint, status }, errorData: null };
                        } else {
                            console.log(`No 'actual' data points found for station ${station.id}`);
                            return { stationId: station.id, latestData: null, errorData: null };
                        }
                    } catch (err) {
                        console.error(`Failed to fetch latest 'actual' data for station ${station.id}:`, err);
                        return { stationId: station.id, latestData: null, errorData: "Lỗi tải dữ liệu điểm đo" };
                    }
                });

                const results = await Promise.allSettled(dataPointPromises);

                // Update the stationDisplayData state with the results
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
                        } else if (result.status === 'rejected') {
                            console.error("Promise rejected (should have been caught):", result.reason);
                            // Handle potential error updating state here if needed
                        }
                    });
                    return Array.from(newDataMap.values());
                });

            } catch (err) {
                console.error("Failed to fetch stations or process data:", err);
                setError(err instanceof Error ? err.message : "Lỗi không xác định khi tải dữ liệu bản đồ.");
            } finally {
                setIsLoading(false); // Mark overall map data loading as complete
            }
        };

        fetchMapData();
    }, []); // Run only once on mount

    // --- Memoized Data Derivations ---

    // Derive unique stations for map markers (handles potential coordinate overlaps)
    const uniqueStationsForMap = useMemo(() => {
        const coordMap = new Map<string, StationDisplayData>();
        stationDisplayData.forEach((station) => {
            if (typeof station.latitude === 'number' && typeof station.longitude === 'number' && !isNaN(station.latitude) && !isNaN(station.longitude)) {
                const coordKey = `${station.latitude.toFixed(5)},${station.longitude.toFixed(5)}`;
                if (!coordMap.has(coordKey)) {
                    coordMap.set(coordKey, station);
                }
                // Simple handling: first station at a coordinate wins. Could be improved if needed.
            } else {
                console.warn(`Station ${station.id} (${station.name}) has invalid coordinates.`);
            }
        });
        return Array.from(coordMap.values());
    }, [stationDisplayData]); // Recalculate when display data changes

    // --- Event Handlers ---

    const handleSelectStation = (station: StationDisplayData) => {
        setSelectedStationId(station.id === selectedStationId ? null : station.id); // Toggle selection
    };

    // --- Render Logic ---

    // Show loader only during initial map data load if there's no data yet
    if (isLoading && stationDisplayData.length === 0) {
        return <PageLoader message="Đang tải dữ liệu bản đồ và trạm..." />;
    }

    // Show error only if loading failed completely for map data
    if (error && stationDisplayData.length === 0) {
        return (
            <div className="container mx-auto px-4 py-6 flex justify-center items-center h-[calc(100vh-200px)]">
                <Card className="border-red-500 bg-red-50 border-l-4 max-w-md">
                    <CardHeader className="flex flex-row items-center space-x-3 pb-4">
                        <ServerCrash className="w-6 h-6 text-red-600" />
                        <CardTitle className="text-red-700">Lỗi Tải Dữ Liệu Bản Đồ</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="text-sm text-red-800">{error}</p>
                        <p className="text-sm text-red-800 mt-2">Vui lòng thử làm mới trang hoặc kiểm tra kết nối mạng.</p>
                    </CardContent>
                </Card>
            </div>
        );
    }

    // --- Main Page Structure ---
    return (
        <div className="container mx-auto px-4 py-6 space-y-8"> {/* Increased space-y */}

            {/* === News Section (Remains from your original code) === */}
            <section>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Bài viết liên quan</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {articles.map((article, index) => (
                        <Link key={index} href="/newsguestdetail" className="block h-full"> {/* Adjust link */}
                            <Card className="hover:shadow-lg transition-shadow flex flex-col h-full">
                                <img
                                    src={article.imageUrl}
                                    alt={article.title}
                                    className="w-full h-64 object-cover rounded-t-md" // Increased height
                                />
                                <CardHeader className="flex-grow pb-2"> {/* Reduced padding bottom */}
                                    <CardTitle className="text-lg font-semibold">{article.title}</CardTitle> {/* Slightly smaller title */}
                                </CardHeader>
                                <CardContent>
                                    <p className="text-sm text-gray-600 mb-2 line-clamp-3">{article.description}</p> {/* Added line-clamp */}
                                    <p className="text-xs text-gray-500">{article.date}</p>
                                </CardContent>
                            </Card>
                        </Link>
                    ))}
                </div>
                <div className="flex justify-center mt-6">
                    <Link href="/newsguest"> {/* Adjust link */}
                        <Button className="px-6 py-2 text-white bg-black hover:bg-gray-800">
                            Xem thêm
                        </Button>
                    </Link>
                </div>
            </section>

            {/* === Geographic Map Section (Moved from sample code) === */}
            <section>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Bản đồ Trạm quan trắc</h2>
                {/* Display general map loading indicator or error if needed */}
                {isLoading && stationDisplayData.length > 0 && (
                     <p className="text-gray-500 italic mb-4">Đang cập nhật dữ liệu trạm trên bản đồ...</p>
                )}
                 {error && stationDisplayData.length > 0 && (
                     <p className="text-red-500 italic mb-4">Lỗi tải một phần dữ liệu trạm: {error}</p>
                )}

                <div className="relative w-full h-[500px] border border-gray-300 rounded-lg overflow-hidden shadow-md z-0"> {/* Ensure z-index is lower than popups */}
                    {/* Check if window is defined for client-side rendering */}
                    {typeof window !== 'undefined' && MapContainer && TileLayer && Marker && Popup && MarkerClusterGroup && redIcon && blueIcon && createClusterCustomIcon ? (
                        <MapContainer
                            key="leaflet-map-stations" // Unique key for this map instance
                            center={initialMapCenter}
                            zoom={initialMapZoom}
                            style={{ height: "100%", width: "100%" }}
                            scrollWheelZoom={true}
                        >
                            <TileLayer
                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'
                                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            />

                            {/* Marker Clustering */}
                            {uniqueStationsForMap.length > 0 ? (
                                <MarkerClusterGroup
                                    chunkedLoading
                                    iconCreateFunction={createClusterCustomIcon}
                                    maxClusterRadius={60} // Adjust for desired clustering sensitivity
                                >
                                    {/* Map through unique stations */}
                                    {uniqueStationsForMap.map((station) => {
                                        const isValidPosition = typeof station.latitude === 'number' && typeof station.longitude === 'number';
                                        if (!isValidPosition) return null; // Skip if invalid coords

                                        const isSelected = selectedStationId === station.id;
                                        const status = station.latestData?.status || "Không xác định";
                                        const wqi = station.latestData?.wqi;
                                        const time = formatMonitoringTime(station.latestData?.monitoringTime);
                                        const statusStyle = getStatusTailwindClasses(status);
                                        const displayIcon = isSelected ? blueIcon : redIcon; // Use defined icons

                                        return (
                                            <Marker
                                                key={station.id}
                                                position={[station.latitude, station.longitude]}
                                                icon={displayIcon!} // Add non-null assertion (!) as it's checked above
                                                eventHandlers={{
                                                    click: () => handleSelectStation(station),
                                                }}
                                                zIndexOffset={isSelected ? 1000 : 0} // Bring selected to front
                                            >
                                                <Popup minWidth={200}> {/* Popup content */}
                                                    <div className="text-sm space-y-1">
                                                        <h3 className="text-md font-bold mb-1">{station.name}</h3>
                                                        <p className="text-xs text-gray-600">
                                                            {station.location || `(${station.latitude.toFixed(4)}, ${station.longitude.toFixed(4)})`}
                                                        </p>
                                                        <hr className="my-1" />
                                                        <p className="text-xs font-semibold text-gray-700">Dữ liệu thực tế mới nhất:</p>
                                                        {/* Show data loading state or actual data */}
                                                        {station.isLoadingData ? (
                                                            <p className="text-xs text-gray-500 italic">Đang tải trạng thái...</p>
                                                        ) : station.errorData ? (
                                                            <p className="text-xs text-red-500 italic">{station.errorData}</p>
                                                        ) : station.latestData ? (
                                                            <>
                                                                <p>WQI: <strong className={cn("font-bold", statusStyle.text)}>{wqi !== null && wqi !== undefined ? wqi.toFixed(1) : "N/A"}</strong></p>
                                                                <p className={cn(statusStyle.text)}>Trạng thái: {status}</p>
                                                                <p className="text-xs text-gray-500 mt-1">Lúc: {time}</p>
                                                            </>
                                                        ) : (
                                                            <p className="text-xs text-gray-500 italic">Không có dữ liệu thực tế gần đây.</p>
                                                        )}
                                                        {/* Link to details page */}
                                                        {/* <div className="mt-2 pt-1 border-t">
                                                            <Button asChild variant="link" size="sm" className="p-0 h-auto text-xs text-blue-600 hover:text-blue-800 font-medium">
                                                                
                                                                <Link href={`/stations/${station.id}`}> 
                                                                    Xem thêm
                                                                    <ArrowRight className="w-3 h-3 ml-1" />
                                                                </Link>
                                                                
                                                            </Button>
                                                        </div> */}
                                                    </div>
                                                </Popup>
                                            </Marker>
                                        );
                                    })}
                                </MarkerClusterGroup>
                            ) : (
                                // Optional: Message if no stations have valid coordinates or data loaded yet
                                !isLoading && <div className="p-4 text-center text-gray-500">Không có trạm nào để hiển thị trên bản đồ.</div>
                            )}
                        </MapContainer>
                    ) : (
                         /* Fallback or message if Leaflet components aren't ready (e.g., during SSR or initial client load) */
                         <div className="flex items-center justify-center h-full bg-gray-100">
                            <p className="text-gray-500">Đang tải bản đồ...</p>
                         </div>
                    )}
                </div>
            </section>

        </div>
    );
}