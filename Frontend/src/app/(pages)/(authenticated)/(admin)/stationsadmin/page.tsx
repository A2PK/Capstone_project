// File: src/app/stationsadmin/page.tsx
"use client";

import React, { useEffect, useState, useMemo, ChangeEvent, useCallback } from "react";
import dynamic from "next/dynamic";
import { useSearchParams, useRouter } from 'next/navigation'; // Hook để đọc URL search params
import { format } from 'date-fns';
import L from 'leaflet';
import { Station, DataPoint, QueryOptions, Indicator } from "@/types/station2"; // Đảm bảo đường dẫn đúng
import { getStations, getDataPointsOfStationById } from "@/lib/station"; // Đảm bảo đường dẫn đúng
import { getStatusTextColor } from "@/lib/utils"; // Đảm bảo đường dẫn đúng

// Import các UI components
import { Pagination } from "@/components/pagination2"; // Đảm bảo đường dẫn đúng
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"; // Đảm bảo đường dẫn đúng
import { Input } from "@/components/ui/input"; // Đảm bảo đường dẫn đúng

// Import các components tùy chỉnh
import StationDetails from "@/components/stationsDetails"; // Đảm bảo đường dẫn đúng
import Chartline from "@/components/linechart"; // Đảm bảo đường dẫn đúng
import PageLoader from "@/components/pageloader"; // Đảm bảo đường dẫn đúng
import MapUpdater from '@/components/MapUpdater'; // Đảm bảo đường dẫn đúng

// Import CSS cho Leaflet
import "leaflet/dist/leaflet.css";

// --- Dynamic Imports cho các component của Leaflet (chỉ chạy phía client) ---
const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(mod => mod.Popup), { ssr: false });
const MarkerClusterGroup = dynamic(() => import("react-leaflet-cluster"), { ssr: false });


// --- Định nghĩa Icons cho Leaflet (chỉ chạy phía client) ---
let redIcon: L.Icon | undefined, blueIcon: L.Icon | undefined, createClusterCustomIcon: ((cluster: any) => L.DivIcon) | undefined;
if (typeof window !== "undefined") {
    // Đảm bảo Leaflet chỉ được require ở client-side
    const LGlobal = require("leaflet");

    // Icon màu đỏ cho trạm không được chọn
    redIcon = new LGlobal.Icon({
        iconUrl: "/red_one.png", // Đường dẫn tương đối từ thư mục /public
        iconSize: [38, 38] as L.PointTuple,
        iconAnchor: [19, 38] as L.PointTuple,
        popupAnchor: [0, -38] as L.PointTuple,
    });

    // Icon màu xanh cho trạm đang được chọn
    blueIcon = new LGlobal.Icon({
        iconUrl: "/blue_one.png", // Đường dẫn tương đối từ thư mục /public
        iconSize: [38, 38] as L.PointTuple,
        iconAnchor: [19, 38] as L.PointTuple,
        popupAnchor: [0, -38] as L.PointTuple,
    });

    // Hàm tạo icon tùy chỉnh cho các cụm marker
    createClusterCustomIcon = (cluster: any): L.DivIcon => {
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
        return LGlobal.divIcon({
            html: `<span style="${style}">${count}</span>`,
            className: "custom-marker-cluster",
            iconSize: LGlobal.point(40, 40, true),
        });
    };
}


// --- Hàm hỗ trợ ---

// Hàm xác định trạng thái chất lượng nước từ chỉ số WQI
function deriveStatusFromWqi(wqi: number | null | undefined): string {
    if (wqi === null || wqi === undefined || isNaN(wqi)) {
        return "Không xác định";
    }
    // Dựa trên QCVN 08-MT:2015/BTNMT (ví dụ)
    if (wqi > 91) return "Rất Tốt";
    if (wqi > 76) return "Tốt";
    if (wqi > 51) return "Trung Bình";
    if (wqi > 26) return "Kém";
    if (wqi >= 0) return "Rất Kém";
    return "Không xác định";
}

// Hàm định dạng thời gian quan trắc
function formatMonitoringTime(timeString: string | undefined | null): string {
    if (!timeString) return "N/A";
    try {
        const date = new Date(timeString);
        if (isNaN(date.getTime())) {
            console.error("Invalid date format received:", timeString);
            return timeString; // Trả về chuỗi gốc nếu không hợp lệ
        }
        return format(date, 'HH:mm, dd/MM/yyyy'); // Định dạng giờ:phút, ngày/tháng/năm
    } catch (e) {
        console.error("Failed to format time:", timeString, e);
        return timeString; // Trả về chuỗi gốc nếu có lỗi
    }
}

// --- Component chính ---
export default function StationsPage() {
    // --- State Variables ---
    const [stations, setStations] = useState<Station[]>([]); // Danh sách tất cả trạm
    const [selectedStation, setSelectedStation] = useState<Station | null>(null); // Trạm đang được chọn
    const [selectedStationDataPoints, setStationDataPoints] = useState<DataPoint[]>([]); // Dữ liệu gốc của trạm được chọn (nếu cần)
    const [historicalDataPoints, setHistoricalDataPoints] = useState<DataPoint[]>([]); // Dữ liệu lịch sử
    // *** THAY ĐỔI: State mới để lưu dữ liệu dự đoán đã nhóm theo source ***
    const [groupedPredictionDataPoints, setGroupedPredictionDataPoints] = useState<Map<string, DataPoint[]>>(new Map());
    const [latestDataPoint, setLatestDataPoint] = useState<DataPoint | null>(null); // Điểm dữ liệu lịch sử mới nhất
    const [isLoadingStations, setIsLoadingStations] = useState<boolean>(true); // Trạng thái tải danh sách trạm
    const [isLoadingDataPoints, setIsLoadingDataPoints] = useState<boolean>(false); // Trạng thái tải dữ liệu chi tiết của trạm
    const [error, setError] = useState<string | null>(null); // Lỗi (chung hoặc khi tải dữ liệu trạm)
    const [selectedFeature, setSelectedFeature] = useState<string>("pH"); // Chỉ số đang chọn để hiển thị trên biểu đồ
    const [currentPage, setCurrentPage] = useState<number>(1); // Trang hiện tại của bảng
    const [searchTerm, setSearchTerm] = useState<string>(""); // Từ khóa tìm kiếm trong bảng
    const [initialSelectionAttempted, setInitialSelectionAttempted] = useState(false); // Cờ đánh dấu đã thử chọn trạm từ URL hay chưa

    // --- Constants ---
    const itemsPerPage = 5; // Số trạm trên mỗi trang của bảng
    const initialMapCenter: L.LatLngExpression = [10.8231, 106.6297]; // Tọa độ trung tâm bản đồ mặc định (TP.HCM)
    const initialMapZoom = 10; // Mức zoom bản đồ mặc định
    const SELECTED_STATION_ZOOM = 14; // Mức zoom khi chọn một trạm

    // --- Hook để lấy search params từ URL ---
    const searchParams = useSearchParams();
    const router = useRouter();
    // --- Data Fetching Effects ---

    // Effect: Tải danh sách tất cả các trạm khi component được mount lần đầu
    useEffect(() => {
        setIsLoadingStations(true);
        setError(null);
        setInitialSelectionAttempted(false); // Reset cờ khi tải lại danh sách
        getStations({ limit: 1000 }) // Lấy nhiều trạm, cân nhắc pagination ở backend nếu quá lớn
            .then(data => {
                setStations(data);
                // Việc chọn trạm ban đầu (nếu có ID từ URL) sẽ được xử lý ở useEffect khác
            })
            .catch(err => {
                console.error("Failed to fetch stations:", err);
                setError("Không thể tải danh sách trạm quan trắc. Vui lòng thử lại.");
            })
            .finally(() => {
                setIsLoadingStations(false);
            });
    }, []); // Mảng dependency rỗng = chỉ chạy 1 lần khi mount

    // --- Event Handlers ---

    // Hàm xử lý khi chọn một trạm (từ bảng hoặc map)
    // Sử dụng useCallback để tối ưu, vì hàm này là dependency của useEffect khác
    const handleSelectStation = useCallback((station: Station) => {
        // Chỉ xử lý nếu chọn một trạm khác với trạm đang chọn
        if (selectedStation?.id !== station.id) {
            console.log(`Selecting station: ${station.name} (ID: ${station.id})`);
            setSelectedStation(station); // Cập nhật state trạm được chọn

            // Tìm vị trí của trạm trong danh sách gốc để tính toán trang
            const stationIndex = stations.findIndex(s => s.id === station.id);
            if (stationIndex !== -1) {
                // Tính trang mục tiêu (bắt đầu từ 1)
                const targetPage = Math.ceil((stationIndex + 1) / itemsPerPage);
                // setCurrentPage(targetPage); // Cập nhật trang hiện tại của bảng
            } else {
                 // setCurrentPage(1); // Nếu không tìm thấy, về trang 1
            }
             const newUrl = `/stationsadmin?id=${station.id}`;
             router.push(newUrl, { scroll: false });
        }
    }, [selectedStation?.id, stations, itemsPerPage, router]); // Dependencies của useCallback

    // *** Effect MỚI: Xử lý chọn trạm dựa trên tham số 'id' từ URL ***
    useEffect(() => {
        if (!isLoadingStations && stations.length > 0 && searchParams && !selectedStation && !initialSelectionAttempted) {
            const stationIdFromUrl = searchParams.get('id');
            let stationToSelect: Station | undefined = undefined;

            if (stationIdFromUrl) {
                stationToSelect = stations.find(s => s.id === stationIdFromUrl);
                if (stationToSelect) {
                    console.log(`Found station from URL parameter "id": ${stationToSelect.name}`);
                } else {
                    console.warn(`Station ID "${stationIdFromUrl}" from URL parameter "id" not found.`);
                }
            } else {
                const firstKey = searchParams.keys().next().value;
                if (firstKey) {
                    console.warn(`URL parameter "id" not found. Attempting to use first query key "${firstKey}" as ID.`);
                    stationToSelect = stations.find(s => s.id === firstKey);
                    if (stationToSelect) {
                        console.log(`Found station from URL (first key fallback): ${stationToSelect.name}`);
                    } else {
                        console.warn(`Station ID "${firstKey}" from first URL query key not found.`);
                    }
                }
            }

            if (stationToSelect) {
                handleSelectStation(stationToSelect);
            }
            setInitialSelectionAttempted(true);
        }
    }, [
        stations,
        searchParams,
        selectedStation,
        isLoadingStations,
        initialSelectionAttempted,
        handleSelectStation
    ]);

    // *** THAY ĐỔI TRONG useEffect TẢI DỮ LIỆU CHI TIẾT ***
    // Effect: Tải dữ liệu chi tiết (data points) khi `selectedStation` thay đổi
    useEffect(() => {
        if (!selectedStation) {
            setStationDataPoints([]);
            setHistoricalDataPoints([]);
            // THAY ĐỔI: Reset state mới
            setGroupedPredictionDataPoints(new Map());
            setLatestDataPoint(null);
            setError(null);
            return;
        }

        setIsLoadingDataPoints(true);
        setError(null);
        setStationDataPoints([]);
        setHistoricalDataPoints([]);
        // THAY ĐỔI: Reset state mới
        setGroupedPredictionDataPoints(new Map());
        setLatestDataPoint(null);

        const queryOptions: QueryOptions = { limit: 200, sortBy: "monitoring_time", sortDesc: true };

        getDataPointsOfStationById(selectedStation.id, queryOptions)
            .then(data => {
                setStationDataPoints(data); // Lưu dữ liệu gốc (nếu cần)

                const historical: DataPoint[] = [];
                // THAY ĐỔI: Sử dụng Map để nhóm dự đoán theo source
                const predictionsBySource = new Map<string, DataPoint[]>();

                data.forEach(dp => {
                    try {
                        // Validate timestamp ngay từ đầu
                        const monitoringDate = new Date(dp.monitoringTime);
                        if (isNaN(monitoringDate.getTime())) {
                            console.warn("Invalid monitoringTime found, skipping DataPoint:", dp.monitoringTime, "for ID:", dp.id);
                            return; // Bỏ qua điểm dữ liệu này
                        }

                        if (dp.observationType === 'OBSERVATION_TYPE_PREDICTED') {
                            const source = dp.source || "Unknown"; // Gán source mặc định nếu thiếu
                            if (!predictionsBySource.has(source)) {
                                predictionsBySource.set(source, []);
                            }
                            // ?.push() an toàn hơn nếu get trả về undefined (dù không nên xảy ra với logic này)
                            predictionsBySource.get(source)?.push(dp);
                        } else {
                            historical.push(dp);
                        }
                    } catch (e) {
                         console.error("Error processing raw data point:", dp, e);
                    }
                });

                // Điểm lịch sử mới nhất là điểm đầu tiên trong mảng historical (do đã sort desc từ API)
                const latestHistorical = historical.length > 0 ? historical[0] : null;

                // Sắp xếp lịch sử tăng dần (ASC)
                const sortedHistoricalAsc = [...historical].sort((a, b) => new Date(a.monitoringTime).getTime() - new Date(b.monitoringTime).getTime());

                // Sắp xếp từng nhóm dự đoán tăng dần (ASC)
                const sortedGroupedPredictions = new Map<string, DataPoint[]>();
                predictionsBySource.forEach((points, source) => {
                    const sortedPoints = [...points].sort((a, b) => new Date(a.monitoringTime).getTime() - new Date(b.monitoringTime).getTime());
                    sortedGroupedPredictions.set(source, sortedPoints);
                });

                console.log("Processed Historical Data (ASC):", sortedHistoricalAsc.length);
                console.log("Processed Grouped Prediction Data (ASC):", sortedGroupedPredictions);

                // Cập nhật state
                setHistoricalDataPoints(sortedHistoricalAsc);
                // THAY ĐỔI: Cập nhật state mới
                setGroupedPredictionDataPoints(sortedGroupedPredictions);
                setLatestDataPoint(latestHistorical); // Cập nhật điểm dữ liệu mới nhất
                setSelectedFeature("pH"); // Reset về WQI khi chọn trạm mới
            })
            .catch(err => {
                console.error(`Failed to fetch data points for station ${selectedStation.id}:`, err);
                setError(`Không thể tải dữ liệu cho trạm ${selectedStation.name}.`);
                // Đảm bảo xóa dữ liệu khi có lỗi
                setLatestDataPoint(null);
                setHistoricalDataPoints([]);
                 // THAY ĐỔI: Reset state mới
                setGroupedPredictionDataPoints(new Map());
            })
            .finally(() => {
                // Kết thúc trạng thái tải
                setIsLoadingDataPoints(false);
            });
    }, [selectedStation]); // Dependency: Chạy lại effect này mỗi khi `selectedStation` thay đổi


    // --- Memoized Calculations (Tính toán được ghi nhớ để tối ưu) ---

    // Lọc danh sách trạm dựa trên từ khóa tìm kiếm
    const filteredStations = useMemo(() => {
        if (!searchTerm) {
            return stations; // Trả về tất cả nếu không có từ khóa
        }
        const lowerCaseSearch = searchTerm.toLowerCase();
        return stations.filter(station =>
            station.name.toLowerCase().includes(lowerCaseSearch) // Lọc theo tên trạm
        );
    }, [stations, searchTerm]); // Tính lại khi `stations` hoặc `searchTerm` thay đổi

    // Tạo danh sách các trạm duy nhất dựa trên tọa độ để tránh trùng lặp marker trên map
    const uniqueStations = useMemo(() => {
        const coordMap = new Map<string, Station>();
        stations.forEach((station) => {
            // Kiểm tra tọa độ hợp lệ
            if (typeof station.latitude === 'number' && typeof station.longitude === 'number' && !isNaN(station.latitude) && !isNaN(station.longitude)) {
                const coordKey = `${station.latitude.toFixed(5)},${station.longitude.toFixed(5)}`; // Tạo key từ tọa độ (làm tròn)
                if (!coordMap.has(coordKey)) {
                    coordMap.set(coordKey, station); // Chỉ thêm nếu tọa độ chưa tồn tại
                }
            } else {
                console.warn(`Station ${station.id} (${station.name}) has invalid coordinates:`, station.latitude, station.longitude);
            }
        });
        return Array.from(coordMap.values()); // Chuyển Map thành Array
    }, [stations]); // Tính lại khi `stations` thay đổi

    // Tính toán thông tin chi tiết (WQI, status, recommendation) cho trạm đang chọn
    const selectedStationInfo = useMemo(() => {
        if (!latestDataPoint) {
            // Trả về giá trị mặc định nếu không có dữ liệu mới nhất
            return { wqi: null, status: "Không xác định", time: "N/A", recommendation: "Không có dữ liệu mới nhất." };
        }
        const wqi = latestDataPoint.wqi;
        const status = deriveStatusFromWqi(wqi);
        let recommendation = "Chất lượng nước tốt."; // Khuyến nghị mặc định

        // Điều chỉnh khuyến nghị dựa trên trạng thái
        if (status === "Rất Kém") recommendation = "Nước ô nhiễm nặng, chỉ thích hợp cho giao thông thủy và các mục đích tương đương. Cần có biện pháp xử lý và cảnh báo.";
        else if (status === "Kém") recommendation = "Chất lượng nước kém, chỉ sử dụng cho mục đích giao thông thủy và các mục đích tương đương khác.";
        else if (status === "Trung Bình") recommendation = "Chất lượng nước trung bình, sử dụng cho mục đích tưới tiêu và các mục đích tương đương khác.";
        else if (status === "Tốt") recommendation = "Chất lượng nước tốt, có thể sử dụng cho mục đích cấp nước sinh hoạt nhưng cần các biện pháp xử lý phù hợp.";
        else if (status === "Rất Tốt") recommendation = "Chất lượng nước rất tốt, sử dụng tốt cho mục đích cấp nước sinh hoạt.";

        return {
            wqi: wqi ?? "N/A", // Hiển thị "N/A" nếu WQI là null/undefined
            status: status,
            time: formatMonitoringTime(latestDataPoint.monitoringTime), // Định dạng thời gian
            recommendation: recommendation
        };
    }, [latestDataPoint]); // Tính lại khi `latestDataPoint` thay đổi

    // Xác định các chỉ số thành phần có sẵn để chọn trên biểu đồ
    const availableFeatures = useMemo(() => {
        const desiredFeatures = ['pH', 'DO', 'N-NO2', 'N-NH4', 'P-PO4', 'TSS', 'COD', 'EC', 'AH']; // Danh sách chỉ số mong muốn và thứ tự
        if (!latestDataPoint || !latestDataPoint.features || latestDataPoint.features.length === 0) {
            return []; // Không có chỉ số nếu không có dữ liệu hoặc mảng features rỗng
        }
        // Lấy tên các chỉ số duy nhất có trong điểm dữ liệu mới nhất
        const actualFeatureNames = new Set(latestDataPoint.features.map(f => f.name).filter((name): name is string => !!name));
        // Chỉ trả về những chỉ số mong muốn mà thực sự có trong dữ liệu
        return desiredFeatures.filter(desiredName => actualFeatureNames.has(desiredName));
    }, [latestDataPoint]); // Tính lại khi `latestDataPoint` thay đổi

    // *** THAY ĐỔI TRONG useMemo chuẩn bị dữ liệu cho Chartline ***
    const chartInputData = useMemo(() => {
        // Chỉ xử lý nếu có dữ liệu lịch sử hoặc dự đoán
        if (historicalDataPoints.length === 0 && groupedPredictionDataPoints.size === 0) return null;

        // Trả về dữ liệu thô đã sắp xếp và nhóm, Chartline sẽ xử lý chi tiết hơn
        return {
            historicalDataPoints: historicalDataPoints,         // Dữ liệu lịch sử đã sắp xếp ASC
            groupedPredictionDataPoints: groupedPredictionDataPoints, // Dữ liệu dự đoán đã nhóm và sắp xếp ASC
            selectedFeature: selectedFeature                   // Chỉ số đang được chọn
        };
    }, [historicalDataPoints, groupedPredictionDataPoints, selectedFeature]); // Dependencies


    // Hàm xử lý khi thay đổi trang của bảng
    const handlePageChange = (page: number) => {
        // Tính tổng số trang dựa trên danh sách đã lọc
        const currentTotalPages = Math.ceil(filteredStations.length / itemsPerPage);
        // Đảm bảo trang nằm trong giới hạn hợp lệ
        if (page > 0 && page <= (currentTotalPages > 0 ? currentTotalPages : 1) ) {
            setCurrentPage(page);
        }
    };

    // Hàm xử lý khi thay đổi chỉ số hiển thị trên biểu đồ
    const handleFeatureChange = (event: ChangeEvent<HTMLSelectElement>) => {
        setSelectedFeature(event.target.value);
    };

    // Hàm xử lý khi thay đổi nội dung ô tìm kiếm
    const handleSearchChange = (event: ChangeEvent<HTMLInputElement>) => {
        setSearchTerm(event.target.value);
        setCurrentPage(1); // Reset về trang 1 khi tìm kiếm
    };

    // --- Render Logic ---

    // Hiển thị loader chính nếu đang tải danh sách trạm ban đầu
    if (isLoadingStations) {
        return <PageLoader message="Đang tải danh sách trạm..." />;
    }

    // Hiển thị lỗi nếu tải danh sách trạm thất bại và không có trạm nào
    if (error && stations.length === 0 && !isLoadingStations) {
        return <div className="flex justify-center items-center h-screen text-red-600 font-semibold p-4">{error}</div>;
    }

    // Tính toán các biến phân trang dựa trên danh sách đã lọc (filteredStations)
    const totalPages = Math.ceil(filteredStations.length / itemsPerPage);
    // Đảm bảo trang hiện tại không vượt quá tổng số trang (an toàn khi lọc làm giảm số trang)
    const safeCurrentPage = Math.max(1, Math.min(currentPage, totalPages > 0 ? totalPages : 1));
    // Lấy danh sách trạm cho trang hiện tại
    const paginatedStations = filteredStations.slice(
        (safeCurrentPage - 1) * itemsPerPage,
        safeCurrentPage * itemsPerPage
    );

    // Trả về JSX để render giao diện
    return (
        <div className="flex flex-1 overflow-hidden"> {/* Container chính, chiếm hết không gian flex */}
            <div className="flex flex-col flex-grow overflow-y-auto space-y-4 p-4 bg-gray-50"> {/* Khu vực nội dung cuộn được */}

                {/* Header của trang */}
                <header className="flex justify-between items-center border-b pb-2 mb-4 bg-white p-4 rounded-lg shadow-sm">
                    <h1 className="text-2xl font-bold text-gray-800">Trạm Quan Trắc Chất Lượng Nước</h1>
                    {/* Hiển thị lỗi nếu có lỗi khi tải dữ liệu chi tiết của trạm đang chọn */}
                    {error && !isLoadingDataPoints && selectedStation && <span className="text-sm text-red-500">{error}</span>}
                 </header>

                 {/* Layout chứa Map và Bảng */}
                <div className="flex flex-col md:flex-row flex-grow gap-4 min-h-[60vh]">

                    {/* Khu vực bản đồ */}
                    <div className="flex-grow md:w-2/3 lg:w-3/4 min-w-0 relative z-10 h-[50vh] md:h-auto border rounded-lg shadow-md overflow-hidden">
                        {typeof window !== 'undefined' && (
                            <MapContainer
                                key="my-leaflet-map"
                                center={initialMapCenter}
                                zoom={initialMapZoom}
                                style={{ height: "100%", width: "100%" }}
                                scrollWheelZoom={true}
                            >
                                <TileLayer
                                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors'
                                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                />
                                <MapUpdater station={selectedStation} zoomLevel={SELECTED_STATION_ZOOM} />

                                {createClusterCustomIcon && uniqueStations.length > 0 && (
                                    <MarkerClusterGroup
                                        chunkedLoading
                                        iconCreateFunction={createClusterCustomIcon}
                                        maxClusterRadius={60}
                                    >
                                        {uniqueStations.map((station) => {
                                            const isValidPosition = typeof station.latitude === 'number' && typeof station.longitude === 'number';
                                            if (!isValidPosition) return null;
                                            const isSelected = selectedStation?.id === station.id;
                                            return (
                                                <Marker
                                                    key={station.id}
                                                    position={[station.latitude, station.longitude]}
                                                    icon={isSelected ? blueIcon : redIcon}
                                                    eventHandlers={{
                                                        click: () => handleSelectStation(station),
                                                    }}
                                                    zIndexOffset={isSelected ? 1000 : 0}
                                                >
                                                    <Popup minWidth={180}>
                                                        <div className="text-sm">
                                                            <h3 className="text-md font-bold mb-1">{station.name}</h3>
                                                            <p className="text-xs text-gray-600 mb-1">
                                                                {station.location || `(${station.latitude.toFixed(4)}, ${station.longitude.toFixed(4)})`}
                                                            </p>
                                                            {isSelected && latestDataPoint && !isLoadingDataPoints && !error && (
                                                                <>
                                                                    <hr className="my-1" />
                                                                    <p>WQI: <span className="font-bold text-blue-600">{selectedStationInfo.wqi}</span></p>
                                                                    <p className={`${getStatusTextColor(selectedStationInfo.status)}`}>Trạng thái: {selectedStationInfo.status}</p>
                                                                    <p className="text-xs text-gray-500 mt-1">Lúc: {selectedStationInfo.time}</p>
                                                                </>
                                                            )}
                                                            {isSelected && isLoadingDataPoints && (<p className="text-xs text-gray-500 italic mt-1">Đang tải dữ liệu...</p>)}
                                                            {isSelected && !isLoadingDataPoints && error && (<p className="text-xs text-red-500 italic mt-1">{error}</p>)}
                                                            {isSelected && !isLoadingDataPoints && !latestDataPoint && !error && (<p className="text-xs text-orange-500 italic mt-1">Không có dữ liệu mới.</p>)}
                                                            {!isSelected && (<p className="text-xs text-blue-500 italic mt-2">Nhấn để xem chi tiết</p>)}
                                                         </div>
                                                     </Popup>
                                                 </Marker>
                                             );
                                         })}
                                     </MarkerClusterGroup>
                                 )}
                             </MapContainer>
                         )}
                     </div> {/* Kết thúc khu vực bản đồ */}

                     {/* Khu vực bảng danh sách trạm */}
                    <div className="w-full md:w-1/3 lg:w-1/4 p-0 min-w-[300px] flex flex-col border rounded-lg shadow-md bg-white">
                         {/* Ô tìm kiếm */}
                        <div className="p-3 border-b">
                            <Input
                                type="text"
                                placeholder="Tìm kiếm theo tên trạm..."
                                value={searchTerm}
                                onChange={handleSearchChange}
                                className="w-full text-sm"
                             />
                         </div>

                         {/* Khu vực bảng có thể cuộn */}
                        <div className="flex-grow overflow-y-auto">
                            <Table>
                                {/* Header của bảng */}
                                <TableHeader className="sticky top-0 bg-gray-100 z-10 shadow-sm"><TableRow><TableHead className="py-2 px-3 text-sm font-semibold text-gray-600">Tên Trạm</TableHead><TableHead className="text-right py-2 px-3 text-sm font-semibold text-gray-600">Vị Trí</TableHead></TableRow></TableHeader>
                                <TableBody>
                                    {isLoadingStations ? (
                                         <TableRow><TableCell colSpan={2} className="text-center text-gray-500 py-4">Đang tải trạm...</TableCell></TableRow>
                                     ) : paginatedStations.length > 0 ? (
                                         paginatedStations.map((station) => {
                                             const isSelected = selectedStation?.id === station.id;
                                             return (
                                                 <TableRow
                                                     key={station.id}
                                                     onClick={() => handleSelectStation(station)}
                                                     className={`cursor-pointer transition-colors duration-150 ${isSelected ? "bg-blue-100 hover:bg-blue-200" : "hover:bg-gray-50"}`}
                                                 >
                                                     <TableCell className={`font-medium py-2 px-3 text-sm ${isSelected ? 'text-blue-800' : ''}`}>{station.name}</TableCell>
                                                     <TableCell className="text-right text-xs text-gray-500 py-2 px-3">
                                                         {station.location || (typeof station.latitude === 'number' && typeof station.longitude === 'number' ? `(${station.latitude.toFixed(2)}, ${station.longitude.toFixed(2)})` : 'Không rõ')}
                                                     </TableCell>
                                                 </TableRow>
                                             );
                                         })
                                     ) : (
                                          <TableRow><TableCell colSpan={2} className="text-center text-gray-500 py-4">{(searchTerm && filteredStations.length === 0) ? 'Không tìm thấy trạm phù hợp.' : 'Không có trạm nào.'}</TableCell></TableRow>
                                      )}
                                  </TableBody>
                              </Table>
                          </div> {/* Kết thúc khu vực bảng cuộn */}

                         {/* Khu vực phân trang */}
                         {totalPages > 1 && (
                            <div className="mt-auto p-2 border-t bg-gray-50">
                                <Pagination
                                    currentPage={safeCurrentPage}
                                    totalPages={totalPages}
                                    onPageChange={handlePageChange}
                                    siblingCount={0}
                                />
                            </div>
                         )}
                     </div> {/* Kết thúc khu vực bảng */}
                </div> {/* Kết thúc layout Map và Bảng */}

                 {/* Khu vực chi tiết trạm */}
                <div className="w-full mt-6">
                     {isLoadingDataPoints && selectedStation && (
                         <PageLoader message={`Đang tải dữ liệu chi tiết cho trạm ${selectedStation.name}...`} />
                     )}
                    {!isLoadingDataPoints && selectedStation && latestDataPoint && !error && (
                        <StationDetails
                            selectedStation={{
                                ...(selectedStation as Station),
                                ...selectedStationInfo
                            }}
                            latestDataPoint={latestDataPoint}
                            availableFeatures={availableFeatures}
                        />
                    )}
                    {!isLoadingDataPoints && selectedStation && !latestDataPoint && !error && (
                         <div className="text-center text-gray-600 mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg shadow-sm">Không có dữ liệu đo gần đây cho trạm {selectedStation.name}.</div>
                    )}
                    {!isLoadingDataPoints && selectedStation && error && (
                         <div className="text-center text-red-600 mt-4 p-4 bg-red-50 border border-red-200 rounded-lg shadow-sm">{error}</div>
                    )}
                    {!selectedStation && !isLoadingStations && stations.length > 0 && (
                         <div className="text-center text-gray-500 mt-4 p-4 bg-gray-100 border rounded-lg shadow-sm">Chọn một trạm trên bản đồ hoặc từ bảng bên trái để xem chi tiết và biểu đồ.</div>
                     )}
                     {!isLoadingStations && stations.length === 0 && error && (
                         <div className="text-center text-red-600 mt-4 p-4 bg-red-50 border border-red-200 rounded-lg shadow-sm">Không thể tải danh sách trạm. Vui lòng kiểm tra kết nối và thử lại.</div>
                     )}
                </div> {/* Kết thúc khu vực chi tiết trạm */}

                 {/* Khu vực biểu đồ */}
                <div className="mt-6 w-full p-4 border rounded-lg shadow-md bg-white">
                    <h2 className="text-xl font-semibold mb-4 text-center text-gray-700">Biểu đồ diễn biến chất lượng nước</h2>
                    {isLoadingDataPoints && selectedStation && (
                         <div className="h-80 flex justify-center items-center bg-gray-50 rounded-lg"><p className="text-gray-500 italic">Đang tải dữ liệu biểu đồ...</p></div>
                     )}
                     {!isLoadingDataPoints && selectedStation && (
                         <>
                             {!error ? (
                                 <>
                                     {/* Kiểm tra xem có dữ liệu đầu vào cho biểu đồ không */}
                                    {(chartInputData && (chartInputData.historicalDataPoints.length > 0 || chartInputData.groupedPredictionDataPoints.size > 0)) ? (
                                         <>
                                             {/* Dropdown chọn chỉ số */}
                                            <div className="flex justify-center mb-4">
                                                <select
                                                     value={selectedFeature}
                                                     onChange={handleFeatureChange}
                                                     className="border rounded-md p-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-150 ease-in-out text-sm"
                                                     aria-label="Chọn chỉ số hiển thị trên biểu đồ"
                                                 >
                                                     
                                                     {availableFeatures.map((featureName) => (
                                                         <option key={featureName} value={featureName}>Chỉ số {featureName}</option>
                                                     ))}
                                                     <option value="WQI">Chỉ số chất lượng nước (WQI)</option>
                                                     {availableFeatures.length === 0 && latestDataPoint && <option disabled>Không có chỉ số thành phần</option>}
                                                 </select>
                                             </div>
                                             {/* Component biểu đồ - Truyền props mới */}
                                            <div className="relative h-100"> {/* Tăng chiều cao biểu đồ */}
                                                <Chartline
                                                     // *** THAY ĐỔI: Truyền props mới ***
                                                     historicalDataPoints={chartInputData.historicalDataPoints}
                                                     groupedPredictionDataPoints={chartInputData.groupedPredictionDataPoints}
                                                     selectedFeature={chartInputData.selectedFeature}
                                                     title={`${selectedFeature === 'WQI' ? 'WQI' : `Chỉ số ${selectedFeature}`}`}
                                                 />
                                             </div>
                                         </>
                                     ) : (
                                         /* Thông báo nếu không có đủ dữ liệu để vẽ */
                                         <div className="flex justify-center items-center h-60 bg-gray-50 text-gray-600 font-semibold rounded-lg mt-4 p-4">Không có đủ dữ liệu (lịch sử hoặc dự đoán) để vẽ biểu đồ cho trạm này.</div>
                                     )}
                                 </>
                             ) : (
                                  /* Hiển thị lỗi nếu có lỗi xảy ra khi tải dữ liệu */
                                  <div className="flex justify-center items-center h-60 bg-red-50 text-red-700 font-semibold rounded-lg mt-4 p-4 border border-red-200">{error}</div>
                              )}
                          </>
                      )}
                      {!selectedStation && (
                           <div className="flex justify-center items-center h-60 bg-gray-50 text-gray-500 font-semibold rounded-lg mt-4 p-4">Vui lòng chọn một trạm để xem biểu đồ.</div>
                       )}
                 </div> {/* Kết thúc khu vực biểu đồ */}

             </div> {/* Kết thúc khu vực nội dung chính */}
         </div> // Kết thúc container chính
     );
}