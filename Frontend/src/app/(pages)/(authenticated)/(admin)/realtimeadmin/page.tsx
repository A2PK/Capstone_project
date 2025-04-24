"use client";

import React, { useEffect, useState, useMemo, useCallback } from "react";
import { format, parseISO, isValid as isValidDate } from "date-fns";
import type { DateRange, SelectRangeEventHandler } from "react-day-picker";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import * as XLSX from "xlsx";
import Link from "next/link";

// --- UI Components ---
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { Pagination } from "@/components/pagination";
// *** QUAN TRỌNG: Bạn cần tạo hoặc cập nhật component AlertConfigModal ***
import { AlertConfigModal } from '@/components/alert-modal'; // Giả định đường dẫn đúng
import PageLoader from "@/components/pageloader";

// --- Icons ---
import { CalendarIcon, SettingsIcon, DownloadIcon, ChevronDown, ArrowUp, ArrowDown } from "lucide-react";

// --- Libs & Hooks ---
import { cn, getDonvi, getStatusTextColor } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

// --- API & Types ---
import { Station, DataPoint, Indicator, QueryOptions } from "@/types/station2";
import { getStations, getAllDataPoints } from "@/lib/station";
// --- Threshold Imports ---
import {
    getAllThresholdConfigs,
    updateThresholdConfigs,
    createThresholdConfigs,
    deleteThresholdConfigs
} from "@/lib/threshold";
import { Thresholds, ElementRange, CreateElementRangeDto, CreateThresholdsDto, DeleteThresholdsDto } from "@/types/threshold"; // Đảm bảo types đúng

// --- Font Import ---
import "../../../../fonts/times"; // Đảm bảo font được import đúng cách

// --- Constants ---
// Danh sách các header cho bảng (bao gồm cả WQI nếu muốn hiển thị)
const metricHeaders = ["pH", "EC", "DO", "N-NH4", "N-NO2", "P-PO4", "TSS", "COD", "AH", "WQI"];
// Danh sách tên element hợp lệ để cấu hình ngưỡng (Lấy từ yêu cầu của bạn, loại bỏ WQI và AH nếu không cần)
const validThresholdElementNames = ["AH","pH", "EC", "DO", "N-NH4", "N-NO2", "P-PO4", "TSS", "COD"];

// --- Helper Functions ---
function deriveStatusFromWqi(wqi: number | null | undefined): string {
    if (wqi === null || wqi === undefined || isNaN(wqi)) return "Không xác định";
    if (wqi > 91) return "Rất Tốt";
    if (wqi > 76) return "Tốt";
    if (wqi > 51) return "Trung Bình";
    if (wqi > 26) return "Kém";
    if (wqi > 0) return "Rất Kém";
    return "Không xác định";
}

function deriveRecommendationFromWqi(wqi: number | null | undefined): string {
    if (wqi === null || wqi === undefined || isNaN(wqi)) return "Không có dữ liệu WQI.";
    const status = deriveStatusFromWqi(wqi);
    if (status === "Rất Kém") return "Nước ô nhiễm nặng, chỉ thích hợp cho giao thông thủy. Cần xử lý và cảnh báo.";
    if (status === "Kém") return "Chất lượng nước kém, chỉ sử dụng cho giao thông thủy.";
    if (status === "Trung Bình") return "Chất lượng nước trung bình, sử dụng cho tưới tiêu.";
    if (status === "Tốt") return "Chất lượng nước tốt, có thể dùng cho cấp nước sinh hoạt (cần xử lý).";
    if (status === "Rất Tốt") return "Chất lượng nước rất tốt, sử dụng tốt cho cấp nước sinh hoạt.";
    return "Không có khuyến nghị.";
}

function formatMonitoringTimestamp(timeString: string | undefined | null): string {
    if (!timeString) return "N/A";
    try {
        const date = parseISO(timeString);
        if (!isValidDate(date)) return timeString;
        //return format(date, 'HH:mm, dd/MM/yyyy'); // Giữ lại giờ phút nếu cần
         return format(date, 'dd/MM/yyyy');
    } catch (e) { return timeString; }
}

// --- Types ---
interface SortConfig {
    key: string;
    direction: 'ascending' | 'descending';
}

interface DisplayRowData extends DataPoint {
    stationName: string;
    stationLocation: string;
    status: string;
    recommendation: string;
}

// --- Component ---
export default function Realtimedata() {
    // --- State ---
    const [isLoading, setIsLoading] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false); // State loading cho các thao tác API (CUD)
    const [error, setError] = useState<string | null>(null);
    const [allStations, setAllStations] = useState<Station[]>([]);
    const [stationMap, setStationMap] = useState<Map<string, Station>>(new Map());
    const [allDataPoints, setAllDataPoints] = useState<DataPoint[]>([]);
    const [displayData, setDisplayData] = useState<DisplayRowData[]>([]);
    const [filteredDisplayData, setFilteredDisplayData] = useState<DisplayRowData[]>([]);

    // --- Threshold State ---
    const [thresholdConfigs, setThresholdConfigs] = useState<ElementRange[]>([]); // State lưu threshold configs từ backend

    // Filters
    const [date, setDate] = useState<DateRange | undefined>(undefined);
    const [selectedStationId, setSelectedStationId] = useState<string | null>(null);
    const [selectedStatus, setSelectedStatus] = useState<string | null>(null);

    // Sorting
    const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'monitoringTime', direction: 'descending' });

    // Pagination
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [itemsPerPage, setItemsPerPage] = useState<number>(12);

    // UI State
    const [isExportOpen, setIsExportOpen] = useState(false);
    const [isAlertModalOpen, setIsAlertModalOpen] = useState(false);

    // Other Hooks
    const { toast } = useToast();

    // --- Data Fetching (Initial Load) ---
    const fetchData = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            // Fetch Stations
            const stationsPromise = getStations({ limit: 1000 });
            // Fetch DataPoints
            const dataPointsPromise = getAllDataPoints({
                options: {
                    filters: { observation_type: 'actual' },
                    sortBy: 'monitoring_time',
                    sortDesc: true,
                    limit: 10000 // Consider backend pagination for large datasets
                }
            });
            // --- Fetch Thresholds ---
            const thresholdsPromise = getAllThresholdConfigs(); // Gọi API lấy threshold

            // Wait for all promises to resolve
            const [stations, dataPointsResponse, fetchedThresholds] = await Promise.all([
                stationsPromise,
                dataPointsPromise,
                thresholdsPromise
            ]);

            // Process Stations
            setAllStations(stations);
            const newStationMap = new Map(stations.map(s => [s.id, s]));
            setStationMap(newStationMap);

            // Process DataPoints
            setAllDataPoints(dataPointsResponse);

            // Process Thresholds
            setThresholdConfigs(fetchedThresholds);
            console.log("Fetched Thresholds:", fetchedThresholds); // Log để kiểm tra

        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : "Lỗi không xác định khi tải dữ liệu.";
            console.error("Failed to fetch initial data:", err);
            setError(errorMsg);
            toast({ variant: "destructive", title: "Lỗi tải dữ liệu", description: errorMsg });
        } finally {
            setIsLoading(false);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [toast]); // Dependency on toast

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    // --- Data Combination ---
    useEffect(() => {
        if (stationMap.size === 0 || allDataPoints.length === 0) {
            setDisplayData([]);
            return;
        }
        const combinedData = allDataPoints.map(dp => {
            const station = stationMap.get(dp.stationId);
            const wqi = dp.wqi;
            return {
                ...dp,
                stationName: station?.name ?? 'N/A',
                stationLocation: station?.location ?? '',
                status: deriveStatusFromWqi(wqi),
                recommendation: deriveRecommendationFromWqi(wqi),
            };
        });
        setDisplayData(combinedData);
    }, [allDataPoints, stationMap]);

    // --- Filtering Logic ---
    const applyFilters = useCallback(() => {
        let dataToFilter = [...displayData];
        const fromDate = date?.from ? new Date(date.from) : null;
        const toDate = date?.to ? new Date(date.to) : null;
        if (fromDate) fromDate.setHours(0, 0, 0, 0);
        if (toDate) toDate.setHours(23, 59, 59, 999);

        dataToFilter = dataToFilter.filter(item => {
            const stationMatch = selectedStationId ? item.stationId === selectedStationId : true;
            const statusMatch = selectedStatus ? item.status === selectedStatus : true;
            let dateMatch = true;
            if (fromDate || toDate) {
                try {
                    const itemDate = parseISO(item.monitoringTime);
                    if (!isValidDate(itemDate)) dateMatch = false;
                    else {
                        const fromOk = fromDate ? itemDate >= fromDate : true;
                        const toOk = toDate ? itemDate <= toDate : true;
                        dateMatch = fromOk && toOk;
                    }
                } catch { dateMatch = false; }
            }
            return stationMatch && statusMatch && dateMatch;
        });
        setFilteredDisplayData(dataToFilter);
        setCurrentPage(1); // Reset page on filter change
    }, [displayData, date, selectedStationId, selectedStatus]);

    useEffect(() => {
        applyFilters();
    }, [applyFilters]);

    // --- Sorting Logic ---
    const getSortValue = useCallback((item: DisplayRowData, key: string): number | string | Date | null => {
        if (key === 'monitoringTime') {
            try { return parseISO(item.monitoringTime); } catch { return null; }
        }
        if (key === 'stationName') { return item.stationName; }
        if (key === 'status') { return item.status; }

        let potentialValue: any = null;
        const keyLower = key.toLowerCase();

        if (keyLower === 'wqi') {
            potentialValue = item.wqi;
        }

        if (potentialValue === null || potentialValue === undefined) {
            const indicator = item.features.find(f => f.name.toLowerCase() === keyLower) ||
                              item.features.find(f => f.name.toLowerCase().includes(keyLower));
            if (indicator) {
                potentialValue = indicator.value ?? indicator.textualValue;
            }
        }

        const numericKeys = ["wqi", "ph", "ec", "do", "n-nh4", "n-no2", "p-po4", "tss", "cod"]; // AH is often text

        if (numericKeys.includes(keyLower)) {
             if (potentialValue === null || potentialValue === undefined || potentialValue === '') return null;
             const numValue = parseFloat(String(potentialValue));
             return isNaN(numValue) ? null : numValue;
        } else {
            return potentialValue !== null && potentialValue !== undefined ? String(potentialValue) : null;
        }
    }, []);

    const sortedData = useMemo(() => {
        let sortableItems = [...filteredDisplayData];
        if (sortConfig) {
            sortableItems.sort((a, b) => {
                const valueA = getSortValue(a, sortConfig.key);
                const valueB = getSortValue(b, sortConfig.key);

                if (valueA === null || valueA === undefined) return 1;
                if (valueB === null || valueB === undefined) return -1;

                let comparison = 0;
                if (valueA instanceof Date && valueB instanceof Date) {
                    comparison = valueA.getTime() - valueB.getTime();
                } else if (typeof valueA === 'number' && typeof valueB === 'number') {
                    comparison = valueA - valueB;
                } else if (typeof valueA === 'string' && typeof valueB === 'string') {
                    comparison = valueA.localeCompare(valueB, 'vi');
                }
                return sortConfig.direction === 'ascending' ? comparison : -comparison;
            });
        }
        return sortableItems;
    }, [filteredDisplayData, sortConfig, getSortValue]);

    // --- Pagination Logic ---
    const totalPages = Math.ceil(sortedData.length / itemsPerPage);
    const paginatedData = sortedData.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

    // --- Handlers ---
    const handleResetFilter = () => {
        setDate(undefined);
        setSelectedStationId(null);
        setSelectedStatus(null);
        setSortConfig({ key: 'monitoringTime', direction: 'descending' });
    };

    const handlePageChange = (page: number) => {
        setCurrentPage(page);
    };

    const handleSelectDate: SelectRangeEventHandler = (range) => {
        setDate(range);
    };

    const handleSort = (key: string) => {
        setSortConfig(prevConfig => {
            let direction: 'ascending' | 'descending' = 'ascending';
            if (prevConfig?.key === key && prevConfig.direction === 'ascending') {
                direction = 'descending';
            }
            if (key === 'monitoringTime' && (prevConfig?.key !== key || prevConfig.direction === 'ascending')) {
                 direction = 'descending';
            }
            return { key, direction };
        });
        setCurrentPage(1);
    };

    // --- Alert Checking (Uses Backend Thresholds) ---
    const isValueOutOfRange = useCallback((indicator: Indicator | undefined, metricKey: string): boolean => {
        if (!indicator || !thresholdConfigs || thresholdConfigs.length === 0) {
            return false; // No indicator or thresholds not loaded
        }

        const keyLower = metricKey.toLowerCase(); // Normalize key

        // Find config from backend data
        const config = thresholdConfigs.find(c => c.elementName.toLowerCase() === keyLower);

        if (!config) {
            // console.log(`No threshold config found for: ${metricKey}`);
            return false; // No config for this metric
        }

        // Get numeric value from indicator
        let value: number | undefined;
        if (typeof indicator.value === 'number') {
            value = indicator.value;
        } else if (typeof indicator.textualValue === 'string') {
            const parsed = parseFloat(indicator.textualValue);
            if (!isNaN(parsed)) value = parsed;
        }

        if (value === undefined) return false; // No comparable value

        // Compare with backend min/max
        const { minValue, maxValue } = config;
        // console.log(`Checking ${metricKey} (value: ${value}) against config:`, config);

        if (minValue !== null && value < minValue) return true;
        if (maxValue !== null && value > maxValue) return true;

        return false; // Within range or no valid range defined
    }, [thresholdConfigs]); // Dependency on backend thresholds

    // --- Export Functions ---
    const handleExportXLSX = () => {
        if (sortedData.length === 0) {
            toast({ variant: "warning", title: "Không có dữ liệu để xuất." }); return;
        }
        // Filter out WQI and AH from headers if they are not regular metrics for columns
        const exportHeaders = metricHeaders.filter(h => !['WQI', 'AH'].includes(h)); // Adjust as needed
        const dataForExport = sortedData.map(item => {
            const row: Record<string, any> = {
                'Place': item.stationId,
                'Trạm': item.stationName,
                'Vị trí': item.stationLocation,
                'Thời gian': formatMonitoringTimestamp(item.monitoringTime),
            };
            // Add metric values
            exportHeaders.forEach(header => {
                const keyLower = header.toLowerCase();
                let indicator = item.features.find(f => f.name.toLowerCase() === keyLower) || item.features.find(f => f.name.toLowerCase().includes(keyLower));
                row[header] = indicator ? (indicator.value ?? indicator.textualValue ?? 'N/A') : 'N/A';
            });
            // Add WQI, Status, Recommendation at the end
             row['WQI'] = item.wqi ?? 'N/A';
             row['Trạng thái'] = item.status;
             row['Khuyến nghị'] = item.recommendation;
            return row;
        });

        const headerOrder = ["Place", "Trạm", "Vị trí", "Thời gian", ...exportHeaders, "WQI", "Trạng thái", "Khuyến nghị"];
        const ws = XLSX.utils.json_to_sheet(dataForExport, { header: headerOrder });
        const cols = headerOrder.map(key => ({ wch: key === 'Khuyến nghị' ? 50 : (key === 'Trạm' || key === 'Vị trí' ? 25 : 12) }));
        ws['!cols'] = cols;
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "DuLieuQuanTrac");
        XLSX.writeFile(wb, "du_lieu_quan_trac.xlsx"); // Use .xlsx
        setIsExportOpen(false);
    };

    const handleExportPDF = () => {
         if (sortedData.length === 0) {
            toast({ variant: "warning", title: "Không có dữ liệu để xuất." }); return;
        }
        const doc = new jsPDF({ orientation: "landscape" });
        doc.setFont("timr45w", "normal"); // Make sure this font is loaded correctly
        const pageWidth = doc.internal.pageSize.width;
        const pageHeight = doc.internal.pageSize.height;
        const margin = 10;

        // Title
        doc.setFontSize(16);
        doc.text("BÁO CÁO KẾT QUẢ QUAN TRẮC CHẤT LƯỢNG NƯỚC", pageWidth / 2, margin + 5, { align: 'center' });
        // Date Range
        doc.setFontSize(10);
        if (date?.from) {
            const dateText = `Từ ngày: ${format(date.from, "dd/MM/yyyy")}${date.to ? ` đến ngày: ${format(date.to, "dd/MM/yyyy")}` : ''}`;
            doc.text(dateText, pageWidth / 2, margin + 15, { align: 'center' });
        }
        // Filters
        let filterText = [];
        if (selectedStationId && stationMap.has(selectedStationId)) filterText.push(`Trạm: ${stationMap.get(selectedStationId)?.name}`);
        if (selectedStatus) filterText.push(`Trạng thái: ${selectedStatus}`);
        if (filterText.length > 0) {
            doc.text(filterText.join(" | "), margin, margin + 25);
        }

        // Use all metricHeaders for the PDF table if desired
        const tableHeaders = ["TT", "Trạm", "Thời gian", ...metricHeaders, "Trạng thái", "Khuyến nghị"];
        const tableBody = sortedData.map((item, index) => {
            const rowData: (string | number | null)[] = [
                index + 1,
                item.stationName,
                formatMonitoringTimestamp(item.monitoringTime),
            ];
            metricHeaders.forEach(header => {
                 const keyLower = header.toLowerCase();
                 let value: any = 'N/A';
                 if (header === 'WQI') {
                     value = item.wqi ?? 'N/A';
                 } else {
                     let indicator = item.features.find(f => f.name.toLowerCase() === keyLower) || item.features.find(f => f.name.toLowerCase().includes(keyLower));
                     value = indicator ? (indicator.value ?? indicator.textualValue ?? 'N/A') : 'N/A';
                 }
                 // Optional: Format numbers
                 if (typeof value === 'number') value = value.toFixed(header === 'WQI' ? 1 : 2);
                 rowData.push(value);
            });
            rowData.push(item.status, item.recommendation);
            return rowData;
        });

        autoTable(doc, {
            startY: margin + 30,
            styles: { font: "timr45w", fontSize: 8, cellPadding: 1.5, overflow: 'linebreak' },
            head: [tableHeaders],
            body: tableBody,
            theme: "grid",
            headStyles: { fillColor: [41, 128, 185], textColor: 255, fontSize: 9, halign: 'center', fontStyle: 'bold' },
            columnStyles: {
                0: { cellWidth: 8, halign: 'center' }, // TT
                1: { cellWidth: 25 }, // Trạm
                2: { cellWidth: 18 }, // Thời gian
                // WQI column index (3 + metricHeaders.length - 1)
                [2 + metricHeaders.findIndex(h => h === 'WQI') + 1]: { halign: 'center' }, // Center WQI values if needed
                [tableHeaders.length - 1]: { cellWidth: 50 }, // Khuyến nghị
                [tableHeaders.length - 2]: { cellWidth: 18 }, // Trạng thái
            },
            didDrawPage: function (data) {
                doc.setFontSize(8);
                doc.text(`Trang ${data.pageNumber}`, pageWidth - margin, pageHeight - 5, { align: 'right' });
            },
            margin: { left: margin, right: margin, bottom: 10 },
        });

        doc.save("ket_qua_quan_trac_nuoc.pdf");
        setIsExportOpen(false);
    };

    // --- Threshold API Handlers (Passed to Modal) ---
    const handleFetchThresholds = useCallback(async () => {
        // console.log("Refetching thresholds..."); // Debug
        try {
            const fetchedThresholds = await getAllThresholdConfigs();
            setThresholdConfigs(fetchedThresholds);
            // console.log("Refetched Thresholds:", fetchedThresholds); // Debug
        } catch (error) {
            console.error("Failed to refetch thresholds:", error);
            toast({ variant: "destructive", title: "Lỗi", description: "Không thể tải lại cấu hình ngưỡng." });
        }
    }, [toast]);

    const handleUpdateAllThresholds = useCallback(async (updatedConfigs: ElementRange[]) => {
        setIsProcessing(true); // Show processing state
        // console.log("Updating thresholds with:", updatedConfigs); // Debug
        try {
            const payload: Thresholds = { configs: updatedConfigs.filter(c => c.id) }; // Ensure only configs with ID are sent for update
             if (payload.configs.length === 0) {
                 toast({ variant: "default", title: "Thông tin", description: "Không có thay đổi nào để cập nhật." });
                 setIsAlertModalOpen(false);
                 return;
             }
            await updateThresholdConfigs(payload);
            toast({ variant: "success", title: "Thành công", description: "Đã cập nhật cấu hình ngưỡng." });
            await handleFetchThresholds();
            setIsAlertModalOpen(false);
        } catch (error: any) {
            console.error("Failed to update thresholds:", error);
            toast({ variant: "destructive", title: "Lỗi cập nhật ngưỡng", description: error.message || "Vui lòng thử lại." });
        } finally {
            setIsProcessing(false);
        }
    }, [toast, handleFetchThresholds]);

    const handleCreateThreshold = useCallback(async (newConfigData: CreateElementRangeDto) => {
        setIsProcessing(true);
        // console.log("Creating threshold:", newConfigData); // Debug
        try {
            // Validate basic data
            if (!newConfigData.elementName) throw new Error("Tên chỉ số không được để trống.");
            // More specific validation can be added here (e.g., min < max)

            const payload: CreateThresholdsDto = { configs: [newConfigData] };
            await createThresholdConfigs(payload);
            toast({ variant: "success", title: "Thành công", description: `Đã tạo ngưỡng cho ${newConfigData.elementName}.` });
            await handleFetchThresholds();
            // Keep modal open or close based on UX preference
            // setIsAlertModalOpen(false);
        } catch (error: any) {
            console.error("Failed to create threshold:", error);
            toast({ variant: "destructive", title: "Lỗi tạo ngưỡng", description: error.message || "Vui lòng thử lại." });
        } finally {
            setIsProcessing(false);
        }
    }, [toast, handleFetchThresholds]);

    const handleDeleteThreshold = useCallback(async (idToDelete: string, elementName: string) => {
        // Confirmation dialog
        if (!window.confirm(`Bạn có chắc chắn muốn xóa cấu hình ngưỡng cho "${elementName}" không?`)) {
            return;
        }
        setIsProcessing(true);
        // console.log(`Deleting threshold ID: ${idToDelete} (${elementName})`); // Debug
        try {
            const payload: DeleteThresholdsDto = { ids: [idToDelete], hardDelete: false }; // Default to soft delete
            await deleteThresholdConfigs(payload);
            toast({ variant: "success", title: "Thành công", description: `Đã xóa ngưỡng cho ${elementName}.` });
            await handleFetchThresholds();
            // Keep modal open or close based on UX preference
        } catch (error: any) {
            console.error("Failed to delete threshold:", error);
            toast({ variant: "destructive", title: "Lỗi xóa ngưỡng", description: error.message || "Vui lòng thử lại." });
        } finally {
            setIsProcessing(false);
        }
    }, [toast, handleFetchThresholds]);


    // --- Dropdown Options ---
    const uniqueStationsForFilter = useMemo(() => {
        return allStations.map(s => ({ id: s.id, name: s.name }))
            .sort((a, b) => a.name.localeCompare(b.name, 'vi'));
    }, [allStations]);

    const uniqueStatusesForFilter = useMemo(() => {
        return Array.from(new Set(displayData.map(item => item.status)))
            .sort((a, b) => a.localeCompare(b, 'vi'));
    }, [displayData]);

    // --- Render Helper for Sortable Header ---
    const SortableHeader = ({ sortKey, label }: { sortKey: string; label: string }) => (
        <Button
            variant="ghost"
            onClick={() => handleSort(sortKey)}
            className="px-2 py-1 h-auto -ml-2 hover:bg-gray-200 font-semibold text-gray-700" // Adjusted styling
        >
            {label}
            {sortConfig?.key === sortKey && (
                sortConfig.direction === 'ascending'
                    ? <ArrowUp className="ml-1 h-4 w-4 text-blue-600" />
                    : <ArrowDown className="ml-1 h-4 w-4 text-blue-600" />
            )}
        </Button>
    );
    console.log("Dữ liệu truyền vào Modal (thresholdConfigs):", thresholdConfigs); 
    // --- Main Render ---
    if (isLoading && displayData.length === 0) return <PageLoader message="Đang tải dữ liệu ban đầu..." />;

    return (
        <div className="space-y-6 p-4 md:p-6"> {/* Added some padding */}
            {/* Header */}
            <header className="flex flex-wrap items-center justify-between gap-4 py-4 border-b">
                <h1 className="text-2xl font-bold whitespace-nowrap">Dữ Liệu Quan Trắc</h1>
                <div className="flex flex-wrap items-center gap-2">
                    {/* Date Picker */}
                    <Popover>
                        <PopoverTrigger asChild>
                            <Button id="date" variant={"outline"} className={cn("w-[260px] justify-start text-left font-normal", !date && "text-muted-foreground")}>
                                <CalendarIcon className="mr-2 h-4 w-4" />
                                {date?.from ? (date.to ? (<>{format(date.from, "dd/MM/y")} - {format(date.to, "dd/MM/y")}</>) : format(date.from, "dd/MM/y")) : (<span>Chọn khoảng thời gian</span>)}
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-auto p-0" align="end">
                            <Calendar initialFocus mode="range" selected={date} onSelect={handleSelectDate} numberOfMonths={2} />
                        </PopoverContent>
                    </Popover>
                    {/* Action Buttons */}
                    <Button variant="outline" onClick={handleResetFilter} disabled={isProcessing}>Reset</Button>
                    <Button variant="outline" onClick={() => setIsAlertModalOpen(true)} disabled={isProcessing}>
                        <SettingsIcon className="mr-2 h-4 w-4" />
                        Cấu hình Ngưỡng
                    </Button>
                    {/* Export */}
                    <div className="relative">
                        <Button variant="outline" onClick={() => setIsExportOpen(!isExportOpen)} disabled={isProcessing || sortedData.length === 0}>
                            <DownloadIcon className="mr-2 h-4 w-4" />Xuất File<ChevronDown size={16} className={`ml-1 transition-transform ${isExportOpen ? 'rotate-180' : ''}`} />
                        </Button>
                        {isExportOpen && (
                            <div className="absolute right-0 mt-1 w-28 bg-white border rounded-md shadow-lg z-50">
                                <button onClick={handleExportXLSX} className="block w-full text-left px-3 py-1.5 text-sm hover:bg-gray-100">Excel</button>
                                <button onClick={handleExportPDF} className="block w-full text-left px-3 py-1.5 text-sm hover:bg-gray-100">PDF</button>
                            </div>
                        )}
                    </div>
                </div>
            </header>

            {/* Table Section */}
            <div className="overflow-x-auto shadow-md rounded-lg border bg-white">
                <Table className="min-w-full whitespace-nowrap text-sm">
                    <TableHeader className="bg-gray-50">
                        <TableRow>
                             {/* Station Filter Dropdown */}
                            <TableHead className={cn(
                                "sticky left-0 bg-gray-50 z-10 px-3 py-2",
                                "w-[220px] md:w-[280px]" // Responsive width
                            )}>
                                <DropdownMenu>
                                    <DropdownMenuTrigger className="flex items-center gap-1 cursor-pointer hover:text-blue-600 font-semibold w-full text-left">
                                        {selectedStationId ? stationMap.get(selectedStationId)?.name ?? 'Chọn Trạm' : 'Tất cả trạm'}
                                        <ChevronDown size={14} className="ml-auto" />
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent className="max-h-[300px] overflow-y-auto w-[--radix-dropdown-menu-trigger-width]">
                                        <DropdownMenuItem onSelect={() => setSelectedStationId(null)}>Tất cả trạm</DropdownMenuItem>
                                        {uniqueStationsForFilter.map((station) => (
                                            <DropdownMenuItem key={station.id} onSelect={() => setSelectedStationId(station.id)}>{station.name}</DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </TableHead>
                            {/* Other Headers */}
                            <TableHead className="px-3 py-2 text-center w-[80px]">Chi tiết</TableHead> {/* Fixed width for details */}
                            <TableHead className="px-3 py-2 w-[130px]"><SortableHeader sortKey="monitoringTime" label="Thời gian" /></TableHead>
                            {metricHeaders.map((header) => {
                                const unit = getDonvi(header);
                                return (
                                    <TableHead key={header} className="text-center px-2 py-2 w-[80px]"> {/* Adjust width as needed */}
                                        <div className="flex flex-col items-center">
                                            <SortableHeader sortKey={header} label={header}/>
                                            {unit && (
                                                <span className="text-xs font-normal opacity-75 mt-0.5">({unit})</span>
                                            )}
                                        </div>
                                    </TableHead>
                                );
                            })}
                            {/* Status Filter Dropdown */}
                            <TableHead className="px-3 py-1 w-[150px]">
                                 <DropdownMenu>
                                    <DropdownMenuTrigger className="flex items-center gap-1 cursor-pointer hover:text-blue-600 font-semibold w-full text-left">
                                        {selectedStatus ? selectedStatus : 'Trạng thái'}
                                        <ChevronDown size={14} className="ml-auto"/>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent className="max-h-[270px] overflow-y-auto w-[--radix-dropdown-menu-trigger-width]">
                                        <DropdownMenuItem onSelect={() => setSelectedStatus(null)}> Trạng thái</DropdownMenuItem>
                                        {uniqueStatusesForFilter.map((status) => (
                                            <DropdownMenuItem key={status} onSelect={() => setSelectedStatus(status)}>{status}</DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </TableHead>
                            <TableHead className="min-w-[200px] px-3 py-1">Khuyến cáo</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {(isLoading && paginatedData.length === 0) && (
                            <TableRow><TableCell colSpan={metricHeaders.length + 5} className="text-center h-24">Đang tải dữ liệu...</TableCell></TableRow>
                        )}
                        {(!isLoading && paginatedData.length === 0) && (
                            <TableRow><TableCell colSpan={metricHeaders.length + 5} className="text-center h-24">{error ? `Lỗi: ${error}` : "Không tìm thấy dữ liệu phù hợp."}</TableCell></TableRow>
                        )}
                        {paginatedData.map((row) => {
                            // Recalculate alert status for row highlighting using the new function
                            let isRowAlerted = false;
                            metricHeaders.forEach(header => {
                                const keyLower = header.toLowerCase();
                                if(header !== 'WQI' && header !== 'AH') { // Only check metrics that have thresholds
                                    const indicator = row.features.find(f => f.name.toLowerCase() === keyLower) || row.features.find(f => f.name.toLowerCase().includes(keyLower));
                                    if (isValueOutOfRange(indicator, header)) {
                                        isRowAlerted = true;
                                    }
                                }
                            });

                            return (
                                <TableRow key={row.id} className={cn("hover:bg-gray-50", isRowAlerted && "bg-yellow-50 hover:bg-yellow-100")}>
                                    {/* Station Cell */}
                                    <TableCell className={cn(
                                        "sticky left-0 bg-white dark:bg-gray-800 z-10 font-medium px-3 py-2",
                                        "w-[220px] md:w-[280px]", // Match header width
                                        "whitespace-normal",
                                        "break-words",
                                         isRowAlerted && "bg-yellow-50 hover:bg-yellow-100" // Ensure sticky cell also gets row highlight
                                    )}>
                                        {row.stationName}
                                        <p className="text-xs text-gray-500 font-normal mt-1">{row.stationLocation || "Không có vị trí"}</p>
                                    </TableCell>
                                    {/* Details Cell */}
                                    <TableCell className="px-3 py-2 text-center">
                                        <Link href={`/stationsadmin?id=${row.stationId}`}>
                                            Xem
                                        </Link>
                                    </TableCell>
                                    {/* Time Cell */}
                                    <TableCell className="px-3 py-2">{formatMonitoringTimestamp(row.monitoringTime)}</TableCell>
                                    {/* Metric Cells */}
                                    {metricHeaders.map((header, metricIndex) => {
                                        const keyLower = header.toLowerCase();
                                        let value: string | number | null = 'N/A';
                                        let indicator: Indicator | undefined;

                                        if (header === 'WQI') {
                                            value = typeof row.wqi === 'number' ? row.wqi.toFixed(1) : 'N/A';
                                            // WQI typically doesn't have min/max thresholds to check
                                        } else {
                                            indicator = row.features.find(f => f.name.toLowerCase() === keyLower)
                                                     || row.features.find(f => f.name.toLowerCase().includes(keyLower));
                                            if (indicator) {
                                                value = indicator.value ?? indicator.textualValue ?? 'N/A';
                                                if (typeof value === 'number') {
                                                    value = value.toFixed(2); // Standard 2 decimal places
                                                }
                                            }
                                        }

                                        // Check if this specific cell is out of range
                                        const isCellOutOfRange = header !== 'WQI' && header !== 'AH' && isValueOutOfRange(indicator, header);

                                        return (
                                            <TableCell
                                                key={metricIndex}
                                                className={cn(
                                                    "text-center px-2 py-2",
                                                    // Apply highlight if cell value is out of range
                                                    isCellOutOfRange && "bg-red-100 font-bold text-red-700"
                                                )}
                                            >
                                                {value}
                                            </TableCell>
                                        );
                                    })}
                                    {/* Status Cell */}
                                    <TableCell className={cn("px-3 py-2", getStatusTextColor(row.status))}>{row.status}</TableCell>
                                    {/* Recommendation Cell */}
                                    <TableCell className="whitespace-normal px-3 py-2 text-xs">{row.recommendation}</TableCell>
                                </TableRow>
                            );
                        })}
                    </TableBody>
                </Table>
            </div>

            {/* Footer */}
            {(sortedData.length > 0) && !isLoading && (
                <footer className="mt-6 flex flex-col sm:flex-row justify-center sm:justify-between items-center gap-4">
                    {/* Items per page */}
                    <div className="flex items-center order-2 sm:order-1">
                        <label htmlFor="itemsPerPage" className="mr-2 text-sm text-gray-700">Hiển thị:</label>
                        <select id="itemsPerPage" value={itemsPerPage} onChange={(e) => { setItemsPerPage(Number(e.target.value) || 12); setCurrentPage(1); }}
                            className="border px-2 py-1 rounded-md text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="6">6</option> <option value="12">12</option> <option value="18">18</option> <option value="24">24</option> <option value="50">50</option>
                        </select>
                        <span className="ml-2 text-sm text-gray-700">dòng/trang</span>
                    </div>
                    {/* Pagination */}
                    {totalPages > 1 && (
                        <div className="order-1 sm:order-2">
                            <Pagination currentPage={currentPage} totalPages={totalPages} onPageChange={handlePageChange} />
                        </div>
                    )}
                    <div className="hidden sm:block sm:w-[180px] order-3"></div> {/* Spacer */}
                </footer>
            )}

            {/* Alert Modal (Threshold Config Modal) */}
            {/* *** QUAN TRỌNG: Component này cần được bạn TẠO MỚI hoặc CẬP NHẬT *** */}
            <AlertConfigModal
                isOpen={isAlertModalOpen}
                onClose={() => setIsAlertModalOpen(false)}
                currentThresholds={thresholdConfigs} // Pass current thresholds
                onUpdate={handleUpdateAllThresholds} // Pass update handler
                onCreate={handleCreateThreshold}     // Pass create handler
                onDelete={handleDeleteThreshold}     // Pass delete handler
                validElementNames={validThresholdElementNames} // Pass valid element names for configuration
                isProcessing={isProcessing} // Pass processing state for disabling buttons inside modal
            />
        </div>
    );
}