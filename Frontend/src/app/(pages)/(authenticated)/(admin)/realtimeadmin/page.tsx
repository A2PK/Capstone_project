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
import { Pagination } from "@/components/pagination"; // Component Pagination đã tùy chỉnh
import { AlertConfigModal, type AlertConfiguration, type MetricAlertConfig, alertableMetrics } from '@/components/alert-modal';
import PageLoader from "@/components/pageloader";

// --- Icons ---
import { CalendarIcon, SettingsIcon, DownloadIcon, ChevronDown, ArrowUp, ArrowDown } from "lucide-react";

// --- Libs & Hooks ---
import { cn, getDonvi, getStatusTextColor } from "@/lib/utils"; // Giả định getStatusTextColor tồn tại
import { useToast } from "@/hooks/use-toast";

// --- API & Types ---
import { Station, DataPoint, Indicator, QueryOptions } from "@/types/station2"; // Types từ backend
import { getStations, getDataPointsOfStationById, getAllDataPoints, getAllDataPointsByStationID } from "@/lib/station"; // API functions

// --- Font Import ---
import "../../../../fonts/times"; // Đảm bảo font được import đúng cách

// --- Constants ---
const metricHeaders = ["pH", "EC", "DO", "N-NH4", "N-NO2", "P-PO4", "TSS", "COD", "AH", "WQI"];
const metricKeysForAlert = metricHeaders
    .map(h => h.toLowerCase() as keyof AlertConfiguration)
    .filter(key => alertableMetrics.includes(key));

const initialAlertConfig = alertableMetrics.reduce<AlertConfiguration>((acc, key) => {
    acc[key] = { min: null, max: null };
    return acc;
}, {} as AlertConfiguration);

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
}//HH:mm, 

function formatMonitoringTimestamp(timeString: string | undefined | null): string {
    if (!timeString) return "N/A";
    try {
        const date = parseISO(timeString);
        if (!isValidDate(date)) return timeString;
        return format(date, 'dd/MM/yyyy');
    } catch (e) { return timeString; }
}

// --- Types ---
// Type cho cấu hình sắp xếp
interface SortConfig {
    key: string; // Key để sort (có thể là key của DataPoint hoặc tên metricHeader)
    direction: 'ascending' | 'descending';
}

// Interface cho dữ liệu hiển thị (kết hợp Station và DataPoint)
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
    const [error, setError] = useState<string | null>(null);
    const [allStations, setAllStations] = useState<Station[]>([]);
    const [stationMap, setStationMap] = useState<Map<string, Station>>(new Map());
    const [allDataPoints, setAllDataPoints] = useState<DataPoint[]>([]);
    const [displayData, setDisplayData] = useState<DisplayRowData[]>([]);
    const [filteredDisplayData, setFilteredDisplayData] = useState<DisplayRowData[]>([]);

    // Filters
    const [date, setDate] = useState<DateRange | undefined>(undefined);
    const [selectedStationId, setSelectedStationId] = useState<string | null>(null);
    const [selectedStatus, setSelectedStatus] = useState<string | null>(null);

    // Sorting - Mặc định sort theo thời gian mới nhất
    const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'monitoringTime', direction: 'descending' });

    // Pagination
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [itemsPerPage, setItemsPerPage] = useState<number>(12);

    // UI State
    const [isExportOpen, setIsExportOpen] = useState(false);
    const [isAlertModalOpen, setIsAlertModalOpen] = useState(false);

    // Other Hooks
    const { toast } = useToast();
    const [alertConfig, setAlertConfig] = useState<AlertConfiguration>(initialAlertConfig);

    // --- Data Fetching ---
    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            setError(null);
            try {
                // Fetch Stations
                const stations = await getStations({ limit: 1000 });
                setAllStations(stations);
                const newStationMap = new Map(stations.map(s => [s.id, s]));
                setStationMap(newStationMap);

                // Fetch DataPoints (CẦN TỐI ƯU HÓA - Fetch tất cả ban đầu chỉ là giải pháp tạm)
                const response = await getAllDataPoints({
                    options: {
                        // Áp dụng filter ngay trên backend
                        filters: {
                            observation_type: 'actual' // Hoặc giá trị đúng cho "không phải predicted"
                            // Có thể thêm filter ngày mặc định nếu backend hỗ trợ
                            // monitoring_time_gte: 'YYYY-MM-DDTHH:mm:ssZ' // Ví dụ: Lấy dữ liệu 7 ngày gần nhất
                        },
                        sortBy: 'monitoring_time', // Sắp xếp sẵn trên backend
                        sortDesc: true,
                        limit: 10000 // Lấy một lượng lớn ban đầu, hoặc dùng phân trang backend
                        // offset: 0 // Nếu dùng phân trang backend
                    }
                });
                setAllDataPoints(response);

            } catch (err) {
                const errorMsg = err instanceof Error ? err.message : "Lỗi không xác định khi tải dữ liệu.";
                console.error("Failed to fetch initial data:", err);
                setError(errorMsg);
                toast({ variant: "destructive", title: "Lỗi tải dữ liệu", description: errorMsg });
            } finally {
                setIsLoading(false);
            }
        };
        fetchData();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // --- Data Combination ---
    useEffect(() => {
        if (stationMap.size === 0 || allDataPoints.length === 0) {
            setDisplayData([]);
            return;
        }
        // Map DataPoints to DisplayRowData
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
        // Đặt lại trang về 1 khi filter thay đổi, TRỪ KHI chỉ có displayData thay đổi (tránh reset khi data refresh)
        // Điều này hơi phức tạp, tạm thời luôn reset về 1 để đơn giản
        setCurrentPage(1);
    }, [displayData, date, selectedStationId, selectedStatus]);

    useEffect(() => {
        applyFilters();
    }, [applyFilters]);

    // --- Sorting Logic ---
    // --- Sorting Logic ---
    const getSortValue = useCallback((item: DisplayRowData, key: string): number | string | Date | null => {
        // Xử lý các key đặc biệt trước
        if (key === 'monitoringTime') {
            try { return parseISO(item.monitoringTime); } catch { return null; }
        }
        if (key === 'stationName') { return item.stationName; }
        if (key === 'status') { return item.status; }

        let potentialValue: any = null;
        const keyLower = key.toLowerCase();

        // Ưu tiên item.wqi nếu key là 'wqi'
        if (keyLower === 'wqi') {
            potentialValue = item.wqi;
        }

        // Nếu không tìm thấy trực tiếp hoặc key khác, kiểm tra features
        if (potentialValue === null || potentialValue === undefined) {
            const indicator = item.features.find(f => f.name.toLowerCase() === keyLower) ||
                item.features.find(f => f.name.toLowerCase().includes(keyLower));
            if (indicator) {
                potentialValue = indicator.value ?? indicator.textualValue;
            }
        }

        // Xác định các key nên là số
        const numericKeys = ["wqi", "ph", "ec", "do", "nh4", "no2", "po4", "tss", "cod"]; // Kiểm tra lại nếu AH là số

        // Nếu key thuộc nhóm số, cố gắng ép kiểu số
        if (numericKeys.includes(keyLower)) {
            if (potentialValue === null || potentialValue === undefined || potentialValue === '') {
                return null;
            }
            const numValue = parseFloat(String(potentialValue));
            return isNaN(numValue) ? null : numValue; // Trả về số hoặc null nếu không hợp lệ
        }
        // Nếu không phải key số, trả về dạng chuỗi hoặc null
        else {
            return potentialValue !== null && potentialValue !== undefined ? String(potentialValue) : null;
        }
    }, []); // <-- Empty dependency array for useCallback

    const sortedData = useMemo(() => {
        let sortableItems = [...filteredDisplayData];
        if (sortConfig) {
            sortableItems.sort((a, b) => {
                // Pass the memoized getSortValue function here
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
                    comparison = valueA.localeCompare(valueB, 'vi'); // Sort tiếng Việt
                }
                // Add cases for other types if necessary

                return sortConfig.direction === 'ascending' ? comparison : -comparison;
            });
        }
        return sortableItems;
    // Keep getSortValue in the dependency array now that it's stable
    }, [filteredDisplayData, sortConfig, getSortValue]);

    // --- Pagination Logic ---
    const totalPages = Math.ceil(sortedData.length / itemsPerPage);
    const paginatedData = sortedData.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

    // --- Handlers ---
    const handleResetFilter = () => {
        setDate(undefined);
        setSelectedStationId(null);
        setSelectedStatus(null);
        setSortConfig({ key: 'monitoringTime', direction: 'descending' }); // Reset sort về mặc định
        // applyFilters sẽ tự chạy lại
    };

    const handlePageChange = (page: number) => {
        setCurrentPage(page);
    };

    const handleSelectDate: SelectRangeEventHandler = (range) => {
        setDate(range);
        // applyFilters sẽ tự chạy lại
    };

    const handleSort = (key: string) => {
        setSortConfig(prevConfig => {
            let direction: 'ascending' | 'descending' = 'ascending';
            if (prevConfig && prevConfig.key === key && prevConfig.direction === 'ascending') {
                direction = 'descending';
            }
            // Mặc định descending cho thời gian khi click lần đầu hoặc khi đang sort cột khác
            if (key === 'monitoringTime' && (!prevConfig || prevConfig.key !== key || prevConfig.direction === 'ascending')) {
                direction = 'descending';
            }
            return { key, direction };
        });
        setCurrentPage(1); // Reset về trang 1 khi sort
    };

    // --- Alert Checking ---
    const isValueOutOfRange = useCallback((indicator: Indicator | undefined, metricKey: keyof AlertConfiguration): boolean => {
        if (!indicator || !alertConfig[metricKey]) return false;
        const config = alertConfig[metricKey];
        // Cố gắng lấy giá trị số từ value hoặc textualValue
        let value: number | undefined;
        if (typeof indicator.value === 'number') {
            value = indicator.value;
        } else if (typeof indicator.textualValue === 'string') {
            const parsed = parseFloat(indicator.textualValue);
            if (!isNaN(parsed)) value = parsed;
        }

        if (value === undefined) return false; // Không có giá trị số để so sánh

        const { min, max } = config;
        if (min !== null && value < min) return true;
        if (max !== null && value > max) return true;
        return false;
    }, [alertConfig]);
    const filteredHeaders = metricHeaders.filter(header => header.toLowerCase() !== 'wqi');
    // --- Export Functions ---
    const handleExportXLSX = () => {
        if (sortedData.length === 0) {
            toast({ variant: "warning", title: "Không có dữ liệu để xuất." }); return;
        }
        const dataForExport = sortedData.map(item => {
            const row: Record<string, any> = {
                'Place': item.stationId,
                'Trạm': item.stationName,
                'Vị trí': item.stationLocation,
                'Thời gian': formatMonitoringTimestamp(item.monitoringTime),
                'Trạng thái': item.status,
                'Khuyến nghị': item.recommendation,
                'WQI': item.wqi,
            };


            // Bây giờ lặp qua mảng đã được lọc
            filteredHeaders.forEach(header => {
                const keyLower = header.toLowerCase();
                let indicator = item.features.find(f => f.name.toLowerCase() === keyLower);
                if (!indicator) indicator = item.features.find(f => f.name.toLowerCase().includes(keyLower));
                row[header] = indicator ? (indicator.value ?? indicator.textualValue ?? 'N/A') : 'N/A';
            });
            return row;
        });
        const ws = XLSX.utils.json_to_sheet(dataForExport, { header: ["Place", "Trạm", "Vị trí", "Thời gian", ...filteredHeaders, "WQI", "Trạng thái", "Khuyến nghị"] }); // Chỉ định thứ tự header
        const cols = Object.keys(dataForExport[0]).map(key => ({ wch: key === 'Khuyến nghị' ? 50 : (key === 'Trạm' || key === 'Vị trí' ? 25 : 12) }));
        ws['!cols'] = cols;
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "DuLieuQuanTrac");
        XLSX.writeFile(wb, "du_lieu_quan_trac.csv");
        setIsExportOpen(false);
    };

    const handleExportPDF = () => {
        if (sortedData.length === 0) {
            toast({ variant: "warning", title: "Không có dữ liệu để xuất." }); return;
        }
        const doc = new jsPDF({ orientation: "landscape" });
        doc.setFont("timr45w", "normal");
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


        const tableHeaders = ["TT", "Trạm", "Thời gian", ...metricHeaders, "Trạng thái", "Khuyến nghị"];
        const tableBody = sortedData.map((item, index) => {
            const rowData: (string | number | null)[] = [
                index + 1, // Số thứ tự
                item.stationName, // + (item.stationLocation ? `\n(${item.stationLocation})` : ''), // Thêm vị trí nếu cần
                formatMonitoringTimestamp(item.monitoringTime),
            ];
            metricHeaders.forEach(header => {
                const keyLower = header.toLowerCase();
                let indicator = item.features.find(f => f.name.toLowerCase() === keyLower);
                if (!indicator) indicator = item.features.find(f => f.name.toLowerCase().includes(keyLower));
                rowData.push(indicator ? (indicator.value ?? indicator.textualValue ?? 'N/A') : 'N/A');
            });
            rowData.push(item.status, item.recommendation);
            return rowData;
        });

        autoTable(doc, {
            startY: margin + 30,
            styles: { font: "timr45w", fontSize: 8, cellPadding: 1.5, overflow: 'linebreak' }, // Kích hoạt linebreak
            head: [tableHeaders],
            body: tableBody,
            theme: "grid",
            headStyles: { fillColor: [41, 128, 185], textColor: 255, fontSize: 9, halign: 'center', fontStyle: 'bold' },
            columnStyles: {
                0: { cellWidth: 8, halign: 'center' }, // TT
                1: { cellWidth: 25 }, // Trạm
                2: { cellWidth: 18 }, // Thời gian
                [tableHeaders.length - 1]: { cellWidth: 50 }, // Khuyến nghị
                [tableHeaders.length - 2]: { cellWidth: 18 }, // Trạng thái
                // Các cột chỉ số còn lại sẽ tự động điều chỉnh
            },
            didDrawPage: function (data) {
                // Footer
                doc.setFontSize(8);
                doc.text(`Trang ${data.pageNumber}`, pageWidth - margin, pageHeight - 5, { align: 'right' });
            },
            margin: { left: margin, right: margin, bottom: 10 },
        });

        doc.save("ket_qua_quan_trac_nuoc.pdf");
        setIsExportOpen(false);
    };

    // --- Alert Config Save Handler ---
    const handleSaveAlertConfig = (newConfig: AlertConfiguration) => {
        setAlertConfig(newConfig);
        toast({ variant: "success", title: "Đã lưu cấu hình cảnh báo." });
        setIsAlertModalOpen(false);
    };

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
            className="px-2 py-1 h-auto -ml-2 hover:bg-gray-200 font-semibold text-gray-700"
        >
            {label}
            {sortConfig?.key === sortKey && (
                sortConfig.direction === 'ascending'
                    ? <ArrowUp className="ml-1 h-4 w-4 text-blue-600" />
                    : <ArrowDown className="ml-1 h-4 w-4 text-blue-600" />
            )}
            {/* Add default unsorted icon maybe? */}
            {/* {sortConfig?.key !== sortKey && <ArrowUpDown className="ml-1 h-4 w-4 opacity-30" />} */}
        </Button>
    );

    // --- Main Render ---
    if (isLoading && displayData.length === 0) return <PageLoader message="Đang tải dữ liệu ban đầu..." />; // Cập nhật thông báo

    return (
        <div className="space-y-6">
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
                    <Button variant="outline" onClick={handleResetFilter}>Reset</Button>
                    <Button variant="outline" onClick={() => setIsAlertModalOpen(true)}><SettingsIcon className="mr-2 h-4 w-4" />Cảnh báo</Button>
                    {/* Export */}
                    <div className="relative">
                        <Button variant="outline" onClick={() => setIsExportOpen(!isExportOpen)}>
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
                                "w-[280px]" // <-- THÊM CLASS WIDTH Ở ĐÂY (Ví dụ: 220px)
                            )}>
                                <DropdownMenu>
                                    <DropdownMenuTrigger className="flex items-center gap-1 cursor-pointer hover:text-blue-600 font-semibold">
                                        Trạm <ChevronDown size={14} />
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent className="max-h-[300px] overflow-y-auto">
                                        <DropdownMenuItem onSelect={() => setSelectedStationId(null)}>Tất cả trạm</DropdownMenuItem>
                                        {uniqueStationsForFilter.map((station) => (
                                            <DropdownMenuItem key={station.id} onSelect={() => setSelectedStationId(station.id)}>{station.name}</DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </TableHead>
                            {/* Other Headers */}
                            <TableHead className="px-3 py-2">Chi tiết</TableHead>
                            <TableHead className="px-3 py-2"><SortableHeader sortKey="monitoringTime" label="Thời gian" /></TableHead>
                            {metricHeaders.map((header) => {
                                // Lấy đơn vị trước
                                const unit = getDonvi(header);

                                return (
                                    <TableHead key={header} className="text-center px-3 py-2">
                                        {/* Bọc nội dung bằng div để dễ dàng sắp xếp */}
                                        <div className="flex flex-col items-center"> {/* Sắp xếp theo cột, căn giữa */}
                                            {/* Phần tên header (vẫn có thể sort) */}
                                            <SortableHeader
                                                sortKey={header}
                                                label={header} // Chỉ truyền tên header cho SortableHeader
                                            // Bạn có thể cần thêm class vào đây nếu SortableHeader không chiếm đủ rộng
                                            // className="w-full" // Ví dụ
                                            />

                                            {/* Phần đơn vị (chỉ hiển thị nếu có) */}
                                            {unit && ( // Chỉ render nếu unit không rỗng
                                                <span className="text-xs font-normal opacity-75 mt-0.5"> {/* Kiểu chữ nhỏ hơn, mờ hơn, có chút margin top */}
                                                    ({unit}) {/* Hiển thị đơn vị trong ngoặc đơn */}
                                                </span>
                                            )}
                                        </div>
                                    </TableHead>
                                );
                            })}
                            {/* Status Filter Dropdown */}
                            <TableHead className="px-3 py-1">
                                <DropdownMenu>
                                    <DropdownMenuTrigger className="flex items-center gap-1 cursor-pointer hover:text-blue-600 font-semibold">
                                        Trạng thái <ChevronDown size={14} />
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent className="max-h-[270px] overflow-y-auto">
                                        <DropdownMenuItem onSelect={() => setSelectedStatus(null)}>Tất cả trạng thái</DropdownMenuItem>
                                        {uniqueStatusesForFilter.map((status) => (
                                            <DropdownMenuItem key={status} onSelect={() => setSelectedStatus(status)}>{status}</DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </TableHead>
                            <TableHead className="min-w-[230px] px-3 py-1">Khuyến cáo</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {isLoading && paginatedData.length === 0 && ( // Show loader inside table only if loading and no data yet
                            <TableRow><TableCell colSpan={metricHeaders.length + 5} className="text-center h-24">Đang tải dữ liệu...</TableCell></TableRow>
                        )}
                        {!isLoading && paginatedData.length === 0 && ( // Show no data message if not loading and no data
                            <TableRow><TableCell colSpan={metricHeaders.length + 5} className="text-center h-24">{error ? error : "Không tìm thấy dữ liệu phù hợp."}</TableCell></TableRow>
                        )}
                        {paginatedData.map((row) => {
                            // Alert logic remains the same
                            let isRowAlerted = false;
                            const alertStatusPerCell: boolean[] = metricHeaders.map(header => {
                                const metricKey = header.toLowerCase() as keyof AlertConfiguration;
                                const indicator = row.features.find(f => f.name.toLowerCase() === metricKey) || row.features.find(f => f.name.toLowerCase().includes(metricKey));
                                const outOfRange = isValueOutOfRange(indicator, metricKey);
                                if (outOfRange) isRowAlerted = true;
                                return outOfRange;
                            });
                            return (
                                <TableRow key={row.id} className={cn("hover:bg-gray-50", isRowAlerted && "bg-yellow-50 hover:bg-yellow-100")}>
                                    <TableCell className={cn(
                                        "sticky left-0 bg-white dark:bg-gray-800 z-10 font-medium px-3 py-2",
                                        "w-[230px]", // <-- THÊM CLASS WIDTH (Giống TableHead)
                                        "whitespace-normal", // <-- CHO PHÉP XUỐNG DÒNG
                                        "break-words" // <-- TÙY CHỌN: Ngắt từ nếu cần
                                    )}>
                                        {row.stationName}
                                        <p className="text-xs text-gray-500 font-normal mt-1">{row.stationLocation}</p>
                                    </TableCell>
                                    <TableCell className="px-3 py-2">
                                        <Link href={`/stationsadmin?id=${row.stationId}`} passHref>
                                            <Button size="sm" variant="link" className="h-auto p-0 text-blue-600 hover:underline">Xem</Button>
                                        </Link>
                                    </TableCell>
                                    <TableCell className="px-3 py-2">{formatMonitoringTimestamp(row.monitoringTime)}</TableCell>
                                    {metricHeaders.map((header, metricIndex) => {
                                        const keyLower = header.toLowerCase();
                                        let value: string | number | null = 'N/A'; // Khởi tạo giá trị mặc định

                                        // ---- THÊM LOGIC KIỂM TRA HEADER Ở ĐÂY ----
                                        if (header === 'WQI') {
                                            // Nếu header là WQI, lấy giá trị trực tiếp từ row.wqi
                                            value = typeof row.wqi === 'number' ? row.wqi.toFixed(1) : 'N/A'; // Làm tròn 1 chữ số thập phân
                                        } else {
                                            // Nếu là các header khác, tìm trong features như cũ
                                            let indicator = row.features.find(f => f.name.toLowerCase() === keyLower);
                                            if (!indicator) {
                                                indicator = row.features.find(f => f.name.toLowerCase().includes(keyLower));
                                            }
                                            if (indicator) {
                                                // Ưu tiên value số, sau đó đến textualValue
                                                value = indicator.value ?? indicator.textualValue ?? 'N/A';
                                                // Có thể thêm làm tròn cho các chỉ số khác nếu cần
                                                if (typeof value === 'number') {
                                                    value = value.toFixed(2); // Ví dụ làm tròn 2 chữ số
                                                }
                                            }
                                        }
                                        // ------------------------------------------

                                        return (
                                            <TableCell
                                                key={metricIndex}
                                                className={cn(
                                                    "text-center px-2 py-2", // Căn giữa giá trị chỉ số
                                                    alertStatusPerCell[metricIndex] && "bg-red-100 font-bold text-red-800" // Highlight cell nếu out of range
                                                )}
                                            >
                                                {value} {/* Hiển thị giá trị đã xử lý */}
                                            </TableCell>
                                        );
                                    })}
                                    <TableCell className={cn("px-3 py-2", getStatusTextColor(row.status))}>{row.status}</TableCell>
                                    <TableCell className="whitespace-normal px-3 py-2">{row.recommendation}</TableCell>
                                </TableRow>
                            );
                        })}
                    </TableBody>
                </Table>
            </div>

            {/* Footer */}
            {(sortedData.length > 0) && !isLoading && ( // Chỉ hiển thị footer khi có dữ liệu và không loading
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

            {/* Alert Modal */}
            <AlertConfigModal isOpen={isAlertModalOpen} onClose={() => setIsAlertModalOpen(false)} initialConfig={alertConfig} onSave={handleSaveAlertConfig} />
        </div>
    );
}