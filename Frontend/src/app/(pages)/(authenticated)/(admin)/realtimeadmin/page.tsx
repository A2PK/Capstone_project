"use client";
import { Pagination } from "@/components/pagination";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useEffect, useState } from "react";
import { format } from "date-fns";
import type { DateRange, SelectRangeEventHandler } from "react-day-picker";
import { CalendarIcon, BellIcon, SettingsIcon } from "lucide-react"; // Thêm BellIcon hoặc SettingsIcon
import { Calendar } from "@/components/ui/calendar";
import { ChevronDown } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { parse } from "date-fns";
import { getStatusTextColor } from "@/lib/utils";
import { DownloadIcon } from "lucide-react";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import * as XLSX from "xlsx";
import "../../../../fonts/times"; // Đảm bảo font được import đúng cách
import Link from "next/link";
import PageLoader from "@/components/pageloader";
import { AlertConfigModal, type AlertConfiguration, type MetricAlertConfig, alertableMetrics } from '@/components/alert-modal'; // Import modal và types

// Dữ liệu mẫu (Giữ nguyên hoặc thay bằng API call)
const monitoringStations = [
    { station: "Phú Giềng", time: "12:32, Sep 9, 2023", metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 40], status: "Nguy hiểm", recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp" },
    { station: "Châu Phú", time: "12:32, Sep 9, 2024", metrics: [8.8, 500, 2.1, 1.5, 0.5, 0.8, 120, 45, 3.5, 80], status: "Tốt", recommendation: "WQI tốt, chú ý lọc nước" }, // Ví dụ giá trị khác biệt
    { station: "Thoại Sơn", time: "12:31, Sep 9, 2025", metrics: [7.2, 350, 6.5, 0.2, 0.01, 0.1, 30, 15, 3.5, 100], status: "Rất tốt", recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm" },
    { station: "Phú Giềng", time: "12:32, Sep 9, 2024", metrics: [6.1, 1500, 4.0, 2.5, 0.8, 1.2, 80, 60, 3.5, 40], status: "Nguy hiểm", recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp" }, // Ví dụ giá trị khác biệt
    { station: "Châu Phú", time: "12:32, Sep 9, 2024", metrics: [7.5, 400, 5.5, 0.4, 0.05, 0.2, 40, 20, 3.5, 80], status: "Tốt", recommendation: "WQI tốt, chú ý lọc nước" },
    { station: "Thoại Sơn", time: "12:31, Sep 9, 2024", metrics: [7.8, 300, 7.0, 0.1, 0.005, 0.05, 20, 10, 3.5, 100], status: "Rất tốt", recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm" },
    { station: "Phú Giềng", time: "12:32, Sep 9, 2024", metrics: [5.5, 800, 3.2, 3.0, 1.0, 1.5, 150, 75, 3.5, 40], status: "Nguy hiểm", recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp" }, // Ví dụ giá trị khác biệt
    { station: "Châu Phú", time: "12:32, Sep 9, 2024", metrics: [7.1, 450, 5.8, 0.3, 0.03, 0.15, 55, 25, 3.5, 80], status: "Tốt", recommendation: "WQI tốt, chú ý lọc nước" },
    { station: "Thoại Sơn", time: "12:31, Sep 9, 2024", metrics: [8.1, 250, 7.5, 0.05, 0.001, 0.02, 15, 8, 3.5, 100], status: "Rất tốt", recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm" },
];

// Tên các cột header tương ứng với index trong mảng metrics
const metricHeaders = ["pH", "EC", "DO", "NH4", "NO2", "PO4", "TSS", "COD", "VS", "WQI"];

// Giá trị mặc định cho cấu hình cảnh báo
const initialAlertConfig = alertableMetrics.reduce<AlertConfiguration>((acc, key) => {
  // Gán giá trị cho từng key cụ thể trong accumulator đã được định kiểu
  acc[key] = { min: null, max: null };
  return acc;
}, {} as AlertConfiguration); // Khởi tạo accumulator là một đối tượng rỗng được ép kiểu thành AlertConfiguration


export default function Realtimedata() {
    const [isLoading, setIsLoading] = useState(true);
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [itemsPerPage, setItemsPerPage] = useState<number>(6); // Đổi tên biến để rõ ràng hơn
    const [isExportOpen, setIsExportOpen] = useState(false); // State riêng cho dropdown xuất file
    const [date, setDate] = useState<DateRange | undefined>({ from: undefined, to: undefined });
    const [selectedStation, setSelectedStation] = useState<string | null>(null);
    const [selectedStatus, setSelectedStatus] = useState<string | null>(null);
    const [selectedWQI, setSelectedWQI] = useState<number | null>(null);
    const { toast } = useToast();
    const [filteredData, setFilteredData] = useState(monitoringStations);

    // === State cho cấu hình cảnh báo ===
    const [isAlertModalOpen, setIsAlertModalOpen] = useState(false);
    const [alertConfig, setAlertConfig] = useState<AlertConfiguration>(initialAlertConfig);
    // ===================================

    useEffect(() => {
        const timeout = setTimeout(() => {
            setIsLoading(false);
        }, 1000);
        return () => clearTimeout(timeout);
    }, []);

    const uniqueStations = Array.from(new Set(monitoringStations.map((s) => s.station)));
    const uniqueStatuses = Array.from(new Set(monitoringStations.map((s) => s.status)));
    const uniqueWQI = Array.from(new Set(monitoringStations.map((s) => s.metrics[9]))); // WQI ở index 9

    // Hàm kiểm tra giá trị có nằm ngoài khoảng cảnh báo không
    const isValueOutOfRange = (value: number | any, metricKey: keyof AlertConfiguration): boolean => {
        if (typeof value !== 'number' || !alertConfig[metricKey]) {
            return false; // Không phải số hoặc không có cấu hình thì bỏ qua
        }
        const config = alertConfig[metricKey];
        const { min, max } = config;

        if (min !== null && value < min) {
            return true; // Nhỏ hơn min
        }
        if (max !== null && value > max) {
            return true; // Lớn hơn max
        }
        return false;
    };

    // Lọc dữ liệu dựa trên các lựa chọn filter (station, status, WQI, date)
    const applyFilters = () => {
        let dataToFilter = monitoringStations;

        // Lọc theo ngày tháng
        if (date?.from && date?.to) {
             const fromDate = new Date(date.from);
             // Đặt giờ của toDate về cuối ngày để bao gồm cả ngày đó
             const toDate = new Date(date.to);
             toDate.setHours(23, 59, 59, 999);

            dataToFilter = dataToFilter.filter((station) => {
                try {
                    const stationDate = parse(station.time, "HH:mm, MMM d, yyyy", new Date());
                    return !isNaN(stationDate.getTime()) && stationDate >= fromDate && stationDate <= toDate;
                } catch (error) {
                    console.error("Error parsing date:", station.time, error);
                    return false; // Bỏ qua nếu không parse được ngày
                }
            });
        } else if (date?.from && !date.to) {
             // Nếu chỉ chọn ngày bắt đầu, lọc từ ngày đó trở đi (hoặc xử lý khác tùy yêu cầu)
              const fromDate = new Date(date.from);
               dataToFilter = dataToFilter.filter((station) => {
                try {
                    const stationDate = parse(station.time, "HH:mm, MMM d, yyyy", new Date());
                    return !isNaN(stationDate.getTime()) && stationDate >= fromDate;
                } catch (error) {
                     console.error("Error parsing date:", station.time, error);
                    return false;
                }
            });
        }


        // Lọc theo station, status, WQI
        dataToFilter = dataToFilter.filter((station) => {
            return (
                (selectedStation ? station.station === selectedStation : true) &&
                (selectedStatus ? station.status === selectedStatus : true) &&
                (selectedWQI !== null ? station.metrics[9] === selectedWQI : true) // Index 9 là WQI
            );
        });

        setFilteredData(dataToFilter);
        setCurrentPage(1); // Reset về trang đầu sau khi lọc
    };


    // Xử lý khi nhấn nút "Lọc"
    const handleFilter = () => {
        // Bỏ kiểm tra date ở đây vì applyFilters sẽ xử lý
         // if (!date?.from || !date?.to) {
        //     toast({ title: "Vui lòng chọn khoảng thời gian" });
        //     return;
        // }
        applyFilters();
    };

    // Xử lý khi thay đổi lựa chọn filter (station, status, WQI) -> lọc ngay lập tức
    useEffect(() => {
        applyFilters();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedStation, selectedStatus, selectedWQI]); // Không thêm date vào đây để tránh lọc khi đang chọn date range

    const handleResetFilter = () => {
        setFilteredData(monitoringStations);
        setDate(undefined);
        setSelectedStation(null);
        setSelectedStatus(null);
        setSelectedWQI(null);
        setCurrentPage(1);
    };

    // Tính toán dữ liệu hiển thị cho trang hiện tại
    const totalPages = Math.ceil(filteredData.length / itemsPerPage);
    const displayedStations = filteredData.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

    const handlePageChange = (page: number) => {
        setCurrentPage(page);
    };

    const handleSelectDate: SelectRangeEventHandler = (range) => {
        setDate(range);
        // Không lọc ngay ở đây, chờ nhấn nút Lọc
    };


    // --- Các hàm export (giữ nguyên logic) ---
    const handleExportXLSX = () => {
        const data = filteredData.map((item) => ({ /* ... mapping data ... */ }));
        const ws = XLSX.utils.json_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
        XLSX.writeFile(wb, "du_lieu.xlsx");
        setIsExportOpen(false);
    };

    const handleExportPDF = () => {
        const doc = new jsPDF({ orientation: "landscape" });
        doc.setFont("timr45w", "normal"); // Đảm bảo font đã được load
        const pageWidth = doc.internal.pageSize.width;

        // ... (Phần còn lại của code tạo PDF giữ nguyên) ...
        const tableBody = [
            ["Trạm", "Thời gian", ...metricHeaders, "Trạng thái", "Khuyến nghị"],
            ...filteredData.map((item) => [
                item.station,
                item.time,
                ...item.metrics, // Spread metrics array
                item.status,
                item.recommendation,
            ]),
        ];

        autoTable(doc, {
            startY: 60,
            styles: { font: "timr45w", fontSize: 9 },
            head: [tableBody[0]], // Chỉ định header riêng
            body: tableBody.slice(1), // Body là phần còn lại
            theme: "striped",
            headStyles: { // Style riêng cho header
                 fillColor: [41, 128, 185], // Blue background
                 textColor: 255, // White text
                 fontSize: 10, // Font size header
            },
            margin: { left: 10, right: 10 },
        });
         // ... (Phần còn lại của code tạo PDF giữ nguyên) ...
        doc.save("ket_qua_quan_trac_nuoc.pdf");
        setIsExportOpen(false);
    };
    // --- Kết thúc hàm export ---

    // --- Hàm xử lý lưu cấu hình cảnh báo ---
    const handleSaveAlertConfig = (newConfig: AlertConfiguration) => {
        setAlertConfig(newConfig);
        toast({ variant: "success", title: "Đã lưu cấu hình cảnh báo." });
        // Có thể lưu cấu hình này vào localStorage hoặc gửi lên server nếu cần persistent
    };
    // --- Kết thúc hàm xử lý lưu cấu hình cảnh báo ---

    if (isLoading) return <PageLoader message="Đang tải trang dữ liệu theo thời gian thực..." />;

    return (
        <div className="space-y-6 p-1 md:p-2"> {/* Thêm padding */}
            <header className="flex flex-wrap items-center justify-between gap-4 py-4 border-b"> {/* flex-wrap và gap */}
                <h1 className="text-2xl font-bold whitespace-nowrap">Dữ Liệu Quan Trắc</h1>
                <div className="flex flex-wrap items-center gap-2"> {/* flex-wrap và gap */}
                    {/* Date Range Picker */}
                    <Popover>
                        <PopoverTrigger asChild>
                            <Button id="date" variant={"outline"} className={cn("w-[260px] justify-start text-left font-normal", !date && "text-muted-foreground")}>
                                <CalendarIcon className="mr-2 h-4 w-4" />
                                {date?.from ? (date.to ? (<>{format(date.from, "dd/MM/y")} - {format(date.to, "dd/MM/y")}</>) : format(date.from, "dd/MM/y")) : (<span>Chọn khoảng thời gian</span>)}
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-auto p-0" align="start">
                            <Calendar initialFocus mode="range" selected={date} onSelect={handleSelectDate} numberOfMonths={2} />
                        </PopoverContent>
                    </Popover>
                    {/* Filter Buttons */}
                    <Button onClick={handleFilter} className="bg-blue-600 hover:bg-blue-700 text-white">Lọc theo ngày</Button>
                    <Button variant="outline" onClick={handleResetFilter} className="bg-gray-200 hover:bg-gray-300">Reset Bộ Lọc</Button>

                    {/* Alert Config Button */}
                    <Button variant="outline" onClick={() => setIsAlertModalOpen(true)}>
                        <SettingsIcon className="mr-2 h-4 w-4" /> {/* Hoặc BellIcon */}
                        Thiết lập Cảnh báo
                    </Button>

                    {/* Export Button */}
                    <div className="relative">
                        <Button variant="outline" onClick={() => setIsExportOpen(!isExportOpen)}>
                            <DownloadIcon className="mr-2 h-4 w-4" />
                            Xuất File
                        </Button>
                        {isExportOpen && (
                            <div className="absolute right-0 mt-2 bg-white border shadow-lg rounded-md w-28 z-50">
                                <button onClick={handleExportXLSX} className="w-full text-left px-4 py-2 hover:bg-gray-100">Excel</button>
                                <button onClick={handleExportPDF} className="w-full text-left px-4 py-2 hover:bg-gray-100">PDF</button>
                            </div>
                        )}
                    </div>
                </div>
            </header>

            {/* Bảng Dữ Liệu */}
            <div className="overflow-x-auto"> {/* Cho phép cuộn ngang trên màn hình nhỏ */}
                <Table className="min-w-full whitespace-nowrap"> {/* Đảm bảo bảng không bị vỡ layout */}
                    <TableHeader>
                        <TableRow>
                            <TableHead>
                                <DropdownMenu>
                                    <DropdownMenuTrigger className="flex items-center space-x-1 cursor-pointer hover:text-blue-600">
                                        <span>Trạm</span> <ChevronDown size={14} />
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent>
                                        <DropdownMenuItem onClick={() => setSelectedStation(null)}>Tất cả</DropdownMenuItem>
                                        {uniqueStations.map((station) => (
                                            <DropdownMenuItem key={station} onClick={() => setSelectedStation(station)}>
                                                {station}
                                            </DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </TableHead>
                             <TableHead>Chi tiết</TableHead>
                            <TableHead>Thời gian</TableHead>
                            {/* Render Metric Headers */}
                            {metricHeaders.map((header, index) => (
                                <TableHead key={index}>{header}</TableHead>
                            ))}
                            <TableHead>
                                <DropdownMenu>
                                    <DropdownMenuTrigger className="flex items-center space-x-1 cursor-pointer hover:text-blue-600">
                                        <span>Trạng thái</span> <ChevronDown size={14} />
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent>
                                        <DropdownMenuItem onClick={() => setSelectedStatus(null)}>Tất cả</DropdownMenuItem>
                                        {uniqueStatuses.map((status) => (
                                            <DropdownMenuItem key={status} onClick={() => setSelectedStatus(status)}>
                                                {status}
                                            </DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </TableHead>
                            <TableHead>Khuyến cáo</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {displayedStations.length > 0 ? (
                            displayedStations.map((station, rowIndex) => {
                                // Kiểm tra xem có giá trị nào trong hàng này nằm ngoài phạm vi không
                                let isRowAlerted = false;
                                const alertStatusPerRow: boolean[] = station.metrics.map((value, metricIndex) => {
                                     const metricKey = metricHeaders[metricIndex].toLowerCase() as keyof AlertConfiguration; // Chuyển header thành key: 'pH' -> 'ph'
                                     if (alertableMetrics.includes(metricKey)) { // Chỉ kiểm tra các metric có thể cấu hình
                                         const outOfRange = isValueOutOfRange(value, metricKey);
                                         if(outOfRange) isRowAlerted = true;
                                         return outOfRange;
                                     }
                                     return false; // Mặc định là không out of range nếu không cấu hình
                                });


                                return (
                                    <TableRow
                                        key={`${station.station}-${station.time}-${rowIndex}`}
                                        className={cn(
                                            isRowAlerted && "bg-yellow-100 hover:bg-yellow-200" 
                                        )}
                                    >
                                        <TableCell>{station.station}</TableCell>
                                         <TableCell>
                                          <Link href={`/stationsadmin`}> 
                                            <Button size="sm" className="px-4 py-1 text-white bg-blue-700 hover:bg-blue-600 border border-black-600">Xem</Button> 
                                           </Link>
                                         </TableCell>
                                        <TableCell>{station.time}</TableCell>
                                        {/* Render Metric Values with Alert Styling */}
                                        {station.metrics.map((value, metricIndex) => (
                                            <TableCell
                                                key={metricIndex}
                                                className={cn(
                                                    alertStatusPerRow[metricIndex] && "bg-red-200 font-bold text-red-900" // Highlight cell nếu out of range
                                                )}
                                            >
                                                {value}
                                            </TableCell>
                                        ))}
                                        <TableCell className={getStatusTextColor(station.status)}>
                                            {station.status}
                                        </TableCell>
                                        <TableCell className="min-w-[200px] whitespace-normal">{station.recommendation}</TableCell> {/* Cho phép xuống dòng */}
                                    </TableRow>
                                );
                            })
                        ) : (
                            <TableRow>
                                <TableCell colSpan={metricHeaders.length + 5} className="text-center"> {/* +5 cột cố định */}
                                    Không tìm thấy dữ liệu phù hợp.
                                </TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </div>

            {/* Footer: Pagination and Items per page */}
            <footer className="mt-6 flex flex-wrap justify-center items-center gap-4"> {/* flex-wrap và gap */}
                {totalPages > 1 && (
                    <Pagination currentPage={currentPage} totalPages={totalPages} onPageChange={handlePageChange} />
                )}
                <div className="flex items-center">
                    <label htmlFor="itemsPerPage" className="mr-2 text-sm">Hiển thị:</label>
                    <select
                        id="itemsPerPage"
                        value={itemsPerPage}
                        onChange={(e) => {
                            setItemsPerPage(Number(e.target.value) || 6);
                            setCurrentPage(1); // Reset về trang 1 khi thay đổi số lượng item
                        }}
                        className="border px-2 py-1 rounded-md text-sm"
                    >
                        <option value="6">6</option>
                        <option value="12">12</option>
                        <option value="18">18</option>
                        <option value="24">24</option>
                    </select>
                     <span className="ml-2 text-sm">dòng/trang</span>
                </div>
            </footer>

            {/* Alert Configuration Modal */}
            <AlertConfigModal
                isOpen={isAlertModalOpen}
                onClose={() => setIsAlertModalOpen(false)}
                initialConfig={alertConfig}
                onSave={handleSaveAlertConfig}
            />
        </div>
    );
}