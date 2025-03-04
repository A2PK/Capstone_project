"use client"
import { Pagination } from "@/components/pagination"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useState } from "react"
import { addDays, format } from "date-fns"
import type { DateRange, SelectRangeEventHandler } from "react-day-picker"
import { CalendarIcon } from "lucide-react"
import { Calendar } from "@/components/ui/calendar"
import { ChevronDown } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"
import { parse } from "date-fns"
import { getStatusTextColor } from "@/lib/utils";
const monitoringStations = [
  {
    station: "Phú Giềng",
    time: "12:32, Sep 9, 2023",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 40],
    status: "Nguy hiểm",
    recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp",
  },
  {
    station: "Châu Phú",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 80],
    status: "Tốt",
    recommendation: "WQI tốt, chú ý lọc nước",
  },
  {
    station: "Thoại Sơn",
    time: "12:31, Sep 9, 2025",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 100],
    status: "Rất tốt",
    recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm",
  },
  {
    station: "Phú Giềng",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 40],
    status: "Nguy hiểm",
    recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp",
  },
  {
    station: "Châu Phú",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 80],
    status: "Tốt",
    recommendation: "WQI tốt, chú ý lọc nước",
  },
  {
    station: "Thoại Sơn",
    time: "12:31, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 100],
    status: "Rất tốt",
    recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm",
  },
  {
    station: "Phú Giềng",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 40],
    status: "Nguy hiểm",
    recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp",
  },
  {
    station: "Châu Phú",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 80],
    status: "Tốt",
    recommendation: "WQI tốt, chú ý lọc nước",
  },
  {
    station: "Thoại Sơn",
    time: "12:31, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 100],
    status: "Rất tốt",
    recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm",
  },
]
const ITEMS_PER_PAGE = 6 // Number of items per page

export default function Realtimedata() {
  const [currentPage, setCurrentPage] = useState<number>(1)
  const [ITEMS_PER_PAGE, setItemAmount] = useState<number>(6)
  const [date, setDate] = useState<DateRange | undefined>({
    from: new Date(),
    to: addDays(new Date(), 7),
  })
  const [selectedStation, setSelectedStation] = useState<string | null>(null)
  const [selectedStatus, setSelectedStatus] = useState<string | null>(null)
  const [selectedWQI, setSelectedWQI] = useState<number | null>(null)
  const { toast } = useToast()
  const [filteredData, setFilteredData] = useState(monitoringStations);

  const uniqueStations = Array.from(new Set(monitoringStations.map((s) => s.station)))
  const uniqueStatuses = Array.from(new Set(monitoringStations.map((s) => s.status)))
  const uniqueWQI = Array.from(new Set(monitoringStations.map((s) => s.metrics[9])))

  const filteredStations = monitoringStations.filter((station) => {
    return (
      (selectedStation ? station.station === selectedStation : true) &&
      (selectedStatus ? station.status === selectedStatus : true) &&
      (selectedWQI ? station.metrics[9] === selectedWQI : true)
    )
  })
  const commonFilteredData = filteredStations.filter(station =>
    filteredData.some(filtered => (filtered.time === station.time)&&(filtered.station === station.station))
  );
  const totalPages = Math.ceil(commonFilteredData.length / ITEMS_PER_PAGE);
  const displayedStations = commonFilteredData.slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE);

  const handlePageChange = (page: number) => {
    setCurrentPage(page)
  }
  const handleSelect: SelectRangeEventHandler = (range) => {
    setDate(range)
  }
  
  const handleFilter = () => {
    if (!date?.from || !date?.to) {
      toast({ title: "Vui lòng chọn khoảng thời gian" });
      return;
    }
  
    const fromDate = new Date(date.from);
    const toDate = new Date(date.to);
  
    const newFilteredData = monitoringStations.filter((station) => {
      const stationDate = parse(station.time, "HH:mm, MMM d, yyyy", new Date());
  
      return (
        !isNaN(stationDate.getTime()) &&
        stationDate >= fromDate &&
        stationDate <= toDate &&
        (selectedStation ? station.station === selectedStation : true) &&
        (selectedStatus ? station.status === selectedStatus : true) &&
        (selectedWQI ? station.metrics[9] === selectedWQI : true)
      );
    });
  
    setFilteredData(newFilteredData);
    setCurrentPage(1);
  };
const handleResetFilter = () => {
  setFilteredData(monitoringStations); // Reset về toàn bộ dữ liệu
  setDate(undefined); 
  setSelectedStation(null);
  setSelectedStatus(null);
  setSelectedWQI(null);
  setCurrentPage(1); 
};
  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between py-4 border-b">
        <h1 className="text-2xl font-bold">Tất cả</h1>
        <div className="flex items-center space-x-4">
          <Popover>
            <PopoverTrigger asChild>
              <Button
                id="date"
                variant={"outline"}
                className={cn("w-[300px] justify-start text-left font-normal", !date && "text-muted-foreground")}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {date?.from ? (
                  date.to ? (
                    <>
                      {format(date.from, "LLL dd, y")} - {format(date.to, "LLL dd, y")}
                    </>
                  ) : (
                    format(date.from, "LLL dd, y")
                  )
                ) : (
                  <span>Chọn khoảng thời gian</span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar initialFocus mode="range" selected={date} onSelect={handleSelect} numberOfMonths={2} />
            </PopoverContent>
          </Popover>
          <Button onClick={handleFilter} className="bg-blue-600 hover:bg-blue-700 text-white">Lọc</Button>
          <Button variant="outline" onClick={handleResetFilter} className="bg-red-300 text-red-600 hover:bg-red-200 border border-red-400">Hủy lọc</Button>
        </div>
      </header>
      <Table className="w-full">
        <TableHeader>
          <TableRow>
            <TableHead>
              <DropdownMenu>
                <DropdownMenuTrigger className="flex items-center space-x-1">
                  <span>Trạm Quan Trắc</span> <ChevronDown size={14} />
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
            <TableHead>Ngày gần nhất</TableHead>
            <TableHead>PH</TableHead>
            <TableHead>EC</TableHead>
            <TableHead>DO</TableHead>
            <TableHead>NH4</TableHead>
            <TableHead>NO2</TableHead>
            <TableHead>PO4</TableHead>
            <TableHead>TSS</TableHead>
            <TableHead>COD</TableHead>
            <TableHead>VS</TableHead>
            <TableHead>
              <DropdownMenu>
                <DropdownMenuTrigger className="flex items-center space-x-1">
                  <span>WQI</span> <ChevronDown size={14} />
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onClick={() => setSelectedWQI(null)}>Tất cả</DropdownMenuItem>
                  {uniqueWQI.map((wqi) => (
                    <DropdownMenuItem key={wqi} onClick={() => setSelectedWQI(wqi)}>
                      {wqi}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </TableHead>
            <TableHead>
              <DropdownMenu>
                <DropdownMenuTrigger className="flex items-center space-x-1">
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
          {displayedStations.map((station, index) => (
            <TableRow key={index}>
              <TableCell>{station.station}</TableCell>
              <TableCell>
                <Button className="px-4 py-1 text-white bg-blue-700 hover:bg-blue-600 border border-black-600">
                  Xem
                </Button>
              </TableCell>
              <TableCell>{station.time}</TableCell>
              {station.metrics.map((value, i) => (
                <TableCell key={i}>{value}</TableCell>
              ))}
              <TableCell
                className={getStatusTextColor(station.status)}
              >
                {station.status}
              </TableCell>
              <TableCell>{station.recommendation}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <footer className="mt-6 flex justify-center items-center">
        {totalPages >= 1 && (
          <Pagination currentPage={currentPage} totalPages={totalPages} onPageChange={handlePageChange} />
        )}
        <div className="flex items-center ml-4">
          <label className="mr-2">Items per page:</label>
          <select
            value={ITEMS_PER_PAGE}
            onChange={(e) => setItemAmount(Number(e.target.value) || 1)}
            className="border px-2 py-1 rounded-md w-20 text-center"
          >
            <option value="3">3</option>
            <option value="6">6</option>
            <option value="9">9</option>
            <option value="12">12</option>
          </select>
        </div>
      </footer>
    </div>
  )
}

