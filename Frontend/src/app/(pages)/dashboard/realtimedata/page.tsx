"use client"
import { Pagination } from "@/components/pagination";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Calendar, Search } from "lucide-react";
import { useEffect, useState } from "react";

const monitoringStations = [
  {
    station: "Phú Giềng",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,40],
    status: "Nguy hiểm",
    recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp",
  },
  {
    station: "Châu Phú",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,80],
    status: "Tốt",
    recommendation: "WQI tốt, chú ý lọc nước",
  },
  {
    station: "Thoại Sơn",
    time: "12:31, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,100],
    status: "Rất tốt",
    recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm",
  },
  {
    station: "Phú Giềng",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,40],
    status: "Nguy hiểm",
    recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp",
  },
  {
    station: "Châu Phú",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,80],
    status: "Tốt",
    recommendation: "WQI tốt, chú ý lọc nước",
  },
  {
    station: "Thoại Sơn",
    time: "12:31, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,100],
    status: "Rất tốt",
    recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm",
  },{
    station: "Phú Giềng",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,40],
    status: "Nguy hiểm",
    recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp",
  },
  {
    station: "Châu Phú",
    time: "12:32, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,80],
    status: "Tốt",
    recommendation: "WQI tốt, chú ý lọc nước",
  },
  {
    station: "Thoại Sơn",
    time: "12:31, Sep 9, 2024",
    metrics: [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,100],
    status: "Rất tốt",
    recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm",
  },
];
let ITEMS_PER_PAGE = 6; // Number of items per page

export default function Realtimedata() {
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [ITEMS_PER_PAGE, setItemAmount] = useState<number>(6);
  const [currentDate, setCurrentDate] = useState<string>("");

  useEffect(() => {
    const today = new Date();
    const formattedDate = today.toLocaleDateString("vi-VN", {
      day: "numeric",
      month: "long",
      year: "numeric",
    });
    setCurrentDate(formattedDate);
  }, []);

  const totalPages = Math.ceil(monitoringStations.length / ITEMS_PER_PAGE);
  const displayedStations = monitoringStations.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between py-4 border-b">
        <h1 className="text-2xl font-bold">Tất cả</h1>
        <div className="flex items-center space-x-4">
          <div className="flex items-center border px-3 py-2 rounded-md">
            <Search className="w-5 h-5 text-gray-500 mr-2" />
            <input
              type="text"
              placeholder="Tìm kiếm"
              className="outline-none text-sm placeholder-gray-400"
            />
          </div>
          <div className="flex items-center border px-3 py-2 rounded-md">
            <Calendar className="w-5 h-5 text-gray-500 mr-2" />
            <span className="text-sm text-gray-600">{currentDate}</span>
          </div>
        </div>
      </header>

      <Table className="w-full">
        <TableHeader>
          <TableRow>
            <TableHead>Trạm Quan Trắc</TableHead>
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
            <TableHead>WQI</TableHead>
            <TableHead>Trạng thái</TableHead>
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
              {station.metrics.map((metric, idx) => (
                <TableCell key={idx}>{metric}</TableCell>
              ))}
              <TableCell
                className={
                  station.status === "Nguy hiểm"
                    ? "text-red-600 font-bold"
                    : station.status === "Rất tốt"
                    ? "text-green-600 font-bold"
                    : "text-yellow-600 font-bold"
                }
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
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
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
  );
}
