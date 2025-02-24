"use client";

import { Pagination } from "@/components/pagination";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import "leaflet/dist/leaflet.css";
import { Calendar, Search } from "lucide-react";
import dynamic from "next/dynamic";
import { useState } from "react";
import Chartline from "@/components/linechart";
const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(mod => mod.Popup), { ssr: false });
const MarkerClusterGroup = dynamic(() => import("react-leaflet-cluster"), { ssr: false });

const stations = [
  { name: "Tân An", lat: 10.535, lng: 106.413, wqi: 35, status: "Nguy hiểm", recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp", time: "12:32, Thg 9, 2024", trend: [20, 30, 40, 50, 45, 55, 60], prediction: [60, 62, 65, 68, 70, 73, 75] },
  { name: "Mỹ Tho", lat: 10.36, lng: 106.365, wqi: 80, status: "Tốt", recommendation: "WQI tốt, chú ý lọc nước", time: "12:32, Thg 9, 2024", trend: [70, 75, 78, 80, 85, 88, 90], prediction: [90, 92, 94, 96, 98, 99, 100] },
  { name: "Bến Tre", lat: 10.241, lng: 106.375, wqi: 90, status: "Rất tốt", recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm", time: "12:31, Thg 9, 2024", trend: [85, 87, 89, 90, 92, 94, 95], prediction: [95, 96, 97, 98, 99, 100, 100] }
];

let customIcon: any;
let createClusterCustomIcon: any;

if (typeof window !== "undefined") {
  const { Icon, divIcon, point } = require("leaflet");

  customIcon = new Icon({
    iconUrl: "/placeholder.png",
    iconSize: [38, 38]
  });

  createClusterCustomIcon = (cluster: any) => {
    return divIcon({
      html: `<span style="background-color: #333; height: 2em; width: 2em; color: #fff; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 1.2rem; box-shadow: 0 0 0px 5px #fff;">${cluster.getChildCount()}</span>`,
      className: "custom-marker-cluster",
      iconSize: point(33, 33, true)
    });
  };
}



export default function StationsPage() {
  const [selectedStation, setSelectedStation] = useState(stations[0]);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 5;

  const handlePageChange = (page: any) => setCurrentPage(page);

  const getMonthLabels = () => {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const now = new Date().getMonth(); // Get current month index (0-based)
  
    return [
      months[(now - 3 + 12) % 12],
      months[(now - 2 + 12) % 12],
      months[(now - 1 + 12) % 12],
      "Tháng này",
      months[(now + 1) % 12],
      months[(now + 2) % 12],
      months[(now + 3) % 12],
    ];
  };
  const labels = getMonthLabels();

  const nowIndex = Math.floor(labels.length / 2); // Find the index of "Now" in the labels array


  return (
    <div className="flex flex-1 overflow-hidden"> {/* Sidebar-aware container */}
  {/* Main Content */}
  <div className="flex flex-col flex-grow overflow-hidden space-y-4 p-4">
    {/* Header */}
    <header className="flex justify-between items-center border-b pb-2">
      <h1 className="text-2xl font-bold">Trạm Quan Trắc</h1>
      <div className="flex space-x-2">
        <div className="flex items-center border px-3 py-1 rounded-md">
          <Search className="w-4 h-4 text-gray-500 mr-2" />
          <input type="text" placeholder="Tìm kiếm" className="outline-none text-sm" />
        </div>
        <div className="flex items-center border px-3 py-1 rounded-md">
          <Calendar className="w-4 h-4 text-gray-500 mr-2" />
          <span className="text-sm">Tháng 9 năm 2024</span>
        </div>
      </div>
    </header>

    {/* Map & Table */}
    <div className="flex flex-grow gap-4 min-h-[60vh] overflow-hidden">
      {/* Map Section */}
      <div className="flex-grow min-w-0"> {/* Fills remaining space */}
        <MapContainer
          center={[10.535, 106.413]}
          zoom={10}
          style={{ height: "100%", width: "100%", borderRadius: "1.5rem" }}
        >
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
                  icon={customIcon}
                  eventHandlers={{ click: () => setSelectedStation(station) }}
                >
                  <Popup>
                    <div className="p-2 rounded-lg bg-white max-w-xs min-w-[200px] border">
                      <h3 className="text-lg font-bold">{station.name}</h3>
                      <p>WQI: <span className="font-bold text-blue-500">{station.wqi}</span></p>
                      <p className={`${
                        station.status === "Nguy hiểm" ? "text-red-500"
                        : station.status === "Tốt" ? "text-green-500"
                        : "text-blue-500"
                      }`}>
                        Trạng thái: {station.status}
                      </p>
                      <p className="text-xs text-gray-500">{station.time}</p>
                    </div>
                  </Popup>
                </Marker>
              ))}
            </MarkerClusterGroup>
          )}
        </MapContainer>
      </div>

      {/* Table Section */}
      <div className="w-full md:w-1/3 max-w-sm overflow-auto p-2 min-w-[250px]"> {/* Responsive width */}
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Trạm</TableHead>
              <TableHead>WQI</TableHead>
              <TableHead>Trạng thái</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {stations
              .slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)
              .map((station, index) => (
                <TableRow
                  key={index}
                  onClick={() => setSelectedStation(station)}
                  className="cursor-pointer"
                >
                  <TableCell>{station.name}</TableCell>
                  <TableCell>{station.wqi}</TableCell>
                  <TableCell className={
                    station.status === "Nguy hiểm" ? "text-red-600"
                    : station.status === "Tốt" ? "text-green-600"
                    : "text-yellow-600"
                  }>
                    {station.status}
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>

        <Pagination
          currentPage={currentPage}
          totalPages={Math.ceil(stations.length / itemsPerPage)}
          onPageChange={handlePageChange}
        />
      </div>
    </div>

    {/* Detail Section */}
    <section className="bg-white p-4 rounded shadow w-full">
      <h2 className="text-xl font-semibold mb-2">Chi tiết: {selectedStation.name}</h2>
      <p>WQI: {selectedStation.wqi}</p>
      <p>Trạng thái: {selectedStation.status}</p>
      <p>Khuyến cáo: {selectedStation.recommendation}</p>
      <p>Thời gian: {selectedStation.time}</p>

      <div className="mt-4">
        
      </div>
    </section><Chartline />
  </div>
</div>
  );
}
