"use client";

import { useState, useEffect, useRef } from "react";
import { MapPin, Search, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Pagination } from "@/components/pagination";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, LineElement, PointElement, LinearScale, CategoryScale } from "chart.js";

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale);

const stations = [
  { name: "Tân An", lat: 10.535, lng: 106.413, wqi: 35, status: "Nguy hiểm", recommendation: "WQI nguy hiểm, cần lọc nước khẩn cấp", time: "12:32, Thg 9, 2024", trend: [20, 30, 40, 50, 45, 55, 60], prediction: [60, 62, 65, 68, 70, 73, 75] },
  { name: "Mỹ Tho", lat: 10.360, lng: 106.365, wqi: 80, status: "Tốt", recommendation: "WQI tốt, chú ý lọc nước", time: "12:32, Thg 9, 2024", trend: [70, 75, 78, 80, 85, 88, 90], prediction: [90, 92, 94, 96, 98, 99, 100] },
  { name: "Bến Tre", lat: 10.241, lng: 106.375, wqi: 90, status: "Rất tốt", recommendation: "WQI rất tốt, chú ý an toàn vệ sinh thực phẩm", time: "12:31, Thg 9, 2024", trend: [85, 87, 89, 90, 92, 94, 95], prediction: [95, 96, 97, 98, 99, 100, 100] }
];

export default function StationsPage() {
  const [selectedStation, setSelectedStation] = useState(stations[0]);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 5;
  const mapRef = useRef<HTMLDivElement>(null); // Ensure proper typing for mapRef
  const [L, setL] = useState<any>(null); // State to store the Leaflet library

  const handlePageChange = (page: any) => setCurrentPage(page);

  const wqiData = {
    labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
    datasets: [
      {
        label: "WQI Năm nay",
        data: selectedStation.trend,
        borderColor: "#4f46e5",
        backgroundColor: "#4f46e5",
        tension: 0.4,
      },
      {
        label: "Dự đoán WQI",
        data: selectedStation.prediction,
        borderColor: "#22c55e",
        backgroundColor: "#22c55e",
        borderDash: [5, 5],
        tension: 0.4,
      }
    ]
  };

  useEffect(() => {
    import("leaflet").then((leaflet) => {
      setL(leaflet); // Set the Leaflet library in state
    });
  }, []);

  useEffect(() => {
    if (L && mapRef.current) {
      const map = L.map(mapRef.current, {
        center: [10.535, 106.413],
        zoom: 10,
        attributionControl: false, // Disable default attribution control
      });

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);

      const customIcon = new L.Icon({
        iconUrl: "/maker_icon.png",
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        shadowUrl: "/maker_icon.png",
        shadowSize: [41, 41],
      });

      stations.forEach((station) => {
        const marker = L.marker([station.lat, station.lng], { icon: customIcon }).addTo(map);

        marker.on("click", () => {
          setSelectedStation(station);
        });
      });

      // Ensure the map resizes when the container size changes
      window.addEventListener("resize", () => map.invalidateSize());

      // Call invalidateSize once after the map is initialized
      setTimeout(() => {
        map.invalidateSize(); // Force a reflow after the map is mounted
      }, 100);
    }
  }, [L]);

  return (
    <div className="flex flex-col space-y-4 p-4">
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

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Map container with explicit height ref={mapRef} */}
        <div className="md:col-span-2 bg-gray-100 flex items-center justify-center"  style={{ height: '400px', width: '100%' }}></div>

        <div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Trạm</TableHead>
                <TableHead>WQI</TableHead>
                <TableHead>Trạng thái</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {stations.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage).map((station, index) => (
                <TableRow key={index} onClick={() => setSelectedStation(station)} className="cursor-pointer">
                  <TableCell>{station.name}</TableCell>
                  <TableCell>{station.wqi}</TableCell>
                  <TableCell className={station.status === "Nguy hiểm" ? "text-red-600" : station.status === "Tốt" ? "text-green-600" : "text-yellow-600"}>{station.status}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          <Pagination currentPage={currentPage} totalPages={Math.ceil(stations.length / itemsPerPage)} onPageChange={handlePageChange} />
        </div>
      </div>

      <section className="bg-white p-4 rounded shadow w-full mx-auto">
        <h2 className="text-xl font-semibold mb-2">Chi tiết: {selectedStation.name}</h2>
        <p>WQI: {selectedStation.wqi}</p>
        <p>Trạng thái: {selectedStation.status}</p>
        <p>Khuyến cáo: {selectedStation.recommendation}</p>
        <p>Thời gian: {selectedStation.time}</p>

        <div className="mt-4">
          <h3 className="text-lg font-bold">Xu hướng và Dự đoán WQI</h3>
          <div className="h-96 md:h-[300px]">
            <Line data={wqiData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "top" } } }} />
          </div>
        </div>
      </section>
    </div>
  );
}
