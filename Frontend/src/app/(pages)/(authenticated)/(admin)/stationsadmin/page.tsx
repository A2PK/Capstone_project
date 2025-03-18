"use client";

import { Pagination } from "@/components/pagination";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import "leaflet/dist/leaflet.css";
import StationDetails from "@/components/stationsDetails";
import dynamic from "next/dynamic";
import { useState } from "react";
import Chartline from "@/components/linechart";
import { Station } from "@/types/station";
import { getStatusTextColor } from "@/lib/utils";
const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(mod => mod.Popup), { ssr: false });
const MarkerClusterGroup = dynamic(() => import("react-leaflet-cluster"), { ssr: false });

const stations: Station[] = [
    {
      name: "Tân An",
      lat: 10.535,
      lng: 106.413,
      wqi: 37,
      status: "Nguy hiểm",
      recommendation: "WQI nguy hiểm, cần xử lý nguồn nước khẩn cấp",
      time: "10:20, Thg 3, 2025",
      trend: [25, 30, 34, 37, 33, 39, 42, 40, 45, 47, 44, 49, 46, 50, 48],
      prediction: [50, 53, 55, 58, 60, 62, 65, 68, 70, 72, 74, 75, 78, 80, 82],
      feature: [
        {
          name: "pH",
          trend: [6.0, 6.1, 6.3, 6.2, 6.4, 6.5, 6.3, 6.6, 6.7, 6.8, 6.5, 6.9, 7.0, 6.8, 7.1],
          prediction: [7.0, 7.1, 7.2, 7.3, 7.2, 7.4, 7.5, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.0, 8.1],
        },
        {
          name: "NH4",
          trend: [1.2, 1.5, 1.8, 2.0, 1.7, 2.2, 2.4, 2.1, 2.5, 2.7, 2.6, 2.9, 3.0, 2.8, 3.1],
          prediction: [3.0, 3.2, 3.3, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.8],
        },
        {
          name: "DO",
          trend: [4.5, 4.3, 4.0, 3.8, 4.1, 3.9, 3.7, 4.0, 4.2, 4.3, 4.1, 4.4, 4.5, 4.6, 4.7],
          prediction: [4.8, 4.9, 5.0, 5.1, 5.0, 5.2, 5.3, 5.4, 5.4, 5.5, 5.6, 5.6, 5.7, 5.8, 6.0],
        },
      ],
    },
    {
      name: "Mỹ Tho",
      lat: 10.36,
      lng: 106.365,
      wqi: 78,
      status: "Tốt",
      recommendation: "WQI tốt, cần theo dõi thường xuyên",
      time: "10:20, Thg 3, 2025",
      trend: [68, 70, 72, 74, 73, 76, 78, 77, 80, 82, 81, 84, 85, 87, 89],
      prediction: [90, 91, 92, 93, 94, 95, 96, 97, 97, 98, 99, 99, 100, 100, 100],
      feature: [
        {
          name: "pH",
          trend: [6.8, 6.9, 7.0, 7.1, 7.0, 7.2, 7.3, 7.2, 7.4, 7.5, 7.4, 7.6, 7.7, 7.8, 7.9],
          prediction: [8.0, 8.0, 8.1, 8.2, 8.2, 8.3, 8.4, 8.4, 8.5, 8.6, 8.6, 8.7, 8.8, 8.8, 8.9],
        },
        {
          name: "NH4",
          trend: [0.8, 0.9, 1.0, 1.1, 1.0, 1.2, 1.3, 1.2, 1.4, 1.5, 1.4, 1.6, 1.7, 1.8, 1.9],
          prediction: [1.9, 2.0, 2.1, 2.2, 2.2, 2.3, 2.4, 2.5, 2.6, 2.6, 2.7, 2.8, 2.9, 3.0, 3.0],
        },
        {
          name: "DO",
          trend: [6.0, 6.2, 6.3, 6.4, 6.3, 6.5, 6.6, 6.7, 6.8, 6.8, 6.9, 7.0, 7.1, 7.1, 7.2],
          prediction: [7.2, 7.3, 7.4, 7.5, 7.5, 7.6, 7.7, 7.7, 7.8, 7.9, 7.9, 8.0, 8.1, 8.2, 8.2],
        },
      ],
    },
    {
      name: "Bến Tre",
      lat: 10.241,
      lng: 106.375,
      wqi: 92,
      status: "Rất tốt",
      recommendation: "WQI rất tốt, đảm bảo an toàn sinh hoạt",
      time: "10:19, Thg 3, 2025",
      trend: [85, 87, 88, 89, 90, 91, 92, 91, 93, 94, 95, 96, 97, 98, 99],
      prediction: [99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
      feature: [
        {
          name: "pH",
          trend: [7.2, 7.3, 7.4, 7.4, 7.5, 7.6, 7.7, 7.6, 7.8, 7.9, 7.9, 8.0, 8.0, 8.1, 8.2],
          prediction: [8.2, 8.3, 8.3, 8.4, 8.4, 8.5, 8.6, 8.6, 8.7, 8.8, 8.8, 8.9, 8.9, 9.0, 9.0],
        },
        {
          name: "NH4",
          trend: [0.6, 0.7, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4],
          prediction: [1.4, 1.5, 1.5, 1.6, 1.6, 1.7, 1.7, 1.8, 1.8, 1.9, 1.9, 2.0, 2.0, 2.1, 2.1],
        },
        {
          name: "DO",
          trend: [7.0, 7.1, 7.2, 7.2, 7.3, 7.4, 7.5, 7.4, 7.6, 7.7, 7.7, 7.8, 7.9, 8.0, 8.0],
          prediction: [8.0, 8.1, 8.2, 8.2, 8.3, 8.4, 8.4, 8.5, 8.5, 8.6, 8.7, 8.7, 8.8, 8.9, 8.9],
        },
      ],
    },
    {
        name: "Vĩnh Long",
        lat: 10.253,
        lng: 105.972,
        wqi: 75,
        status: "Tốt",
        recommendation: "WQI tốt, có thể sử dụng nước sinh hoạt",
        time: "10:19, Thg 3, 2025",
        trend: [65, 68, 70, 72, 74, 73, 75, 76, 78, 80, 82, 81, 84, 85, 87],
        prediction: [87, 88, 89, 90, 91, 92, 93, 94, 95, 95, 96, 97, 98, 99, 100],
        feature: [
          {
            name: "pH",
            trend: [6.9, 7.0, 7.1, 7.1, 7.2, 7.3, 7.4, 7.4, 7.5, 7.6, 7.7, 7.7, 7.8, 7.9, 8.0],
            prediction: [8.0, 8.1, 8.2, 8.3, 8.3, 8.4, 8.5, 8.6, 8.6, 8.7, 8.8, 8.8, 8.9, 9.0, 9.0],
          },
          {
            name: "NH4",
            trend: [1.0, 1.1, 1.2, 1.2, 1.3, 1.4, 1.4, 1.5, 1.6, 1.6, 1.7, 1.8, 1.8, 1.9, 2.0],
            prediction: [2.0, 2.1, 2.2, 2.3, 2.3, 2.4, 2.5, 2.5, 2.6, 2.7, 2.8, 2.8, 2.9, 3.0, 3.0],
          },
          {
            name: "DO",
            trend: [5.5, 5.6, 5.7, 5.7, 5.8, 5.9, 6.0, 6.0, 6.1, 6.2, 6.3, 6.3, 6.4, 6.5, 6.5],
            prediction: [6.6, 6.7, 6.7, 6.8, 6.9, 6.9, 7.0, 7.1, 7.1, 7.2, 7.3, 7.3, 7.4, 7.5, 7.5],
          },
        ],
      },
      {
        name: "Cần Thơ",
        lat: 10.033,
        lng: 105.783,
        wqi: 88,
        status: "Rất tốt",
        recommendation: "WQI rất tốt, đảm bảo an toàn sử dụng",
        time: "10:18, Thg 3, 2025",
        trend: [80, 82, 83, 85, 86, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98],
        prediction: [98, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        feature: [
          {
            name: "pH",
            trend: [7.0, 7.1, 7.2, 7.2, 7.3, 7.4, 7.5, 7.5, 7.6, 7.7, 7.7, 7.8, 7.9, 8.0, 8.0],
            prediction: [8.0, 8.1, 8.2, 8.3, 8.3, 8.4, 8.5, 8.5, 8.6, 8.7, 8.7, 8.8, 8.9, 9.0, 9.0],
          },
          {
            name: "NH4",
            trend: [0.9, 1.0, 1.1, 1.1, 1.2, 1.3, 1.3, 1.4, 1.5, 1.5, 1.6, 1.7, 1.7, 1.8, 1.9],
            prediction: [1.9, 2.0, 2.1, 2.2, 2.2, 2.3, 2.4, 2.4, 2.5, 2.6, 2.6, 2.7, 2.8, 2.9, 3.0],
          },
          {
            name: "DO",
            trend: [6.0, 6.1, 6.2, 6.2, 6.3, 6.4, 6.5, 6.5, 6.6, 6.7, 6.7, 6.8, 6.9, 7.0, 7.0],
            prediction: [7.0, 7.1, 7.2, 7.3, 7.3, 7.4, 7.5, 7.5, 7.6, 7.7, 7.7, 7.8, 7.9, 8.0, 8.0],
          },
        ],
      },
      {
        name: "Trà Vinh",
        lat: 9.934,
        lng: 106.342,
        wqi: 62,
        status: "Trung bình",
        recommendation: "WQI trung bình, cần giám sát thêm",
        time: "10:18, Thg 3, 2025",
        trend: [50, 52, 55, 57, 60, 58, 62, 63, 65, 67, 68, 70, 72, 73, 75],
        prediction: [75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
        feature: [
          {
            name: "pH",
            trend: [6.2, 6.3, 6.4, 6.4, 6.5, 6.6, 6.7, 6.7, 6.8, 6.9, 6.9, 7.0, 7.1, 7.2, 7.3],
            prediction: [7.3, 7.4, 7.5, 7.6, 7.6, 7.7, 7.8, 7.8, 7.9, 8.0, 8.0, 8.1, 8.2, 8.2, 8.3],
          },
          {
            name: "NH4",
            trend: [1.5, 1.6, 1.7, 1.7, 1.8, 1.9, 1.9, 2.0, 2.1, 2.1, 2.2, 2.3, 2.3, 2.4, 2.5],
            prediction: [2.5, 2.6, 2.7, 2.8, 2.8, 2.9, 3.0, 3.0, 3.1, 3.2, 3.3, 3.3, 3.4, 3.5, 3.5],
          },
          {
            name: "DO",
            trend: [4.8, 4.9, 5.0, 5.0, 5.1, 5.2, 5.3, 5.3, 5.4, 5.5, 5.5, 5.6, 5.7, 5.8, 5.8],
            prediction: [5.8, 5.9, 6.0, 6.1, 6.1, 6.2, 6.3, 6.3, 6.4, 6.5, 6.5, 6.6, 6.7, 6.8, 6.8],
          },
        ],
      },
      {
        name: "Long Xuyên",
        lat: 10.371,
        lng: 105.432,
        wqi: 45,
        status: "Kém",
        recommendation: "WQI thấp, cần xử lý trước khi sử dụng",
        time: "10:18, Thg 3, 2025",
        trend: [40, 42, 43, 44, 45, 46, 47, 47, 48, 49, 50, 51, 51, 52, 53],
        prediction: [53, 54, 55, 55, 56, 57, 58, 58, 59, 60, 61, 62, 63, 64, 65],
        feature: [
          {
            name: "pH",
            trend: [6.0, 6.1, 6.2, 6.2, 6.3, 6.4, 6.5, 6.5, 6.6, 6.7, 6.7, 6.8, 6.9, 6.9, 7.0],
            prediction: [7.0, 7.1, 7.2, 7.2, 7.3, 7.4, 7.5, 7.5, 7.6, 7.7, 7.7, 7.8, 7.9, 7.9, 8.0],
          },
          {
            name: "NH4",
            trend: [2.0, 2.1, 2.2, 2.2, 2.3, 2.4, 2.5, 2.5, 2.6, 2.7, 2.8, 2.8, 2.9, 3.0, 3.0],
            prediction: [3.1, 3.2, 3.3, 3.3, 3.4, 3.5, 3.6, 3.6, 3.7, 3.8, 3.9, 3.9, 4.0, 4.1, 4.2],
          },
          {
            name: "DO",
            trend: [4.5, 4.6, 4.7, 4.7, 4.8, 4.9, 5.0, 5.0, 5.1, 5.2, 5.2, 5.3, 5.4, 5.5, 5.5],
            prediction: [5.5, 5.6, 5.7, 5.7, 5.8, 5.9, 6.0, 6.0, 6.1, 6.2, 6.2, 6.3, 6.4, 6.5, 6.5],
          },
        ],
      },
  ];


let redIcon: any, blueIcon: any, createClusterCustomIcon: any;

if (typeof window !== "undefined") {
  const { Icon, divIcon, point } = require("leaflet");

  redIcon = new Icon({
    iconUrl: "/red_one.png",
    iconSize: [38, 38],
  });

  blueIcon = new Icon({
    iconUrl: "/blue_one.png",
    iconSize: [38, 38],
  });

  createClusterCustomIcon = (cluster: any) => {
    return divIcon({
      html: `<span style="background-color: #333; height: 2em; width: 2em; color: #fff; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 1.2rem; box-shadow: 0 0 0px 5px #fff;">${cluster.getChildCount()}</span>`,
      className: "custom-marker-cluster",
      iconSize: point(33, 33, true),
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
  const defaultFeature =
    selectedStation.trend && selectedStation.prediction
      ? "WQI"
      : selectedStation.feature.length > 0
      ? selectedStation.feature[0].name
      : null;

  const [selectedFeature, setSelectedFeature] = useState<string | null>(
    defaultFeature
  );

  const handleFeatureChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedFeature(event.target.value);
  };
  const selectedData =
    selectedFeature === "WQI"
      ? selectedStation.trend && selectedStation.prediction
        ? { trend: selectedStation.trend, prediction: selectedStation.prediction }
        : null
      : selectedStation.feature.find((f) => f.name === selectedFeature);


  return (
    <div className="flex flex-1 overflow-hidden"> {/* Sidebar-aware container */}
      {/* Main Content */}
      <div className="flex flex-col flex-grow overflow-hidden space-y-4 p-4">
        {/* Header */}
        <header className="flex justify-between items-center border-b pb-2">
          <h1 className="text-2xl font-bold">Trạm Quan Trắc</h1>

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
                      icon={selectedStation.name === station.name ? blueIcon : redIcon}
                      eventHandlers={{
                        click: () => setSelectedStation(station),
                        mouseover: (e) => e.target.openPopup(), // Mở popup khi hover
                        mouseout: (e) => e.target.closePopup(), // Đóng popup khi rời chuột
                      }}
                    >
                      <Popup>
                        <div className="p-2 rounded-lg bg-white max-w-xs min-w-[200px] border">
                          <h3 className="text-lg font-bold">{station.name}</h3>
                          <p>WQI: <span className="font-bold text-blue-500">{station.wqi}</span></p>
                          <p className={`${station.status === "Nguy hiểm" ? "text-red-500"
                            : station.status === "Tốt" ? "text-green-800"
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
          <div className="w-full md:w-1/3 max-w-sm overflow-auto p-2 min-w-[250px]">
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
                  .map((station, index) => {
                    const isSelected = selectedStation.name === station.name;
                    return (
                      <TableRow
                        key={index}
                        onClick={() => setSelectedStation(station)}
                        className={`cursor-pointer transition-colors 
                ${isSelected ? "bg-blue-200" : "hover:bg-gray-100"}`}
                        style={isSelected ? { pointerEvents: "none" } : {}}
                      >
                        <TableCell className="font-medium">{station.name}</TableCell>
                        <TableCell>{station.wqi}</TableCell>
                        <TableCell
                          className={getStatusTextColor(station.status)}
                        >
                          {station.status}
                        </TableCell>
                      </TableRow>
                    );
                  })}
              </TableBody>
            </Table>

            {/* Thêm margin-top để phân trang không bị sát với bảng */}
            <div className="mt-3">
              <Pagination
                currentPage={currentPage}
                totalPages={Math.ceil(stations.length / itemsPerPage)}
                onPageChange={handlePageChange}
              />
            </div>
          </div>

        </div>

        {/* Detail Section */}
        <div className="flex flex-col items-center">
          {selectedStation && <StationDetails selectedStation={selectedStation} />}
        </div>
        {/* <Chartline trend={selectedStation.trend} prediction={selectedStation.prediction} title="chất lượng nước WQI"/>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {selectedStation.feature.map((feature, idx) => (
            <div key={idx} className="w-full">
              {/* <h2 className="text-lg font-semibold text-center">{feature.name}</h2> */}
              {/* <Chartline trend={feature.trend} prediction={feature.prediction} title={feature.name}/>
            </div>
          ))}
        </div> */} 
        <div>
      {selectedStation.feature.length > 0 || selectedStation.trend ? (
        <>
          {/* Feature Selection Dropdown */}
          <div className="flex justify-center mb-4">
            <select
              value={selectedFeature || ""}
              onChange={handleFeatureChange}
              className="border p-2 rounded-md"
            >
              {selectedStation.trend && selectedStation.prediction && (
                <option value="WQI">Chất lượng nước WQI</option>
              )}
              {selectedStation.feature.map((feature, idx) => (
                <option key={idx} value={feature.name}>
                  Chỉ số {feature.name}
                </option>
              ))}
            </select>
          </div>

          {/* Display Selected Chart */}
          {selectedData ? (
            <Chartline
              trend={selectedData.trend}
              prediction={selectedData.prediction}
              title={selectedFeature!}
            />
          ) : null}
        </>
      ) : (
        <div className="flex justify-center items-center h-40 bg-gray-200 text-red-600 font-semibold rounded-lg">
          There is no data for this station
        </div>
      )}
    </div>
      </div>
    </div>
  );
}
