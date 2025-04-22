// components/MapUpdater.tsx
"use client"; // Cần thiết vì sử dụng hook

import { useEffect } from 'react';
import { useMap } from 'react-leaflet';
import { Station } from "@/types/station2"; // Đảm bảo đường dẫn đúng
import L from 'leaflet'; // Import L để sử dụng LatLngExpression nếu cần

interface MapUpdaterProps {
  station: Station | null;
  zoomLevel: number; // Mức zoom mong muốn khi chọn trạm
}

function MapUpdater({ station, zoomLevel }: MapUpdaterProps) {
  const map = useMap(); // Lấy instance của map

  useEffect(() => {
    if (station && typeof station.latitude === 'number' && typeof station.longitude === 'number') {
      const center: L.LatLngExpression = [station.latitude, station.longitude];
      map.flyTo(center, zoomLevel, {
          animate: true,
          duration: 1.0 // Thời gian animation (giây)
      });
    }
    // Bạn có thể thêm logic khác ở đây, ví dụ:
    // else {
    //   // Nếu không có station nào được chọn, quay về view mặc định?
    //   // map.flyTo(DEFAULT_CENTER, DEFAULT_ZOOM);
    // }
  }, [station, zoomLevel, map]); // Chạy effect khi station, zoomLevel, hoặc map thay đổi

  return null; // Component này không render gì ra DOM
}

export default MapUpdater;