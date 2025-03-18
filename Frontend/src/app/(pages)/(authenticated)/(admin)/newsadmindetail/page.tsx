"use client";

import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Viewer, Worker } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout";
import "@react-pdf-viewer/default-layout/lib/styles/index.css"; // Import default layout styles

const NewsPage = () => {
  const [isDeleted, setIsDeleted] = useState(false);
  const isAuthenticated = true; // Simulated authentication state

  const handleDelete = () => {
    setIsDeleted(true);
  };

  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  return (
    <div className="h-screen flex flex-col bg-white mb-4">
      {/* Breadcrumb + Delete Button */}
      <div className="flex justify-between items-center px-4 py-2 bg-gray-100 shadow">
        <div className="text-lg font-semibold text-gray-700">
          <Link href="/newsadmin" className="text-blue-500 hover:underline">
            Bản tin
          </Link>
          &nbsp;&gt;&nbsp;Bản tin thông báo cá tra ở ĐBSCL
        </div>
        {!isDeleted && isAuthenticated && (
          <Button onClick={handleDelete} variant="destructive">
            Xóa
          </Button>
        )}
      </div>

      {/* Full-screen PDF Viewer */}
      <div className="flex-grow">
        {!isDeleted ? (
          <div className="w-full h-full flex">
            <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
              <div className="w-full h-[calc(100vh-50px)]"> 
                <Viewer fileUrl="/dot7.pdf" plugins={[defaultLayoutPluginInstance]} />
              </div>
            </Worker>
          </div>
        ) : (
          <p className="text-red-500 text-center text-lg">📌 File đã bị xóa</p>
        )}
      </div>
    </div>
  );
};

export default NewsPage;
