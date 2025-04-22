"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import dynamic from 'next/dynamic'; // Import next/dynamic
import { Button } from "@/components/ui/button";
import { Worker } from "@react-pdf-viewer/core"; // Keep Worker import
import "@react-pdf-viewer/core/lib/styles/index.css";
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout";
import PageLoader from "@/components/pageloader";

// Dynamically import the Viewer component
const Viewer = dynamic(() => import('@react-pdf-viewer/core').then(mod => mod.Viewer), {
  ssr: false,
  // Optional: Add a loading component while the Viewer loads
  loading: () => <p>Loading PDF Viewer...</p>,
});

import "@react-pdf-viewer/default-layout/lib/styles/index.css";


const NewsPage = () => {
  const [isLoading, setIsLoading] = useState(true);
    useEffect(() => {
        // Simulate loading delay (e.g., fetching data)
        const timeout = setTimeout(() => {
          setIsLoading(false);
        }, 1000); // 1s delay
        return () => clearTimeout(timeout);
      }, []);
      
  const [isDeleted, setIsDeleted] = useState(false);
  const isAuthenticated = true; // Simulated authentication state

  const handleDelete = () => {
    setIsDeleted(true);
  };

  // Create the plugin instance - this is fine to do outside dynamic import
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  if (isLoading) return <PageLoader message="Đang tải trang bản tin..." />;

  return (
    <div className="h-screen flex flex-col bg-white mb-4">
      {/* Breadcrumb + Delete Button */}
      <div className="flex justify-between items-center px-4 py-2 bg-gray-100 shadow">
        <div className="text-lg font-semibold text-gray-700">
          <Link href="/newsadmin" className="text-blue-500 hover:underline">
            Bản tin
          </Link>
          &nbsp;&gt;&nbsp;Bản tin thông báo cá tra ở ĐBSCL {/* Maybe make this dynamic later */}
        </div>
        {!isDeleted && isAuthenticated && (
          <Button onClick={handleDelete} variant="destructive">
            Xóa
          </Button>
        )}
      </div>

      {/* Full-screen PDF Viewer Area */}
      <div className="flex-grow">
        {!isDeleted ? (
          <div className="w-full h-full flex">
            {/* Worker needs a valid URL */}
            <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
              {/* Ensure the container div has a defined height */}
              <div style={{ height: 'calc(100vh - 60px)', width: '100%' }}> {/* Adjust height based on header height */}
                <Viewer fileUrl="/dot7.pdf" plugins={[defaultLayoutPluginInstance]} />
              </div>
            </Worker>
          </div>
        ) : (
          <p className="text-red-500 text-center text-lg p-10">📌 File đã bị xóa</p>
        )}
      </div>
    </div>
  );
};

export default NewsPage;