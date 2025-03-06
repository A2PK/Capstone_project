"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Viewer } from "@react-pdf-viewer/core";
import { Worker } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";
import "@react-pdf-viewer/default-layout/lib/styles/index.css";

const PdfViewer = ({ pdfUrl, isAuthenticated }: { pdfUrl: string; isAuthenticated: boolean }) => {
  const [isDeleted, setIsDeleted] = useState(false);

  const handleDelete = () => {
    // Thực hiện xóa file ở backend (API request)
    setIsDeleted(true);
  };

  if (isDeleted) return <p className="text-red-500 text-center mt-4">File đã bị xóa</p>;

  return (
    <div className="flex flex-col items-center p-4 bg-white shadow-lg rounded-lg">
      <div className="w-full h-[600px] border">
        <Worker workerUrl={`https://unpkg.com/pdfjs-dist@2.14.305/build/pdf.worker.min.js`}>
          <Viewer fileUrl={pdfUrl} />
        </Worker>
      </div>

      {isAuthenticated && (
        <Button onClick={handleDelete} variant="destructive" className="mt-4">
          Xóa
        </Button>
      )}
    </div>
  );
};

export default PdfViewer;
