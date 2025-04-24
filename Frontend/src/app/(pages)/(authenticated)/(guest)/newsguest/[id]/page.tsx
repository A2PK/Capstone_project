// src/app/(pages)/newsguest/[id]/page.tsx

"use client";

import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import dynamic from 'next/dynamic';

// UI Components
// Bỏ Button vì không còn dùng cột bên trái nữa
import { Worker } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout";
import "@react-pdf-viewer/default-layout/lib/styles/index.css";
import PageLoader from "@/components/pageloader";
import { Loader2, ArrowLeft, Info, AlertCircle } from "lucide-react"; // Icons

// Types
import { Article } from "@/types/article";
import { FileInfo } from "@/types/file";

// API & Helpers
import { getArticleById, generateProxyUrl } from "@/lib/article";
import { getMultipleFilesByFileIds } from "@/lib/file";

// Dynamic import Viewer - Chỉ hiện loading indicator đơn giản khi Viewer tải
const Viewer = dynamic(() => import('@react-pdf-viewer/core').then(mod => mod.Viewer), {
    ssr: false,
    loading: () => (
        <div className="flex justify-center items-center h-full w-full bg-gray-100">
            <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
        </div>
    ),
});

const NewsGuestDetailPage = () => {
    const params = useParams();
    const articleId = params?.id as string;

    const [article, setArticle] = useState<Article | null>(null);
    const [pdfFile, setPdfFile] = useState<FileInfo | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null); // Lỗi fetch article
    const [pdfFetchError, setPdfFetchError] = useState<string | null>(null); // Lỗi fetch file info
    const [pdfRenderError, setPdfRenderError] = useState<string | null>(null); // Lỗi render PDF của Viewer

    const fetchArticleDetails = useCallback(async () => {
        if (!articleId) {
            setError("ID bản tin không hợp lệ.");
            setIsLoading(false);
            return;
        }
        setIsLoading(true);
        setError(null);
        setPdfFetchError(null);
        setPdfRenderError(null); // Reset lỗi render
        setPdfFile(null);

        try {
            console.log(`Guest: Fetching article details for ID: ${articleId}`);
            const fetchedArticle = await getArticleById(articleId);
            console.log("Guest: Fetched article:", fetchedArticle);
            setArticle(fetchedArticle);

            if (fetchedArticle.fileIds && fetchedArticle.fileIds.length > 0) {
                const mainFileId = fetchedArticle.fileIds[0];
                console.log(`Guest: Attempting to fetch file info for ID [${mainFileId}] using "fake" token.`);
                try {
                    // Cần một token hợp lệ ở đây nếu API yêu cầu, thay "YOUR_VALID_TOKEN_OR_LOGIC"
                    const filesInfo = await getMultipleFilesByFileIds("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7ImVtYWlsIjoidXNlckBnbWFpbC5jb20iLCJyb2xlIjoib2ZmaWNlciIsInN1YiI6ImEyM2ZkMWY0LThmZWEtNGQxZi04OGMxLTE1NjY3ZGI3YTQ3YSJ9LCJpc3MiOiJ3cWltS01UIiwic3ViIjoiYTIzZmQxZjQtOGZlYS00ZDFmLTg4YzEtMTU2NjdkYjdhNDdhIiwiZXhwIjoxNzQ1OTkxMjA5LCJpYXQiOjE3NDUzODY0MDl9.TkPqpDGuhsAdd8qItxecAIiDKQUjlWBGn9rMP8avjOY", { ids: [mainFileId] });
                    console.log("Guest: Fetched file info:", filesInfo);
                    if (filesInfo && filesInfo.length > 0) {
                        if (filesInfo[0].type === 'application/pdf' || filesInfo[0].name?.toLowerCase().endsWith('.pdf')) {
                            setPdfFile(filesInfo[0]);
                        } else {
                            console.warn("Guest: Attached file is not a PDF:", filesInfo[0].name);
                            setPdfFetchError("File đính kèm không phải là định dạng PDF.");
                        }
                    } else {
                        console.warn("Guest: File info not found for ID:", mainFileId);
                        setPdfFetchError("Không tìm thấy thông tin file PDF đính kèm.");
                    }
                } catch (fileErr: any) {
                    console.error("Guest: Failed to fetch file info:", fileErr);
                    setPdfFetchError(`Lỗi tải thông tin file PDF: ${fileErr.message || "Lỗi không xác định"}`);
                }
            } else {
                console.log("Guest: Article has no associated fileIds.");
                setPdfFetchError("Bản tin này không có file PDF đính kèm."); // Set lỗi nếu không có fileIds
            }

        } catch (err: any) {
            console.error("Guest: Failed to fetch article details:", err);
            if (err.message?.includes("404") || err.message?.toLowerCase().includes("not found")) {
                setError("Không tìm thấy bản tin này.");
            } else {
                setError(`Lỗi tải chi tiết bản tin: ${err.message}`);
            }
            setArticle(null);
        } finally {
            setIsLoading(false);
        }
    }, [articleId]);

    useEffect(() => {
        if (articleId) {
            fetchArticleDetails();
        }
    }, [articleId, fetchArticleDetails]);

    const defaultLayoutPluginInstance = defaultLayoutPlugin();

    // --- Render Logic ---

    if (isLoading) return <PageLoader message="Đang tải chi tiết bản tin..." />;

    // Lỗi nghiêm trọng khi không fetch được Article
    if (error && !article) {
        return (
            <div className="flex flex-col min-h-screen bg-gray-50">
                <div className="flex justify-between items-center px-4 py-2 bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
                    <Link href="/newsguest" className="text-orange-600 hover:underline flex items-center text-sm">
                        <ArrowLeft className="h-4 w-4 mr-1" />
                        Quay lại Bản tin
                    </Link>
                    <span className="font-medium text-red-600 text-sm">Lỗi Tải Bản Tin</span>
                </div>
                <div className="flex-grow flex justify-center items-center p-10">
                    <p className="text-red-600 text-center text-lg">⚠️ {error}</p>
                </div>
            </div>
        );
    }

    // Trường hợp có Article nhưng gặp vấn đề với file PDF (fetch hoặc render) hoặc không có file
    const shouldShowPdfError = pdfFetchError || pdfRenderError || (article && !pdfFile && !isLoading); // Thêm đk !isLoading để tránh hiện lỗi khi đang fetch

    return (
        <div className="flex flex-col h-screen bg-gray-100"> {/* h-screen để chiếm toàn bộ chiều cao */}
            {/* Header: Luôn hiển thị */}
            <div className="flex justify-between items-center px-4 py-2 bg-white shadow-sm border-b border-gray-200 flex-shrink-0 h-12"> {/* flex-shrink-0 và h-12 để cố định chiều cao */}
                <div className="flex items-center space-x-4 overflow-hidden">
                    <Link href="/newsguest" className="text-orange-600 hover:underline flex items-center text-sm flex-shrink-0">
                        <ArrowLeft className="h-4 w-4 mr-1" />
                        <span className="hidden sm:inline">Quay lại Bản tin</span> {/* Ẩn chữ trên màn hình nhỏ */}
                    </Link>
                    {/* Hiển thị tiêu đề article nếu có */}
                    {article && (
                        <>
                            <span className="text-gray-300 hidden sm:inline">|</span>
                            <span className="font-medium text-gray-800 text-sm truncate" title={article.title}>
                                {article.title}
                            </span>
                            {/* Có thể thêm Badge ở đây nếu muốn */}
                            <span className={`ml-2 px-1.5 py-0.5 text-xs font-semibold rounded text-white ${
                                article.badge === 'danger' ? 'bg-red-500' : article.badge === 'good' ? 'bg-green-500' : 'bg-gray-500'
                            }`}>
                                {article.badge === 'danger' ? 'Khẩn' : article.badge === 'good' ? 'Tốt' : 'Thường'}
                            </span>
                        </>
                    )}
                     {!article && !isLoading && ( // Chỉ hiển thị nếu không loading và không có article (trường hợp lạ?)
                         <span className="font-medium text-yellow-600 text-sm">Đang chờ dữ liệu...</span>
                     )}
                </div>
                 {/* Có thể thêm nút Info để xem chi tiết content nếu cần */}
                 {/* <Button variant="ghost" size="sm"> <Info className="h-4 w-4"/> </Button> */}
            </div>

            {/* PDF Viewer Area: Chiếm phần còn lại */}
            <div className="flex-grow overflow-hidden"> {/* overflow-hidden để viewer không tràn */}
                {shouldShowPdfError ? (
                     // Hiển thị lỗi liên quan đến PDF
                    <div className="flex flex-col items-center justify-center h-full p-6 text-center bg-gray-50">
                         <AlertCircle className="h-12 w-12 mb-4 text-red-500"/>
                         <p className="font-semibold text-lg text-red-700">Không thể hiển thị file PDF</p>
                         <p className="text-sm mt-2 text-gray-600">
                             {pdfRenderError ? `Lỗi render: ${pdfRenderError}` : pdfFetchError ? pdfFetchError : "Không có file PDF hoặc đã xảy ra lỗi."}
                         </p>
                         {pdfFile && <p className="text-xs mt-3 text-gray-400">URL gốc: {pdfFile.url}</p> }
                         {/* Nút thử lại có thể hữu ích */}
                         {/* <Button onClick={fetchArticleDetails} className="mt-4">Thử lại</Button> */}
                    </div>
                ) : pdfFile ? (
                    // Hiển thị PDF Viewer
                    <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
                        <div className="h-full w-full">
                        <Viewer
                                fileUrl={generateProxyUrl(pdfFile.url)}
                                plugins={[defaultLayoutPluginInstance]}
                                renderError={(error: any) => {
                                    console.error("PDF Viewer Rendering Error:", error);
                                    return (
                                        <div style={{ padding: '1rem', color: 'red', textAlign: 'center' }}>
                                            Đang xử lý lỗi hiển thị...
                                        </div>
                                     );
                                }}
                             />
                        </div>
                    </Worker>
                ) : (
                    // Trạng thái chờ (khi article load xong nhưng pdfFile chưa có, không phải lỗi)
                    <div className="flex justify-center items-center h-full w-full bg-gray-100">
                         <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
                         <span className="ml-2 text-gray-600">Đang chuẩn bị file PDF...</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default NewsGuestDetailPage;