
"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation"; // Import hooks từ next/navigation
import dynamic from 'next/dynamic';
import { Button } from "@/components/ui/button";
import { Worker } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout";
import "@react-pdf-viewer/default-layout/lib/styles/index.css";
import PageLoader from "@/components/pageloader";
import { Article } from "@/types/article";
import { FileInfo } from "@/types/file";
import { getArticleById, deleteArticle, generateProxyUrl } from "@/lib/article";
import { getMultipleFilesByFileIds } from "@/lib/file";
import { useAuth } from "@/hooks/useAuth";
import { toast, Toaster } from "react-hot-toast";
import { Loader2, Edit, Trash2, ArrowLeft } from "lucide-react";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog"; // For delete confirmation
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"; // For Edit Modal
import ArticleForm from "@/components/article-form"; // Import form


// Dynamically import the Viewer component for client-side rendering
const Viewer = dynamic(() => import('@react-pdf-viewer/core').then(mod => mod.Viewer), {
  ssr: false,
  loading: () => <div className="flex justify-center items-center h-full"><Loader2 className="h-8 w-8 animate-spin text-orange-500" /> Đang tải PDF Viewer...</div>,
});

const NewsDetailPage = () => {
  const params = useParams(); // Lấy { id: '...' } từ URL
  const router = useRouter();
  const { session, isLoading: isAuthLoading, token, isLoggedIn } = useAuth();
  const articleId = params?.id as string; // Lấy ID từ params

  const [article, setArticle] = useState<Article | null>(null);
  const [pdfFile, setPdfFile] = useState<FileInfo | null>(null); // Chỉ lưu thông tin file PDF chính
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDeleted, setIsDeleted] = useState(false); // Trạng thái sau khi xóa thành công

   // State cho Dialog Form (Sửa)
  const [isFormModalOpen, setIsFormModalOpen] = useState<boolean>(false);

  // State cho Dialog xác nhận xóa
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);


  const fetchArticleDetails = useCallback(async () => {
     if (!articleId) {
         setError("ID bản tin không hợp lệ.");
         setIsLoading(false);
         return;
     }
    setIsLoading(true);
    setError(null);
    setIsDeleted(false);

    try {
      // 1. Fetch Article data
       console.log(`Workspaceing article details for ID: ${articleId}`);
      const fetchedArticle = await getArticleById(articleId);
       console.log("Fetched article:", fetchedArticle);
      setArticle(fetchedArticle);

      // 2. Fetch FileInfo if fileIds exist and token is available
      if (fetchedArticle.fileIds && fetchedArticle.fileIds.length > 0 && token) {
         const mainFileId = fetchedArticle.fileIds[0]; // Giả sử file đầu tiên là file PDF chính
         console.log(`Workspaceing file info for IDs: [${mainFileId}]`);
         // API getMultipleFilesByFileIds yêu cầu token
         const filesInfo = await getMultipleFilesByFileIds(token, { ids: [mainFileId] });
         console.log("Fetched file info:", filesInfo);
         if (filesInfo && filesInfo.length > 0) {
             setPdfFile(filesInfo[0]); // Lưu thông tin file PDF chính
         } else {
             console.warn("Không tìm thấy thông tin file cho ID:", mainFileId);
             // Có thể set lỗi hoặc hiển thị thông báo file không tồn tại
             setError("Không tìm thấy file PDF đính kèm.");
         }
      } else if (fetchedArticle.fileIds && fetchedArticle.fileIds.length > 0 && !token) {
          console.warn("Có file ID nhưng thiếu token để fetch thông tin file.");
          setError("Không thể tải file PDF do cần xác thực."); // Hoặc cho phép xem nếu API file không cần token?
      }

    } catch (err: any) {
      console.error("Failed to fetch article details:", err);
       if (err.message?.includes("404") || err.message?.toLowerCase().includes("not found")) {
          setError("Không tìm thấy bản tin này.");
       } else {
          setError("Lỗi tải chi tiết bản tin: " + err.message);
       }
      toast.error("Lỗi tải chi tiết bản tin.");
      setArticle(null);
      setPdfFile(null);
    } finally {
      setIsLoading(false);
    }
  }, [articleId, token]); // Thêm token vào dependencies vì nó cần cho getMultipleFilesByFileIds

  useEffect(() => {
    // Chỉ fetch khi articleId có giá trị và quá trình xác thực ban đầu hoàn tất
    if (articleId && !isAuthLoading) {
      fetchArticleDetails();
    }
  }, [articleId, isAuthLoading, fetchArticleDetails]); // Chạy lại khi ID thay đổi hoặc auth hoàn tất hoặc hàm fetch thay đổi

  const openEditModal = () => {
    setIsFormModalOpen(true);
  };

  const handleFormSuccess = (updatedArticle: Article) => {
    setIsFormModalOpen(false);
    // Refresh lại dữ liệu sau khi sửa thành công
    fetchArticleDetails();
  };


  const confirmDelete = async () => {
     if (!articleId || !token) return;

     setIsDeleting(true);
     try {
        console.log(`Attempting to delete article ${articleId}`);
        await deleteArticle(token, articleId, true); // Hard delete = true
        toast.success(`Đã xóa bản tin thành công!`);
        setIsDeleted(true); // Cập nhật trạng thái đã xóa
        setIsDeleteDialogOpen(false); // Đóng dialog xác nhận
        // Không cần redirect ngay, chỉ hiển thị thông báo đã xóa
        //router.push('/dashboardofficer/newspaper'); // Chuyển về trang danh sách sau khi xóa

     } catch (err: any) {
         console.error(`Error deleting article ${articleId}:`, err);
         toast.error(`Lỗi xóa bản tin: ${err.message}`);
     } finally {
         setIsDeleting(false);
     }
  };


  // Khởi tạo plugin PDF viewer
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  if (isLoading || isAuthLoading) return <PageLoader message="Đang tải chi tiết bản tin..." />;


  return (
    <div className="h-screen flex flex-col bg-gray-50">
       <Toaster position="top-right" />
      <div className="flex justify-between items-center px-4 py-2 bg-white shadow-sm border-b border-gray-200">
          <div className="flex items-center text-sm sm:text-base text-gray-600">
             <Link href= "/dashboardofficer/newspaper" className="text-orange-600 hover:underline flex items-center">
                <ArrowLeft className="h-4 w-4 mr-1" />
                Quay lại Bản tin
             </Link>
             {article && !isDeleted && (
                <>
                 <span className="mx-2 text-gray-400">&gt;</span>
                 <span className="font-medium text-gray-800 truncate" title={article.title}>{article.title}</span>
                </>
             )}
             {isDeleted && (
                 <>
                   <span className="mx-2 text-gray-400">&gt;</span>
                   <span className="font-medium text-red-600">Bản tin đã bị xóa</span>
                 </>
             )}
             {error && !article && !isDeleted && (
                <>
                   <span className="mx-2 text-gray-400">&gt;</span>
                   <span className="font-medium text-red-600">Không tải được bản tin</span>
                 </>
             )}
         </div>
         {/* Nút Edit/Delete chỉ hiện khi có bài báo, chưa bị xóa và đã đăng nhập */}
          {isLoggedIn && article && !isDeleted && (
           <div className="flex items-center space-x-2">
               <Button variant="outline" size="sm" onClick={openEditModal}>
                  <Edit className="h-4 w-4 mr-1" /> Sửa
               </Button>
                  <Button variant="destructive" size="sm" onClick={() => setIsDeleteDialogOpen(true)}>
                     <Trash2 className="h-4 w-4 mr-1" /> Xóa
                  </Button>
                
           </div>
         )}
      </div>

      {/* Content Area */}
      <div className="flex-grow overflow-y-auto">
         {/* Trường hợp đang tải hoặc lỗi */}
          {isLoading && <div className="flex justify-center items-center h-full"><Loader2 className="h-8 w-8 animate-spin text-orange-500" /></div>}
          {error && (
              <div className="flex justify-center items-center h-full">
                  <p className="text-red-600 text-center text-lg p-10">⚠️ {error}</p>
              </div>
          )}
          {isDeleted && (
              <div className="flex flex-col justify-center items-center h-full text-center">
                  <p className="text-red-500 text-xl font-semibold mb-4">📌 Bản tin đã bị xóa.</p>
                  <Button onClick={() => router.push('/dashboardofficer/newspaper')}>
                      <ArrowLeft className="h-4 w-4 mr-2" />
                      Quay về danh sách bản tin
                  </Button>
              </div>
          )}

          {!isLoading && !error && article && pdfFile && !isDeleted && (
             <div className="w-full h-full">
                 <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
                    {/* Đảm bảo container có chiều cao */}
                    <div style={{ height: 'calc(100vh - 50px)' }}> {/* Chiều cao trừ đi header */}
                       <Viewer
                           fileUrl={generateProxyUrl(pdfFile.url)} // Sử dụng URL từ FileInfo
                           plugins={[defaultLayoutPluginInstance]}
                           // Xử lý lỗi tải file PDF
                           renderError={(err: any) => (
                               <div style={{ padding: '1rem', color: 'red', textAlign: 'center' }}>
                                   <p>Lỗi tải file PDF:</p>
                                   <p>{err.message || 'Không thể hiển thị file.'}</p>
                                   <p>URL: {pdfFile.url}</p>
                               </div>
                           )}
                       />
                    </div>
                 </Worker>
             </div>
          )}

          {/* Trường hợp có article nhưng không có file PDF */}
          {!isLoading && !error && article && !pdfFile && !isDeleted && (
               <div className="flex justify-center items-center h-full">
                  <p className="text-gray-500 text-center text-lg p-10">📄 Bản tin này không có file PDF đính kèm.</p>
               </div>
           )}
      </div>

       {/* Edit Form Modal (Dialog) */}
       <Dialog open={isFormModalOpen} onOpenChange={setIsFormModalOpen}>
          <DialogContent className="max-w-3xl max-h-[95vh] overflow-y-auto p-6">
             <DialogHeader>
                <DialogTitle className="text-center mb-4 text-2xl font-semibold">
                   Chỉnh sửa bản tin
                </DialogTitle>
             </DialogHeader>
             {/* Render form chỉ khi có token và article */}
             {token && article ? (
                <ArticleForm
                   token={token}
                   existingArticle={article} // Truyền article hiện tại vào form
                   onSuccess={handleFormSuccess}
                   onCancel={() => setIsFormModalOpen(false)}
                />
             ) : (
                <p className="text-red-600 text-center">Không thể tải form chỉnh sửa.</p>
             )}
          </DialogContent>
       </Dialog>


      {/* Delete Confirmation Dialog */}
      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Xác nhận xóa bản tin?</AlertDialogTitle>
            <AlertDialogDescription>
              Bạn có chắc chắn muốn xóa vĩnh viễn bản tin &quot;{article?.title}&quot; không? Hành động này không thể hoàn tác.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => {}} disabled={isDeleting}>Hủy</AlertDialogCancel> {/* Chỉ cần đóng dialog */}
            <AlertDialogAction
               onClick={confirmDelete}
               disabled={isDeleting}
               className="bg-red-600 hover:bg-red-700 text-white"
            >
              {isDeleting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Xóa vĩnh viễn
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default NewsDetailPage;