// src/app/newsadmin/page.tsx
"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Pagination } from "@/components/pagination"; // Đảm bảo component này tồn tại và đúng
import { useEffect, useState, useCallback } from "react";
import { FilterIcon, Loader2, Edit, Trash2 } from "lucide-react";
import Link from "next/link";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import PageLoader from "@/components/pageloader";
import Image from "next/image";
import { Article } from "@/types/article";
import { getAllArticles, deleteArticle, generateProxyUrl } from "@/lib/article";
import { QueryOptions } from "@/types/station2"; // Đổi tên type nếu cần
import { useAuth } from "@/hooks/useAuth"; // Import hook xác thực
import ArticleForm from "@/components/article-form"; // Import form component
import { toast, Toaster } from "react-hot-toast"; // Import toast
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog"; // For delete confirmation


const ITEMS_PER_PAGE = 6;
const DEFAULT_IMAGE_URL = "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg"; // URL ảnh mặc định

const NewsAdminPage = () => {
  const { session, isLoading: isAuthLoading, token, isLoggedIn } = useAuth();
  const [articles, setArticles] = useState<Article[]>([]);
  const [filteredArticles, setFilteredArticles] = useState<Article[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalArticles, setTotalArticles] = useState(0); // Cần API trả về total hoặc tính toán khác

  // State cho Dialog Form (Tạo/Sửa)
  const [isFormModalOpen, setIsFormModalOpen] = useState<boolean>(false);
  const [editingArticle, setEditingArticle] = useState<Article | null>(null);

  // State cho Dialog xác nhận xóa
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
  const [articleToDelete, setArticleToDelete] = useState<Article | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);


  // Hàm fetch dữ liệu
  const fetchArticles = useCallback(async (page: number = 1, query: string = searchQuery) => {
    setIsLoading(true);
    setError(null);
    try {
      const options: QueryOptions = {
        limit: ITEMS_PER_PAGE,
        offset: (page - 1) * ITEMS_PER_PAGE,
        sortBy: 'created_at', // Sắp xếp theo ngày tạo mới nhất
        sortDesc: true,
        // includeDeleted: false // Mặc định API không trả về bài đã xóa mềm
      };
      // console.log("Fetching articles with options:", options);
      // *** Lưu ý: API getAllArticles hiện tại không trả về tổng số lượng ***
      // Chúng ta sẽ fetch và giả định tổng số dựa trên kết quả trả về (không lý tưởng cho phân trang chính xác)
      const fetchedArticles = await getAllArticles(options);
      // console.log("Fetched articles:", fetchedArticles);

      setArticles(fetchedArticles);

      // Lọc phía client (vì API chưa hỗ trợ search)
      const filtered = fetchedArticles.filter(item =>
        item.title.toLowerCase().includes(query.toLowerCase())
      );
       setFilteredArticles(filtered);

      // Ước tính totalPages - Cần cải thiện nếu API trả về total count
       // Nếu số lượng trả về < limit, đây có thể là trang cuối cùng
       const estimatedTotal = (page - 1) * ITEMS_PER_PAGE + fetchedArticles.length;
       // Nếu fetch được đúng số lượng items_per_page, có thể còn trang sau -> ước tính cao hơn
       const potentialTotal = (fetchedArticles.length === ITEMS_PER_PAGE) ? estimatedTotal + 1 : estimatedTotal;
       setTotalArticles(potentialTotal) // Ước tính rất thô sơ

       setCurrentPage(page);

    } catch (err: any) {
      console.error("Failed to fetch articles:", err);
      setError("Không thể tải danh sách bản tin. Vui lòng thử lại.");
      toast.error("Lỗi tải bản tin: " + err.message);
      setArticles([]);
      setFilteredArticles([]);
    } finally {
      setIsLoading(false);
    }
  }, [searchQuery]); // Thêm searchQuery vào dependency array

  // Fetch lần đầu khi component mount hoặc searchQuery thay đổi
  useEffect(() => {
    // Chỉ fetch khi quá trình xác thực hoàn tất
     if (!isAuthLoading) {
        fetchArticles(1, searchQuery); // Fetch trang đầu tiên với query hiện tại
     }
  }, [fetchArticles, isAuthLoading, searchQuery]); // Chạy lại khi hàm fetch hoặc trạng thái auth hoặc query thay đổi

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    // Việc fetch sẽ được trigger bởi useEffect ở trên khi searchQuery thay đổi
  };

  const handlePageChange = (page: number) => {
     fetchArticles(page, searchQuery); // Fetch trang mới với query hiện tại
  };

  const openCreateModal = () => {
    setEditingArticle(null); // Đảm bảo đang ở mode tạo mới
    setIsFormModalOpen(true);
  };

  const openEditModal = (article: Article) => {
    setEditingArticle(article);
    setIsFormModalOpen(true);
  };

  const handleFormSuccess = (updatedOrNewArticle: Article) => {
    setIsFormModalOpen(false);
    setEditingArticle(null);
    // Refresh lại danh sách để thấy thay đổi
    fetchArticles(currentPage, searchQuery);
  };

  const openDeleteDialog = (article: Article) => {
     setArticleToDelete(article);
     setIsDeleteDialogOpen(true);
  };

  const confirmDelete = async () => {
     if (!articleToDelete || !token) return;

     setIsDeleting(true);
     try {
        console.log(`Attempting to delete article ${articleToDelete.id}`);
        await deleteArticle(token, articleToDelete.id, true); // Hard delete = true
        toast.success(`Đã xóa bản tin "${articleToDelete.title}"`);
        setIsDeleteDialogOpen(false);
        setArticleToDelete(null);
        // Refresh lại danh sách sau khi xóa
        // Nếu xóa item cuối cùng của trang hiện tại, nên quay về trang trước (nếu có)
        const newTotalPages = Math.ceil((totalArticles -1) / ITEMS_PER_PAGE); // Tính lại tổng số trang
        const pageToFetch = (filteredArticles.length === 1 && currentPage > 1) ? currentPage - 1 : currentPage;
        fetchArticles(pageToFetch, searchQuery);

     } catch (err: any) {
         console.error(`Error deleting article ${articleToDelete.id}:`, err);
         toast.error(`Lỗi xóa bản tin: ${err.message}`);
     } finally {
         setIsDeleting(false);
     }
  };

  // Tính toán totalPages dựa trên totalArticles (ước tính)
  const totalPages = Math.ceil(totalArticles / ITEMS_PER_PAGE);

  // Dùng filteredArticles để hiển thị
  //const displayedNews = filteredArticles; // Không cần slice nữa vì API đã phân trang

  if (isLoading || isAuthLoading) return <PageLoader message="Đang tải trang bản tin..." />;
  if (error && articles.length === 0) return <div className="container mx-auto px-4 py-6 text-red-600">{error}</div>;

  return (
    <div className="container mx-auto px-4 py-6">
      <Toaster position="top-right" /> {/* Component để hiển thị toast */}
      <header className="flex flex-col sm:flex-row items-center justify-between mb-6 gap-4">
        <h1 className="text-3xl font-bold text-orange-500">Quản lý Bản tin</h1>
        <div className="flex items-center space-x-2 w-full sm:w-auto">
         {isLoggedIn && (
            <Button onClick={openCreateModal} className="bg-orange-500 hover:bg-orange-600 text-white">
              Đăng bản tin mới
            </Button>
          )}
          <div className="relative flex-grow sm:flex-grow-0">
            <Input
              type="text"
              placeholder="Lọc theo tiêu đề..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)} // Chỉ cập nhật state, useEffect sẽ fetch
              className="pl-10 w-full"
            />
            <FilterIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          </div>
        </div>
      </header>

       {filteredArticles.length === 0 && !isLoading && (
           <div className="text-center text-gray-500 mt-10">Không tìm thấy bản tin nào.</div>
       )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
         {filteredArticles.map((item) => (
          <Card key={item.id} className="hover:shadow-lg transition-shadow flex flex-col h-full overflow-hidden relative group">

            <Link href={`/dashboardofficer/newspaper/${item.id}`} className="block h-full">
              {/* Image */}
              <div className="relative w-full h-48 sm:h-56"> {/* Giảm chiều cao ảnh một chút */}
                <Image
                  // Sử dụng ảnh từ article.pictureUrl hoặc ảnh mặc định
                  src={generateProxyUrl(item.pictureUrl) || DEFAULT_IMAGE_URL}
                  alt={item.title}
                  fill
                  style={{ objectFit: "cover" }}
                  sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                  className="rounded-t-md"
                  // Xử lý lỗi nếu ảnh không tải được
                   onError={(e) => { (e.target as HTMLImageElement).src = DEFAULT_IMAGE_URL; }}
                />
                 {/* Badge */}
                 <span className={`absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white ${
                     item.badge === 'danger' ? 'bg-red-500' : item.badge === 'good' ? 'bg-green-500' : 'bg-gray-500'
                 }`}>
                     {item.badge === 'danger' ? 'Khẩn cấp' : item.badge === 'good' ? 'Tốt' : 'Thông thường'}
                 </span>
              </div>

              {/* Content */}
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-semibold line-clamp-2">{item.title}</CardTitle> {/* Giới hạn 2 dòng */}
              </CardHeader>
              <CardContent className="flex-grow flex flex-col justify-between pt-0">
                 <p className="text-sm text-gray-600 mb-2 line-clamp-3">{item.content}</p> {/* Giới hạn 3 dòng */}
                 <p className="text-xs text-gray-500 mt-auto">
                   {new Date(item.createdAt).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                 </p>
              </CardContent>
            </Link>
             {/* Edit/Delete Buttons - Chỉ hiển thị nếu đăng nhập */}
             {isLoggedIn && (
                <div className="absolute top-2 left-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex gap-2">
                     <Button
                         variant="outline"
                         size="icon"
                         className="bg-white hover:bg-gray-100 border-gray-300 rounded-full w-8 h-8"
                         onClick={() => openEditModal(item)}
                         title="Chỉnh sửa"
                     >
                         <Edit className="h-4 w-4 text-blue-600" />
                     </Button>

                      {/* Trigger cho AlertDialog */}
                      
                         <Button
                             variant="outline"
                             size="icon"
                             className="bg-white hover:bg-gray-100 border-gray-300 rounded-full w-8 h-8"
                             onClick={() => openDeleteDialog(item)} // Set article cần xóa khi click trigger
                             title="Xóa"
                         >
                             <Trash2 className="h-4 w-4 text-red-600" />
                         </Button>
                     
                 </div>
             )}
          </Card>
        ))}
      </div>

      {/* Pagination */}
       {totalPages > 1 && (
        <footer className="mt-8 flex justify-center">
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </footer>
      )}

      {/* Form Modal (Dialog) */}
       <Dialog open={isFormModalOpen} onOpenChange={setIsFormModalOpen}>
          <DialogContent className="max-w-3xl max-h-[95vh] overflow-y-auto p-6">
             <DialogHeader>
                <DialogTitle className="text-center mb-4 text-2xl font-semibold">
                   {editingArticle ? 'Chỉnh sửa bản tin' : 'Đăng bản tin mới'}
                </DialogTitle>
             </DialogHeader>
             {/* Render form chỉ khi có token */}
             {token ? (
                <ArticleForm
                   token={token}
                   existingArticle={editingArticle}
                   onSuccess={handleFormSuccess}
                   onCancel={() => setIsFormModalOpen(false)}
                />
             ) : (
                <p className="text-red-600 text-center">Bạn cần đăng nhập để thực hiện hành động này.</p>
             )}
          </DialogContent>
       </Dialog>

       {/* Delete Confirmation Dialog */}
       <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
         <AlertDialogContent>
           <AlertDialogHeader>
             <AlertDialogTitle>Xác nhận xóa bản tin?</AlertDialogTitle>
             <AlertDialogDescription>
               Bạn có chắc chắn muốn xóa vĩnh viễn bản tin &quot;{articleToDelete?.title}&quot; không? Hành động này không thể hoàn tác.
             </AlertDialogDescription>
           </AlertDialogHeader>
           <AlertDialogFooter>
             <AlertDialogCancel onClick={() => setArticleToDelete(null)} disabled={isDeleting}>Hủy</AlertDialogCancel>
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

export default NewsAdminPage;