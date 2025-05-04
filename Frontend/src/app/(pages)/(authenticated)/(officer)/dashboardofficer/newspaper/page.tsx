"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Pagination } from "@/components/pagination";
import { useEffect, useState, useCallback } from "react";
import { FilterIcon, Loader2, Edit, Trash2 } from "lucide-react";
import Link from "next/link";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import PageLoader from "@/components/pageloader";
import Image from "next/image";
import type { Article } from "@/types/article"; 
import { getAllArticles, deleteArticle, generateProxyUrl } from "@/lib/article"; 
import type { QueryOptions } from "@/types/station2"; 
import { useAuth } from "@/hooks/useAuth"; 
import ArticleForm from "@/components/article-form"; 
import { toast, Toaster } from "react-hot-toast";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"; 
import { cn } from "@/lib/utils"; 

const ITEMS_PER_PAGE = 6;
const DEFAULT_IMAGE_URL = "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg";

type BadgeFilter = 'all' | 'danger' | 'good' | 'common';

const filterOptions: { label: string; value: BadgeFilter }[] = [
    { label: "Tất cả mức độ", value: "all" },
    { label: "Khẩn cấp", value: "danger" },
    { label: "Quan trọng", value: "good" },
    { label: "Thông thường", value: "common" },
];

// --- Component Definition ---
const NewsAdminPage = () => {
  const { session, isLoading: isAuthLoading, token, isLoggedIn } = useAuth();
  const [articles, setArticles] = useState<Article[]>([]); 
  const [filteredArticles, setFilteredArticles] = useState<Article[]>([]); 
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalArticles, setTotalArticles] = useState(0); 
  const [selectedBadge, setSelectedBadge] = useState<BadgeFilter>('all');

  // Modal/Dialog States
  const [isFormModalOpen, setIsFormModalOpen] = useState<boolean>(false);
  const [editingArticle, setEditingArticle] = useState<Article | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
  const [articleToDelete, setArticleToDelete] = useState<Article | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);

  // --- Data Fetching ---
  const fetchArticles = useCallback(async (page: number = 1, query: string = searchQuery, badge: BadgeFilter = selectedBadge) => {
    setIsLoading(true);
    setError(null);
    try {
      const options: QueryOptions = {
        limit: 1000, 
        offset: 0,
        sortBy: 'created_at',
        sortDesc: true,
      };

      const fetchedArticles = await getAllArticles(options) || [];
      setArticles(fetchedArticles); 
      let tempFiltered = fetchedArticles;
      if (query) {
        tempFiltered = tempFiltered.filter(item =>
          item.title.toLowerCase().includes(query.toLowerCase())
        );
      }
      if (badge !== 'all') {
        tempFiltered = tempFiltered.filter(item => {
          if (badge === 'common') {
            return !item.badge || (item.badge !== 'danger' && item.badge !== 'good');
          }
          return item.badge === badge;
        });
      }

      const totalFilteredCount = tempFiltered.length;
      setTotalArticles(totalFilteredCount);
      const startIndex = (page - 1) * ITEMS_PER_PAGE;
      const endIndex = startIndex + ITEMS_PER_PAGE;
      const paginatedArticles = tempFiltered.slice(startIndex, endIndex);

      setFilteredArticles(paginatedArticles); 

    } catch (err: any) {
      console.error("Admin: Failed to fetch articles:", err);
      setError("Không thể tải danh sách bản tin. Vui lòng thử lại.");
      toast.error("Lỗi tải bản tin: " + err.message);
      setArticles([]);
      setFilteredArticles([]);
      setTotalArticles(0);
    } finally {
      setIsLoading(false);
    }
  }, [searchQuery, selectedBadge]); 

  useEffect(() => {
    if (!isAuthLoading) {
      fetchArticles(currentPage, searchQuery, selectedBadge);
    }
  }, [fetchArticles, currentPage, searchQuery, selectedBadge, isAuthLoading]); 

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setCurrentPage(1); 
  };

  const handleBadgeFilterChange = (badge: BadgeFilter) => {
    setSelectedBadge(badge);
    setCurrentPage(1);
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const openCreateModal = () => {
    setEditingArticle(null);
    setIsFormModalOpen(true);
  };

  const openEditModal = (article: Article) => {
    setEditingArticle(article);
    setIsFormModalOpen(true);
  };

  const handleFormSuccess = (updatedOrNewArticle: Article) => {
    setIsFormModalOpen(false);
    setEditingArticle(null);
    fetchArticles(currentPage, searchQuery, selectedBadge);
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
       await deleteArticle(token, articleToDelete.id, true); // Assuming hard delete
       toast.success(`Đã xóa bản tin "${articleToDelete.title}"`);
       setIsDeleteDialogOpen(false);
       setArticleToDelete(null);

       // Recalculate page number after deletion
       const newTotalAfterDelete = totalArticles - 1;
       const newTotalPages = Math.ceil(newTotalAfterDelete / ITEMS_PER_PAGE);
       let pageToFetch = currentPage;

       // If the last item on a page > 1 was deleted, go to the previous page
       if (filteredArticles.length === 1 && currentPage > 1 && currentPage > newTotalPages) {
           pageToFetch = currentPage - 1;
           setCurrentPage(pageToFetch); // Update currentPage state as well
       }

       // Re-fetch data for the potentially adjusted page
       fetchArticles(pageToFetch, searchQuery, selectedBadge);

     } catch (err: any) {
        console.error(`Error deleting article ${articleToDelete.id}:`, err);
        toast.error(`Lỗi xóa bản tin: ${err.message}`);
     } finally {
        setIsDeleting(false);
     }
  };

  const totalPages = Math.ceil(totalArticles / ITEMS_PER_PAGE);

  if (isLoading && filteredArticles.length === 0 && articles.length === 0) return <PageLoader message="Đang tải trang bản tin..." />;
  // Error state when initial fetch fails
  if (error && articles.length === 0) return <div className="container mx-auto px-4 py-6 text-red-600">{error}</div>;

  return (
    <div className="container mx-auto px-4 py-6">
      <Toaster position="bottom-right" /> 
      <header className="mb-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <h1 className="text-3xl font-bold text-orange-500 shrink-0">Quản lý Bản tin</h1>
              <div className="flex flex-col sm:flex-row items-center gap-3 w-full md:w-auto">
                  {/* Create Button */}
                  {isLoggedIn && (
                    <Button onClick={openCreateModal} className="bg-orange-500 hover:bg-orange-600 text-white w-full sm:w-auto order-1 sm:order-3">
                      Đăng bản tin mới
                    </Button>
                  )}

                   {/* Badge Filter Dropdown */}
                   <Select
                     value={selectedBadge}
                     onValueChange={(value) => handleBadgeFilterChange(value as BadgeFilter)}
                   >
                     <SelectTrigger className="w-full sm:w-[180px] order-3 sm:order-1">
                       <SelectValue placeholder="Lọc theo mức độ" />
                     </SelectTrigger>
                     <SelectContent>
                       {filterOptions.map((option) => (
                         <SelectItem key={option.value} value={option.value}>
                           {option.label}
                         </SelectItem>
                       ))}
                     </SelectContent>
                   </Select>

                  {/* Search Input */}
                  <div className="relative w-full sm:w-auto flex-grow md:flex-grow-0 order-2 sm:order-2">
                      <Input
                          type="text"
                          placeholder="Tìm theo tiêu đề..."
                          value={searchQuery}
                          onChange={(e) => handleSearch(e.target.value)}
                          className="pl-10 pr-4 py-2 w-full sm:w-64"
                      />
                      <FilterIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                  </div>
              </div>
          </div>
      </header>

      {/* Loading Indicator (during filtering/pagination) */}
      {isLoading && (
          <div className="flex justify-center my-4">
              <Loader2 className="w-6 h-6 animate-spin text-orange-500" />
          </div>
      )}

      {/* No Results Message */}
      {!isLoading && filteredArticles.length === 0 && (
           <div className="text-center text-gray-500 mt-10">Không tìm thấy bản tin nào phù hợp với bộ lọc.</div>
      )}

      {/* Articles Grid */}
      {!isLoading && filteredArticles.length > 0 && (
         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
           {filteredArticles.map((item) => (
             <Card key={item.id} className="hover:shadow-lg transition-shadow flex flex-col h-full overflow-hidden relative group bg-white dark:bg-gray-800">
                {/* Link wraps content, excludes action buttons */}
                <Link href={`/dashboardofficer/newspaper/${item.id}`} className="block flex flex-col h-full flex-grow">
                   <div className="relative w-full h-48 sm:h-56">
                     <Image
                       src={generateProxyUrl(item.pictureUrl) || DEFAULT_IMAGE_URL}
                       alt={item.title}
                       fill
                       style={{ objectFit: "cover" }}
                       sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                       className="rounded-t-md"
                       onError={(e) => { (e.target as HTMLImageElement).src = DEFAULT_IMAGE_URL; }}
                     />
                     {/* Badge Display Logic */}
                     {item.badge && (item.badge === 'danger' || item.badge === 'good') && (
                         <span className={cn(
                             "absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white",
                             item.badge === 'danger' ? 'bg-red-500' : 'bg-green-500'
                         )}>
                             {item.badge === 'danger' ? 'Khẩn cấp' : 'Quan trọng'}
                         </span>
                     )}
                     {(!item.badge || (item.badge !== 'danger' && item.badge !== 'good')) && (
                         <span className="absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white bg-gray-500">
                             Thông thường
                         </span>
                     )}
                   </div>
                   <CardHeader className="pb-2 pt-4">
                     <CardTitle className="text-lg font-semibold line-clamp-2 text-gray-900 dark:text-gray-100">{item.title}</CardTitle>
                   </CardHeader>
                   <CardContent className="flex-grow flex flex-col justify-between pt-0">
                      {/* Ensure content exists before trying to substring */}
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2 line-clamp-3">
                        {item.content ? item.content.substring(0, 100) + (item.content.length > 100 ? '...' : '') : ''}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-500 mt-auto">
                         {new Date(item.createdAt).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                      </p>
                   </CardContent>
                </Link> {/* End of Link */}

                {/* Action Buttons (Edit/Delete) - Positioned absolutely */}
                {isLoggedIn && (
                   <div className="absolute top-2 left-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex gap-2 z-10">
                       {/* Edit Button */}
                       <Button
                           variant="outline"
                           size="icon"
                           className="bg-white hover:bg-gray-100 border-gray-300 rounded-full w-8 h-8"
                           onClick={(e) => { e.stopPropagation(); openEditModal(item); }} // Stop propagation to prevent link click
                           title="Chỉnh sửa"
                       >
                           <Edit className="h-4 w-4 text-blue-600" />
                       </Button>
                       {/* Delete Button */}
                       <Button
                           variant="outline"
                           size="icon"
                           className="bg-white hover:bg-gray-100 border-gray-300 rounded-full w-8 h-8"
                           onClick={(e) => { e.stopPropagation(); openDeleteDialog(item); }} // Stop propagation
                           title="Xóa"
                       >
                           <Trash2 className="h-4 w-4 text-red-600" />
                       </Button>
                   </div>
                )}
             </Card>
           ))}
         </div>
      )}

      {/* Pagination Controls */}
      {!isLoading && totalPages > 1 && filteredArticles.length > 0 && (
         <footer className="mt-8 flex justify-center">
           <Pagination
             currentPage={currentPage}
             totalPages={totalPages}
             onPageChange={handlePageChange}
           />
         </footer>
      )}

      {/* Modals/Dialogs */}
      {/* Create/Edit Form Dialog */}
      <Dialog open={isFormModalOpen} onOpenChange={setIsFormModalOpen}>
         <DialogContent className="max-w-3xl max-h-[95vh] overflow-y-auto p-6">
           <DialogHeader>
             <DialogTitle className="text-center mb-4 text-2xl font-semibold">
               {editingArticle ? 'Chỉnh sửa bản tin' : 'Đăng bản tin mới'}
             </DialogTitle>
           </DialogHeader>
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
               {isDeleting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
               Xóa vĩnh viễn
             </AlertDialogAction>
           </AlertDialogFooter>
         </AlertDialogContent>
       </AlertDialog>

    </div> // End of container div
  );
};

export default NewsAdminPage;