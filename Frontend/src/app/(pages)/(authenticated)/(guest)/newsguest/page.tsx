// src/app/(pages)/newsguest/page.tsx

"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
// Giả sử bạn có component Pagination này
import { Pagination } from "@/components/pagination";
import { useEffect, useState, useCallback } from "react";
import { FilterIcon, Loader2 } from "lucide-react"; // Bỏ Edit, Trash2
import Link from "next/link";
// Bỏ Dialog, ArticleForm
import PageLoader from "@/components/pageloader";
import Image from "next/image";
import { Article } from "@/types/article";
// Bỏ deleteArticle, giữ lại getAllArticles và generateProxyUrl
import { getAllArticles, generateProxyUrl } from "@/lib/article";
import { QueryOptions } from "@/types/station2";
// Bỏ useAuth, useToast, Toaster, AlertDialog
// import { useAuth } from "@/hooks/useAuth";
// import { toast, Toaster } from "react-hot-toast";
// import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";

const ITEMS_PER_PAGE = 6;
const DEFAULT_IMAGE_URL = "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg"; // URL ảnh mặc định

const NewsGuestPage = () => {
    // Bỏ state liên quan đến Auth, Edit, Delete
    // const { session, isLoading: isAuthLoading, token, isLoggedIn } = useAuth();
    const [articles, setArticles] = useState<Article[]>([]);
    const [filteredArticles, setFilteredArticles] = useState<Article[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState<string>("");
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [totalArticles, setTotalArticles] = useState(0);

    // Bỏ state cho Form Modal và Delete Dialog

    // Hàm fetch dữ liệu - Không cần token
    const fetchArticles = useCallback(async (page: number = 1, query: string = searchQuery) => {
        setIsLoading(true);
        setError(null);
        try {
            const options: QueryOptions = {
                limit: ITEMS_PER_PAGE,
                offset: (page - 1) * ITEMS_PER_PAGE,
                sortBy: 'created_at',
                sortDesc: true,
            };
            console.log("Guest: Fetching articles with options:", options);
            // Gọi API không cần token
            const fetchedArticles = await getAllArticles(options);
            console.log("Guest: Fetched articles:", fetchedArticles);

            setArticles(fetchedArticles);

            // Lọc phía client nếu cần (hoặc nếu API hỗ trợ search thì tốt hơn)
            const filtered = fetchedArticles.filter(item =>
                item.title.toLowerCase().includes(query.toLowerCase())
            );
            setFilteredArticles(filtered);

            // Ước tính totalPages (vẫn giữ nguyên cách ước tính thô sơ)
            const estimatedTotal = (page - 1) * ITEMS_PER_PAGE + fetchedArticles.length;
            const potentialTotal = (fetchedArticles.length === ITEMS_PER_PAGE) ? estimatedTotal + 1 : estimatedTotal;
            setTotalArticles(potentialTotal)

            setCurrentPage(page);

        } catch (err: any) {
            console.error("Guest: Failed to fetch articles:", err);
            setError("Không thể tải danh sách bản tin. Vui lòng thử lại.");
            // toast.error("Lỗi tải bản tin: " + err.message); // Bỏ toast
            setArticles([]);
            setFilteredArticles([]);
        } finally {
            setIsLoading(false);
        }
    }, [searchQuery]); // Chỉ phụ thuộc searchQuery

    // Fetch lần đầu khi component mount hoặc searchQuery thay đổi
    useEffect(() => {
       // Fetch ngay không cần chờ auth
       fetchArticles(1, searchQuery);
    }, [fetchArticles, searchQuery]); // Bỏ isAuthLoading

    const handleSearch = (query: string) => {
        setSearchQuery(query);
    };

    const handlePageChange = (page: number) => {
        fetchArticles(page, searchQuery);
    };

    // Bỏ các hàm openModal, handleFormSuccess, openDeleteDialog, confirmDelete

    const totalPages = Math.ceil(totalArticles / ITEMS_PER_PAGE);

    // Không cần isAuthLoading nữa
    if (isLoading) return <PageLoader message="Đang tải danh sách bản tin..." />;
    if (error && articles.length === 0) return <div className="container mx-auto px-4 py-6 text-red-600">{error}</div>;

    return (
        <div className="container mx-auto px-4 py-6">
            {/* <Toaster position="top-right" /> */} {/* Bỏ Toaster */}
            <header className="flex flex-col sm:flex-row items-center justify-between mb-6 gap-4">
                <h1 className="text-3xl font-bold text-orange-500">Bản tin</h1> {/* Đổi tiêu đề */}
                <div className="flex items-center space-x-2 w-full sm:w-auto">
                     {/* Bỏ nút Đăng bài mới */}
                    <div className="relative flex-grow sm:flex-grow-0">
                        <Input
                            type="text"
                            placeholder="Tìm theo tiêu đề..." // Đổi placeholder
                            value={searchQuery}
                            onChange={(e) => handleSearch(e.target.value)}
                            className="pl-10 w-full"
                        />
                        <FilterIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                    </div>
                </div>
            </header>

            {filteredArticles.length === 0 && !isLoading && (
                <div className="text-center text-gray-500 mt-10">Không tìm thấy bản tin nào phù hợp.</div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredArticles.map((item) => (
                    <Card key={item.id} className="hover:shadow-lg transition-shadow flex flex-col h-full overflow-hidden relative group">
                        {/* Link tới trang chi tiết guest */}
                        <Link href={`/newsguest/${item.id}`} className="block h-full">
                            {/* Image */}
                            <div className="relative w-full h-48 sm:h-56">
                                <Image
                                    // Sử dụng generateProxyUrl nếu cần, hoặc URL trực tiếp nếu ảnh public
                                    src={generateProxyUrl(item.pictureUrl) || DEFAULT_IMAGE_URL}
                                    alt={item.title}
                                    fill
                                    style={{ objectFit: "cover" }}
                                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                                    className="rounded-t-md"
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
                                <CardTitle className="text-lg font-semibold line-clamp-2">{item.title}</CardTitle>
                            </CardHeader>
                            <CardContent className="flex-grow flex flex-col justify-between pt-0">
                                <p className="text-sm text-gray-600 mb-2 line-clamp-3">{item.content}</p>
                                <p className="text-xs text-gray-500 mt-auto">
                                    {new Date(item.createdAt).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                                </p>
                            </CardContent>
                        </Link>
                        {/* Bỏ các nút Edit/Delete ở đây */}
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

             {/* Bỏ các Dialog Form và Delete */}

        </div>
    );
};

export default NewsGuestPage; // Đổi tên export