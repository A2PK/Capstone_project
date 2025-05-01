"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Pagination } from "@/components/pagination";
import { useEffect, useState, useCallback } from "react";
import { FilterIcon, Loader2, Tag } from "lucide-react"; 
import Link from "next/link";
import PageLoader from "@/components/pageloader";
import Image from "next/image";
import type { Article } from "@/types/article"; 
import { getAllArticles, generateProxyUrl } from "@/lib/article";
import type { QueryOptions } from "@/types/station2"; 
import { cn } from "@/lib/utils"; 
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
const ITEMS_PER_PAGE = 6;
const DEFAULT_IMAGE_URL = "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg";
type BadgeFilter = 'all' | 'danger' | 'good' | 'common';
const filterOptions: { label: string; value: BadgeFilter }[] = [
    { label: "Tất cả mức độ", value: "all" }, // Thay đổi label cho rõ ràng hơn trong dropdown
    { label: "Khẩn cấp", value: "danger" },
    { label: "Quan trọng", value: "good" },
    { label: "Thông thường", value: "common" },
];
const NewsGuestPage = () => {
    const [articles, setArticles] = useState<Article[]>([]); 
    const [filteredArticles, setFilteredArticles] = useState<Article[]>([]); 
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState<string>("");
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [totalArticles, setTotalArticles] = useState(0); 
    const [selectedBadge, setSelectedBadge] = useState<BadgeFilter>('all'); 

    const fetchArticles = useCallback(async (page: number = 1, query: string = searchQuery, badge: BadgeFilter = selectedBadge) => {
        setIsLoading(true);
        setError(null);
        try {
            const options: QueryOptions = {
                limit: ITEMS_PER_PAGE,
                offset: (page - 1) * ITEMS_PER_PAGE,
                sortBy: 'created_at',
                sortDesc: true,

            };
            //console.log("Guest: Fetching articles with options:", options);
            const fetchedArticles = await getAllArticles(options) || []; 
            //console.log("Guest: Fetched articles:", fetchedArticles);
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

            setFilteredArticles(tempFiltered); 
            const estimatedTotal = (page - 1) * ITEMS_PER_PAGE + fetchedArticles.length;
            const potentialTotal = (fetchedArticles.length === ITEMS_PER_PAGE) ? estimatedTotal + 1 : estimatedTotal;
            setTotalArticles(potentialTotal);
            // setCurrentPage(page); // Cập nhật trang hiện tại (di chuyển vào useEffect hoặc handlePageChange)

        } catch (err: any) {
            console.error("Guest: Failed to fetch articles:", err);
            setError("Không thể tải danh sách bản tin. Vui lòng thử lại.");
            setArticles([]);
            setFilteredArticles([]);
        } finally {
            setIsLoading(false);
        }
    }, [searchQuery, selectedBadge]); // <--- Thêm selectedBadge vào dependencies

    // --- Effect để fetch khi trang, tìm kiếm hoặc bộ lọc thay đổi ---
    useEffect(() => {
        fetchArticles(currentPage, searchQuery, selectedBadge);
    }, [fetchArticles, currentPage, searchQuery, selectedBadge]); 

    // --- Event Handlers ---
    const handleSearch = (query: string) => {
        setSearchQuery(query);
        setCurrentPage(1); // Reset về trang 1 khi tìm kiếm
    };

    const handleBadgeFilterChange = (badge: BadgeFilter) => {
        setSelectedBadge(badge);
        setCurrentPage(1); // Reset về trang 1 khi đổi bộ lọc
    };

    const handlePageChange = (page: number) => {
        setCurrentPage(page);
    };

    const totalPages = Math.ceil(totalArticles / ITEMS_PER_PAGE);

    // --- Render Logic ---
    if (isLoading && filteredArticles.length === 0) return <PageLoader message="Đang tải danh sách bản tin..." />; 
    if (error && articles.length === 0) return <div className="container mx-auto px-4 py-6 text-red-600">{error}</div>;


    return (
        <div className="container mx-auto px-4 py-6">
            <header className="mb-6">
                <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                    {/* Title */}
                    <h1 className="text-3xl font-bold text-orange-500 shrink-0">Bản tin</h1>

                    {/* Filters Container */}
                    <div className="flex flex-col sm:flex-row items-center gap-3 w-full md:w-auto">
                         {/* Badge Filter Dropdown */}
                         <Select
                            value={selectedBadge}
                            // Sử dụng type assertion vì onValueChange trả về string
                            onValueChange={(value) => handleBadgeFilterChange(value as BadgeFilter)}
                        >
                            <SelectTrigger className="w-full sm:w-[180px]"> {/* Điều chỉnh độ rộng */}
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
                        <div className="relative w-full sm:w-auto flex-grow md:flex-grow-0"> {/* Điều chỉnh flex-grow */}
                            <Input
                                type="text"
                                placeholder="Tìm theo tiêu đề..."
                                value={searchQuery}
                                onChange={(e) => handleSearch(e.target.value)}
                                className="pl-10 pr-4 py-2 w-full sm:w-64" // Thêm padding, điều chỉnh width
                            />
                            <FilterIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" /> {/* Thêm pointer-events-none */}
                        </div>
                    </div>
                </div>
            </header>
            {isLoading && (
                <div className="flex justify-center my-4">
                    <Loader2 className="w-6 h-6 animate-spin text-orange-500" />
                </div>
            )}

            {/* Hiển thị khi không có kết quả */}
            {!isLoading && filteredArticles.length === 0 && (
                <div className="text-center text-gray-500 mt-10">Không tìm thấy bản tin nào phù hợp với bộ lọc đã chọn.</div>
            )}

            {/* Danh sách bài viết */}
            {!isLoading && filteredArticles.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredArticles.map((item) => (
                        <Card key={item.id} className="hover:shadow-lg transition-shadow flex flex-col h-full overflow-hidden relative group bg-white dark:bg-gray-800"> {/* Thêm màu nền */}
                            <Link href={`/newsguest/${item.id}`} className="block h-full">
                                <div className="relative w-full h-48 sm:h-56">
                                    <Image
                                        src={generateProxyUrl(item.pictureUrl) || DEFAULT_IMAGE_URL}
                                        alt={item.title}
                                        fill
                                        style={{ objectFit: "cover" }}
                                        sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw" // Cập nhật sizes
                                        className="rounded-t-md"
                                        onError={(e) => { (e.target as HTMLImageElement).src = DEFAULT_IMAGE_URL; }}
                                    />
                                    {/* Hiển thị Badge nếu có */}
                                    {item.badge && (
                                        <span className={`absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white ${
                                            item.badge === 'danger' ? 'bg-red-500' : item.badge === 'good' ? 'bg-green-500' : 'bg-gray-500' // Mặc định cho các badge khác (nếu có)
                                        }`}>
                                            {item.badge === 'danger' ? 'Khẩn cấp' : item.badge === 'good' ? 'Tốt' : item.badge} {/* Hiển thị tên badge gốc nếu không phải danger/good */}
                                        </span>
                                    )}
                                     {/* Hiển thị Badge "Thông thường" nếu không có badge hoặc không phải danger/good */}
                                     {!item.badge || (item.badge !== 'danger' && item.badge !== 'good') && (
                                          <span className="absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white bg-gray-500">
                                                Thông thường
                                          </span>
                                      )}
                                </div>
                                <CardHeader className="pb-2 pt-4">
                                    <CardTitle className="text-lg font-semibold line-clamp-2 text-gray-900 dark:text-gray-100">{item.title}</CardTitle>
                                </CardHeader>
                                <CardContent className="flex-grow flex flex-col justify-between pt-0">
                                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2 line-clamp-3">{item.content?.substring(0, 100) + (item.content && item.content.length > 100 ? '...' : '')}</p>
                                    <p className="text-xs text-gray-500 dark:text-gray-500 mt-auto">
                                        {new Date(item.createdAt).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                                    </p>
                                </CardContent>
                            </Link>
                        </Card>
                    ))}
                </div>
            )}

            {/* Phân trang */}
            {!isLoading && totalPages > 1 && filteredArticles.length > 0 && ( 
                <footer className="mt-8 flex justify-center">
                    <Pagination
                        currentPage={currentPage}
                        totalPages={totalPages}
                        onPageChange={handlePageChange}
                    />
                </footer>
            )}
        </div>
    );
};

export default NewsGuestPage;