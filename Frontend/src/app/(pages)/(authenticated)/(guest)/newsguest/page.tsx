"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Pagination } from "@/components/pagination";
import { useEffect, useState, useMemo } from "react"; // useMemo có thể hữu ích
import { FilterIcon, Loader2, Tag } from "lucide-react";
import Link from "next/link";
import PageLoader from "@/components/pageloader";
import Image from "next/image";
import type { Article } from "@/types/article";
import { getAllArticles, generateProxyUrl } from "@/lib/article";
// Bỏ QueryOptions vì không dùng limit/offset nữa
// import type { QueryOptions } from "@/types/station2";
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
    { label: "Tất cả mức độ", value: "all" },
    { label: "Khẩn cấp", value: "danger" },
    { label: "Quan trọng", value: "good" },
    { label: "Thông thường", value: "common" },
];

const NewsGuestPage = () => {
    // --- State ---
    const [allArticles, setAllArticles] = useState<Article[]>([]); 
    const [filteredArticles, setFilteredArticles] = useState<Article[]>([]); // Bài viết sau khi lọc
    const [displayedArticles, setDisplayedArticles] = useState<Article[]>([]); // Bài viết hiển thị trên trang hiện tại
    const [isLoading, setIsLoading] = useState(true); // Chỉ loading lúc đầu tải tất cả
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState<string>("");
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [selectedBadge, setSelectedBadge] = useState<BadgeFilter>('all');

    useEffect(() => {
        const fetchAllData = async () => {
            setIsLoading(true);
            setError(null);
            try {
                console.log("Guest: Fetching ALL articles...");
                const fetchedData = await getAllArticles();
                console.log(`Guest: Fetched ${fetchedData?.length || 0} total articles.`);
                setAllArticles(fetchedData || []);
            } catch (err: any) {
                console.error("Guest: Failed to fetch all articles:", err);
                setError("Không thể tải danh sách bản tin. Vui lòng thử lại.");
                setAllArticles([]); // Đặt thành mảng rỗng khi lỗi
            } finally {
                setIsLoading(false);
            }
        };

        fetchAllData();
    }, []); // Mảng dependency rỗng đảm bảo chỉ chạy 1 lần

    // Lọc danh sách bài viết mỗi khi `allArticles` hoặc bộ lọc thay đổi
    useEffect(() => {
        console.log("Guest: Applying filters...");
        let tempFiltered = [...allArticles]; // Bắt đầu với tất cả bài viết đã tải

        // Lọc theo searchQuery
        if (searchQuery) {
            tempFiltered = tempFiltered.filter(item =>
                item.title.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        // Lọc theo selectedBadge
        if (selectedBadge !== 'all') {
            tempFiltered = tempFiltered.filter(item => {
                if (selectedBadge === 'common') {
                    // Bài viết là 'common' nếu không có badge hoặc badge không phải 'danger'/'good'
                    return !item.badge || (item.badge !== 'danger' && item.badge !== 'good');
                }
                // Trường hợp 'danger' hoặc 'good'
                return item.badge === selectedBadge;
            });
        }

        console.log(`Guest: Filtering complete. ${tempFiltered.length} articles match.`);
        setFilteredArticles(tempFiltered);
        setCurrentPage(1); // Reset về trang 1 mỗi khi bộ lọc thay đổi
    }, [allArticles, searchQuery, selectedBadge]); // Chạy lại khi dữ liệu gốc hoặc bộ lọc thay đổi

    // Phân trang danh sách đã lọc mỗi khi `filteredArticles` hoặc `currentPage` thay đổi 
    useEffect(() => {
        const totalFilteredCount = filteredArticles.length;
        const totalPagesCalc = Math.ceil(totalFilteredCount / ITEMS_PER_PAGE);
        console.log(`Guest: Paginating ${totalFilteredCount} filtered articles for page ${currentPage}. Total pages: ${totalPagesCalc}`);

        // Đảm bảo currentPage không vượt quá tổng số trang thực tế (sau khi lọc)
        let pageToShow = currentPage;
        if (pageToShow > totalPagesCalc && totalPagesCalc > 0) {
             console.log(`Guest: Current page ${pageToShow} exceeds total pages ${totalPagesCalc}. Resetting to ${totalPagesCalc}.`);
             pageToShow = totalPagesCalc;
             setCurrentPage(totalPagesCalc); // Cập nhật lại state trang hiện tại
        } else if (pageToShow < 1) {
             pageToShow = 1;
             setCurrentPage(1);
        }


        const startIndex = (pageToShow - 1) * ITEMS_PER_PAGE;
        const endIndex = startIndex + ITEMS_PER_PAGE;
        const articlesForPage = filteredArticles.slice(startIndex, endIndex);

        console.log(`Guest: Displaying articles from index ${startIndex} to ${endIndex-1}. Count: ${articlesForPage.length}`);
        setDisplayedArticles(articlesForPage);

    }, [filteredArticles, currentPage]); // Chạy lại khi danh sách đã lọc hoặc trang hiện tại thay đổi

    const handleSearch = (query: string) => {
        setSearchQuery(query);
    };

    const handleBadgeFilterChange = (badge: BadgeFilter) => {
        setSelectedBadge(badge);
        // Không cần setCurrentPage(1) ở đây, effect [allArticles, searchQuery, selectedBadge] sẽ xử lý
    };

    const handlePageChange = (page: number) => {
        console.log("Guest: Changing page to", page);
        setCurrentPage(page);
    };

    // Tính toán tổng số trang DỰA TRÊN danh sách ĐÃ LỌC
    const totalPages = Math.ceil(filteredArticles.length / ITEMS_PER_PAGE);
    if (isLoading) return <PageLoader message="Đang tải danh sách bản tin..." />;
    if (error) return <div className="container mx-auto px-4 py-6 text-red-600">{error}</div>;
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
                            onValueChange={(value) => handleBadgeFilterChange(value as BadgeFilter)}
                        >
                            <SelectTrigger className="w-full sm:w-[180px]">
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
                        <div className="relative w-full sm:w-auto flex-grow md:flex-grow-0">
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

            {/* Không cần loading indicator ở đây nữa vì đã xử lý ở trên */}

            {/* Hiển thị khi không có kết quả (sau khi đã tải xong và lọc) */}
            {/* Kiểm tra filteredArticles thay vì displayedArticles */}
            {filteredArticles.length === 0 && (
                <div className="text-center text-gray-500 mt-10">Không tìm thấy bản tin nào phù hợp với bộ lọc đã chọn.</div>
            )}

            {/* Danh sách bài viết (chỉ hiển thị nếu có bài viết sau khi lọc) */}
            {filteredArticles.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {displayedArticles.map((item) => (
                        <Card key={item.id} className="hover:shadow-lg transition-shadow flex flex-col h-full overflow-hidden relative group bg-white dark:bg-gray-800">
                             <Link href={`/newsguest/${item.id}`} className="block h-full">
                                <div className="relative w-full h-48 sm:h-56">
                                    <Image
                                        src={generateProxyUrl(item.pictureUrl) || DEFAULT_IMAGE_URL}
                                        alt={item.title}
                                        fill
                                        style={{ objectFit: "cover" }}
                                        sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                                        className="rounded-t-md"
                                        onError={(e) => { (e.target as HTMLImageElement).src = DEFAULT_IMAGE_URL; }}
                                    />
                                    {item.badge === 'danger' && (
                                        <span className="absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white bg-red-500">Khẩn cấp</span>
                                    )}
                                    {item.badge === 'good' && (
                                        <span className="absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white bg-green-500">Quan trọng</span>
                                    )}
                                    {/* Hiển thị "Thông thường" nếu không phải danger/good hoặc không có badge */}
                                    {(!item.badge || (item.badge !== 'danger' && item.badge !== 'good')) && (
                                         <span className="absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded text-white bg-gray-500">Thông thường</span>
                                    )}
                                </div>
                                <CardHeader className="pb-2 pt-4">
                                    <CardTitle className="text-lg font-semibold line-clamp-2 text-gray-900 dark:text-gray-100">{item.title}</CardTitle>
                                </CardHeader>
                                <CardContent className="flex-grow flex flex-col justify-between pt-0">
                                    {/* Hiển thị nội dung ngắn gọn */}
                                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2 line-clamp-3">{item.content ? item.content.substring(0, 100) + (item.content.length > 100 ? '...' : '') : ''}</p>
                                    {/* Hiển thị ngày tạo */}
                                    <p className="text-xs text-gray-500 dark:text-gray-500 mt-auto">
                                        {new Date(item.createdAt).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                                    </p>
                                </CardContent>
                            </Link>
                        </Card>
                    ))}
                </div>
            )}

            {/* Phân trang (chỉ hiển thị nếu có nhiều hơn 1 trang) */}
            {totalPages > 1 && (
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