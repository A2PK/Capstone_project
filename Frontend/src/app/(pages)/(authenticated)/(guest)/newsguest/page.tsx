"use client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Pagination } from "@/components/pagination";
import { useEffect, useState } from "react";
import { FilterIcon } from "lucide-react";
import Link from "next/link";
interface NewsItem {
  title: string;
  description: string;
  date: string;
  imageUrl: string;
}

const ITEMS_PER_PAGE = 6;

const NewsPage = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [filteredNews, setFilteredNews] = useState<NewsItem[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const fetchNews = async () => {
    const data: NewsItem[] = [
      {
        title: "Bản tin khẩn chất lượng nước sông Tiền !!",
        description: "Dữ liệu quan trắc được từ trạm sông Tiền cho thấy chất lượng nước nơi này đang có sự ô nhiễm nhẹ...",
        date: "4 Nov 2024",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
      },
      {
        title: "Báo cáo hàng tuần về chất lượng nước sông Hậu!!",
        description: "Dữ liệu quan trắc được từ trạm sông Hồng cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
        date: "4 Nov 2024",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
      },
      {
        title: "Bản tin khẩn chất lượng nước sông Tiền !!",
        description: "Dữ liệu quan trắc được từ trạm sông Tiền cho thấy chất lượng nước nơi này đang có sự ô nhiễm nhẹ...",
        date: "4 Nov 2024",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
      },
      {
        title: "Báo cáo hàng tuần về chất lượng nước sông Hậu!!",
        description: "Dữ liệu quan trắc được từ trạm sông Hồng cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
        date: "4 Nov 2024",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
      },
      {
        title: "Bản tin khẩn chất lượng nước sông Tiền !!",
        description: "Dữ liệu quan trắc được từ trạm sông Tiền cho thấy chất lượng nước nơi này đang có sự ô nhiễm nhẹ...",
        date: "4 Nov 2024",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
      },
      {
        title: "Báo cáo hàng tuần về chất lượng nước sông Hậu!!",
        description: "Dữ liệu quan trắc được từ trạm sông Hồng cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
        date: "4 Nov 2024",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
      },
      {
        title: "Bản tin khẩn chất lượng nước sông Tiền !!",
        description: "Dữ liệu quan trắc được từ trạm sông Tiền cho thấy chất lượng nước nơi này đang có sự ô nhiễm nhẹ...",
        date: "4 Nov 2024",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
      },
      {
        title: "Báo cáo hàng tuần về chất lượng nước sông Hậu!!",
        description: "Dữ liệu quan trắc được từ trạm sông Hồng cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
        date: "4 Nov 2024",
        imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
      },


    ];

    setNews(data);
    setFilteredNews(data);
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    const filtered = news.filter((item) =>
      item.title.toLowerCase().includes(query.toLowerCase())
    );
    setFilteredNews(filtered);
    setCurrentPage(1); // Reset to the first page on search
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  // Calculate total pages based on filtered news and items per page
  const totalPages = Math.ceil(filteredNews.length / ITEMS_PER_PAGE);

  // Get the news items for the current page
  const displayedNews = filteredNews.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  useEffect(() => {
    fetchNews();
  }, []);

  return (
    <div className="container mx-auto px-4 py-6">
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-orange-500">Bản tin</h1>
        <div className="flex items-center space-x-2">
          <div className="relative">
            <Input
              type="text"
              placeholder="Lọc"
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              className="pl-10"
            />
            <FilterIcon className="absolute left-2 top-2.5 w-5 h-5 text-gray-500" />
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {displayedNews.map((item, index) => (
          <Link key={index} href="/newsguestdetail" className="block h-full">
            <Card className="hover:shadow-lg transition-shadow flex flex-col h-full">
              <img
                src={item.imageUrl}
                alt={item.title}
                className="w-full h-64 object-cover rounded-t-md"
              />
              <CardHeader className="flex-grow">
                <CardTitle className="text-xl font-bold">{item.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-2">{item.description}</p>
                <p className="text-xs text-gray-500">{item.date}</p>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>


      {totalPages > 1 && (
        <footer className="mt-6 flex justify-center">
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

export default NewsPage;
