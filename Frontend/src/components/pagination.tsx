import React, { useState, useMemo, ChangeEvent, KeyboardEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input"; // Import Input component
import { ChevronLeft, ChevronRight } from "lucide-react"; // Icons for Prev/Next

// Props Interface - Thêm siblingCount tùy chọn
interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  siblingCount?: number; // Số trang hiển thị bên cạnh trang hiện tại
}

// Hằng số cho dấu ba chấm
const DOTS = '...';

// Helper function tạo range số
const range = (start: number, end: number): number[] => {
  const length = end - start + 1;
  return Array.from({ length }, (_, idx) => idx + start);
};

export const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
  siblingCount = 1, // Mặc định là 1 sibling
}) => {
  const [goToPage, setGoToPage] = useState<string>('');

  // --- Logic tạo dải số trang hiển thị ---
  const paginationRange = useMemo((): (number | string)[] => {
    // Tổng số trang hiển thị bao gồm: FirstPage + LastPage + CurrentPage + siblings + 2*DOTS
    const totalPageNumbers = siblingCount * 2 + 5;

    /*
      Case 1: Nếu tổng số trang nhỏ hơn số trang chúng ta muốn hiển thị -> hiển thị tất cả
    */
    if (totalPageNumbers >= totalPages) {
      return range(1, totalPages);
    }

    const leftSiblingIndex = Math.max(currentPage - siblingCount, 1);
    const rightSiblingIndex = Math.min(currentPage + siblingCount, totalPages);

    /*
      Chúng ta không hiển thị dấu chấm khi chỉ có 1 số trang bị ẩn sau trang đầu tiên/trước trang cuối.
      Tức là: shouldShowLeftDots khi leftSiblingIndex > 2
              shouldShowRightDots khi rightSiblingIndex < totalPages - 1
    */
    const shouldShowLeftDots = leftSiblingIndex > 2;
    const shouldShowRightDots = rightSiblingIndex < totalPages - 1;

    const firstPageIndex = 1;
    const lastPageIndex = totalPages;

    /*
      Case 2: Không có dấu chấm trái, chỉ có dấu chấm phải
    */
    if (!shouldShowLeftDots && shouldShowRightDots) {
      let leftItemCount = 3 + 2 * siblingCount; // 1 (first) + siblings + current + 1 (trước dots)
      let leftRange = range(1, leftItemCount);
      return [...leftRange, DOTS, lastPageIndex];
    }

    /*
      Case 3: Không có dấu chấm phải, chỉ có dấu chấm trái
    */
    if (shouldShowLeftDots && !shouldShowRightDots) {
      let rightItemCount = 3 + 2 * siblingCount; // 1 (last) + siblings + current + 1 (sau dots)
      let rightRange = range(totalPages - rightItemCount + 1, totalPages);
      return [firstPageIndex, DOTS, ...rightRange];
    }

    /*
      Case 4: Có cả hai dấu chấm
    */
    if (shouldShowLeftDots && shouldShowRightDots) {
      let middleRange = range(leftSiblingIndex, rightSiblingIndex);
      return [firstPageIndex, DOTS, ...middleRange, DOTS, lastPageIndex];
    }

    // Trường hợp không rơi vào các case trên (thường không xảy ra với logic hiện tại)
    return range(1, totalPages);

  }, [totalPages, siblingCount, currentPage]);


  // --- Handlers ---
  const handlePrev = () => {
    if (currentPage > 1) onPageChange(currentPage - 1);
  };

  const handleNext = () => {
    if (currentPage < totalPages) onPageChange(currentPage + 1);
  };

  const handleGoToInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setGoToPage(e.target.value);
  };

  const handleGoToPage = () => {
    const pageNum = parseInt(goToPage, 10);
    if (!isNaN(pageNum) && pageNum >= 1 && pageNum <= totalPages) {
      onPageChange(pageNum);
      setGoToPage(''); // Clear input sau khi nhảy trang
    } else {
      // Có thể thêm toast thông báo lỗi ở đây
      console.warn("Invalid page number entered");
      setGoToPage(''); // Clear input nếu lỗi
    }
  };

  const handleGoToPageKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleGoToPage();
    }
  };


  // --- Render ---
  // Nếu chỉ có 1 trang hoặc không có trang nào, không hiển thị pagination
  if (totalPages <= 1) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-center justify-center gap-1 sm:gap-2"> {/* Sử dụng gap thay space-x */}
      {/* Previous Button */}
      <Button
        onClick={handlePrev}
        disabled={currentPage === 1}
        variant="outline"
        size="sm" // Kích thước nhỏ hơn
        className="px-2" // Padding ngang nhỏ
      >
        <ChevronLeft className="h-4 w-4" />
        <span className="hidden sm:inline ml-1">Prev</span> {/* Ẩn chữ trên màn hình nhỏ */}
      </Button>

      {/* Page Numbers and Ellipses */}
      {paginationRange.map((pageNumber, index) => {
        // Nếu là dấu chấm
        if (pageNumber === DOTS) {
          return (
            <span key={`${DOTS}-${index}`} className="px-2 py-1 text-gray-500">
              &#8230; {/* HTML entity for ellipsis */}
            </span>
          );
        }

        // Nếu là số trang
        return (
          <Button
            key={pageNumber}
            onClick={() => onPageChange(pageNumber as number)}
            variant={pageNumber === currentPage ? "default" : "outline"}
            size="sm" // Kích thước nhỏ hơn
            className="h-8 w-8 p-0 sm:h-9 sm:w-9" // Kích thước cố định nhỏ hơn
          >
            {pageNumber}
          </Button>
        );
      })}

      {/* Next Button */}
      <Button
        onClick={handleNext}
        disabled={currentPage === totalPages}
        variant="outline"
        size="sm" // Kích thước nhỏ hơn
        className="px-2" // Padding ngang nhỏ
      >
        <span className="hidden sm:inline mr-1">Next</span> {/* Ẩn chữ trên màn hình nhỏ */}
        <ChevronRight className="h-4 w-4" />
      </Button>

       {/* Go to Page Input (Optional) - Chỉ hiển thị nếu nhiều hơn ~7 trang */}
        {totalPages > 7 && (
             <div className="flex items-center gap-1 ml-2">
                 <Input
                     type="number"
                     min="1"
                     max={totalPages}
                     value={goToPage}
                     onChange={handleGoToInputChange}
                     onKeyDown={handleGoToPageKeyDown}
                     className="h-8 w-14 text-center px-1 text-sm [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" // Style input number
                     placeholder="Trang"
                 />
                 <Button
                     onClick={handleGoToPage}
                     variant="outline"
                     size="sm"
                     className="h-8 px-2 text-xs" // Nút nhỏ hơn
                 >
                     Đi đến
                 </Button>
             </div>
        )}
    </div>
  );
};