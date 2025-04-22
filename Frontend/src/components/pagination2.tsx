import React, { useMemo } from 'react'; // Import useMemo để tối ưu hiệu suất
import { Button } from "@/components/ui/button";

// --- Logic tính toán được đưa vào đây ---
const DOTS = '...'; // Định nghĩa DOTS trực tiếp

const range = (start: number, end: number): number[] => {
  let length = end - start + 1;
  /*
  	Tạo một mảng số nguyên từ start đến end
  */
  return Array.from({ length }, (_, idx) => idx + start);
};
// --- Kết thúc phần logic tính toán ---


interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  siblingCount?: number; // Optional: Số trang lân cận hiển thị ở mỗi bên trang hiện tại
}

export const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
  siblingCount = 1, // Mặc định hiển thị 1 trang lân cận ở mỗi bên (prev, current, next)
}) => {

  // --- Tính toán paginationRange dựa trên currentPage và siblingCount ---
  const paginationRange = useMemo((): (number | string)[] => {
    const totalPageCount = totalPages;

    // Số trang hiển thị = siblingCount + firstPage + lastPage + currentPage + 2*DOTS
    // (Công thức này dùng để xác định khi nào cần hiển thị rút gọn)
    const totalPageNumbers = siblingCount + 5;

    /*
      Trường hợp 1: Nếu số trang nhỏ hơn hoặc bằng số trang tối thiểu cần hiển thị (bao gồm DOTS)
      => Hiển thị tất cả các trang. Không cần rút gọn.
    */
    if (totalPageNumbers >= totalPageCount) {
      return range(1, totalPageCount);
    }

    /*
    	Tính toán chỉ số trang lân cận bên trái và phải của trang hiện tại
    */
    const leftSiblingIndex = Math.max(currentPage - siblingCount, 1);
    const rightSiblingIndex = Math.min(
      currentPage + siblingCount,
      totalPageCount
    );

    /*
      Quyết định xem có nên hiển thị dấu chấm bên trái/phải hay không.
      Không hiển thị dấu chấm nếu chỉ có 1 trang bị ẩn giữa các đầu mút.
      Ví dụ: 1 ... 3 4 5 => sai, nên là 1 2 3 4 5
    */
    const shouldShowLeftDots = leftSiblingIndex > 2;
    const shouldShowRightDots = rightSiblingIndex < totalPageCount - 1;

    const firstPageIndex = 1;
    const lastPageIndex = totalPageCount;

    /*
    	Trường hợp 2: Chỉ hiển thị dấu chấm bên phải
      (Khi trang hiện tại gần đầu danh sách)
      VD: 1 2 3 ... 10 (currentPage=1, 2 hoặc 3 với siblingCount=1)
    */
    if (!shouldShowLeftDots && shouldShowRightDots) {
      // Số lượng trang hiển thị bên trái = 1(first) + siblingCount + 1(current) + siblingCount
       let leftItemCount = 1 + 2 * siblingCount + (currentPage - leftSiblingIndex) + 1; // Điều chỉnh để linh hoạt hơn
       leftItemCount = Math.min(siblingCount + 3, totalPageCount); // Cách tính đơn giản hơn: hiển thị 1, 2, 3, 4, 5...
       let leftRange = range(1, leftItemCount);

      return [...leftRange, DOTS, lastPageIndex];
    }

    /*
    	Trường hợp 3: Chỉ hiển thị dấu chấm bên trái
      (Khi trang hiện tại gần cuối danh sách)
       VD: 1 ... 8 9 10 (currentPage=8, 9 hoặc 10 với siblingCount=1)
    */
    if (shouldShowLeftDots && !shouldShowRightDots) {
       // Số lượng trang hiển thị bên phải = 1(last) + siblingCount + 1(current) + siblingCount
       let rightItemCount = 1 + 2 * siblingCount + (rightSiblingIndex - currentPage) + 1; // Điều chỉnh
       rightItemCount = Math.min(siblingCount + 3, totalPageCount); // Cách tính đơn giản hơn
       let rightRange = range(
        totalPageCount - rightItemCount + 1,
        totalPageCount
      );
      return [firstPageIndex, DOTS, ...rightRange];
    }

    /*
    	Trường hợp 4: Hiển thị dấu chấm cả hai bên
      (Khi trang hiện tại nằm ở giữa)
      VD: 1 ... 4 5 6 ... 10 (currentPage=5 với siblingCount=1)
    */
    if (shouldShowLeftDots && shouldShowRightDots) {
      let middleRange = range(leftSiblingIndex, rightSiblingIndex);
      return [firstPageIndex, DOTS, ...middleRange, DOTS, lastPageIndex];
    }

    // Mặc định (phòng trường hợp, logic trên nên bao phủ hết)
     return range(1, totalPageCount);

  }, [currentPage, totalPages, siblingCount]); // Phụ thuộc vào cả 3 giá trị này
  // --- Kết thúc tính toán paginationRange ---


  // --- Phần Render ---
  const handlePrev = () => {
    if (currentPage > 1) onPageChange(currentPage - 1);
  };

  const handleNext = () => {
    if (currentPage < totalPages) onPageChange(currentPage + 1);
  };

  // Nếu không có trang nào hoặc chỉ có 1 trang
   if (totalPages <= 1) {
     return null;
   }

  return (
    <div className="flex items-center justify-center space-x-1 sm:space-x-2">
      {/* Nút Prev */}
      <Button
        onClick={handlePrev}
        disabled={currentPage === 1}
        variant="outline"
        size="sm"
        aria-label="Go to previous page"
      >
        Prev
      </Button>

      {/* Các nút số trang và dấu ... */}
      {paginationRange.map((pageNumber, index) => {
        const key = typeof pageNumber === 'string' ? `dots-${index}` : pageNumber;

        // Nếu là dấu ...
        if (pageNumber === DOTS) {
          return (
            <span key={key} className="px-2 py-1 text-gray-500 dark:text-gray-400">
              &#8230;
            </span>
          );
        }

        // Nếu là số trang
        return (
          <Button
            key={key}
            onClick={() => onPageChange(pageNumber as number)}
            variant={pageNumber === currentPage ? "default" : "outline"}
            size="sm"
            disabled={pageNumber === currentPage} // Disable nút đang được chọn
            aria-current={pageNumber === currentPage ? 'page' : undefined}
            aria-label={`Go to page ${pageNumber}`}
          >
            {pageNumber}
          </Button>
        );
      })}

      {/* Nút Next */}
      <Button
        onClick={handleNext}
        disabled={currentPage === totalPages}
        variant="outline"
        size="sm"
        aria-label="Go to next page"
      >
        Next
      </Button>
    </div>
  );
};