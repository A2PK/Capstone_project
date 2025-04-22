// src/app/(pages)/api/auth/logout/route.ts
// HOẶC TỐT HƠN: src/app/api/auth/logout/route.ts (Xem lưu ý về đường dẫn ở dưới)

import { type NextRequest, NextResponse } from 'next/server';

// Export hàm xử lý cho phương thức POST
export async function POST(req: NextRequest) {
    console.log("App Route Handler: /api/auth/logout - Clearing cookies...");

    try {
        // Tạo response để có thể set/delete cookie
        const response = NextResponse.json({ message: 'Logout successful' }, { status: 200 });

        // --- Cách 1: Dùng delete (Khuyến nghị) ---
        // Xóa cookies bằng cách sử dụng phương thức delete() của response.cookies
        // Mặc định Next.js sẽ dùng path='/', đảm bảo khớp với lúc bạn đặt cookie.
        // Nếu bạn đặt cookie với path khác, cần chỉ định trong tùy chọn delete.
        response.cookies.delete('access_token');
        response.cookies.delete('refresh_token');
        response.cookies.delete('user_data');

        return response; // Trả về response đã được gắn lệnh xóa/set cookie

    } catch (error) {
        console.error("App Route Handler: /api/auth/logout error:", error);
        return NextResponse.json({ message: 'Internal Server Error during logout' }, { status: 500 });
    }
}
