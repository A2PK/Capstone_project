
import { type NextRequest, NextResponse } from 'next/server';

// Export hàm xử lý cho phương thức POST
export async function POST(req: NextRequest) {
    console.log("App Route Handler: /api/auth/logout - Clearing cookies...");

    try {
        // Tạo response để có thể set/delete cookie
        const response = NextResponse.json({ message: 'Logout successful' }, { status: 200 });
        response.cookies.delete('access_token');
        response.cookies.delete('refresh_token');
        response.cookies.delete('user_data');

        return response; // Trả về response đã được gắn lệnh xóa/set cookie

    } catch (error) {
        console.error("App Route Handler: /api/auth/logout error:", error);
        return NextResponse.json({ message: 'Internal Server Error during logout' }, { status: 500 });
    }
}
