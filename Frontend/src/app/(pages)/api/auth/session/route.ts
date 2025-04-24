import { type NextRequest, NextResponse } from 'next/server';
import type { User } from '@/types/user'; // Đảm bảo đường dẫn type User đúng

export async function GET(req: NextRequest) {
    try {
        // Lấy cookie từ NextRequest
        const accessToken = req.cookies.get('access_token')?.value; // Lấy giá trị của cookie

        if (accessToken) {
            console.log("App Route Handler: /api/auth/session - Access token found.");
            let userData: User | null = null;
            const userCookie = req.cookies.get('user_data')?.value;

            if (userCookie) {
                try {
                    userData = JSON.parse(userCookie);
                } catch (parseError) {
                    console.error("App Route Handler: /api/auth/session - Failed to parse user_data cookie:", parseError);
                }
            }

            // Trả về JSON với NextResponse
            return NextResponse.json({ isLoggedIn: true, user: userData, token: accessToken }, { status: 200 });

        } else {
            // Nếu không có access token -> Người dùng chưa đăng nhập
            console.log("App Route Handler: /api/auth/session - No access token found.");
            return NextResponse.json({ isLoggedIn: false, user: null , token: null }, { status: 200 }); // Trả về 200 theo logic cũ, nhưng 401 có thể hợp lý hơn
        }
    } catch (error) {
        console.error("App Route Handler: /api/auth/session error:", error);
        return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
    }
}
