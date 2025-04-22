
import { NextResponse } from 'next/server';
import { getUserRole } from '@/lib/auth'; 

export const dynamic = 'force-dynamic' 

export async function GET() {
  try {
    // Gọi hàm server-side để lấy role từ cookie (user_data hoặc qua /api/me)
    const role = await getUserRole(); // getUserRole là async nếu getCurrentUser là async

    if (role) {
      // Nếu có role, trả về role đó
      return NextResponse.json({ role: role });
    } else {
      // Nếu không có role (chưa đăng nhập hoặc cookie lỗi/không có)
      return NextResponse.json({ role: null }, { status: 401 }); // Trả về 401 Unauthorized
    }
  } catch (error) {
    console.error('/api/auth/role error:', error);
    return NextResponse.json({ message: 'Internal Server Error' }, { status: 500 });
  }
}