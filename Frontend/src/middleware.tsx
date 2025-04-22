// src/middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getSession } from '@/lib/auth'; // Giả sử hàm này đồng bộ

// Danh sách các đường dẫn công khai không yêu cầu token
const PUBLIC_PATHS = [
  '/auth/login',
  '/homeguest',
  '/newsguest',
  '/newsguestdetail',
  '/',
  // Thêm các đường dẫn public khác nếu cần, ví dụ: '/about', '/contact'
  // Lưu ý: Nếu '/newsguestdetail' có dạng động như '/newsguestdetail/[slug]',
  // bạn cần xử lý phức tạp hơn hoặc dùng regex trong matcher/logic.
  // Hiện tại, nó chỉ khớp chính xác '/newsguestdetail'.
];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  console.log(`Middleware triggered for: ${pathname}`);

  // Lấy thông tin session/token
  const accessToken = getSession();

  // --- Logic xử lý ---

  // 1. Nếu có token, cho phép truy cập mọi trang (trừ khi có logic khác)
  if (accessToken) {
    console.log('Middleware: Access token found, proceeding.');
    if (pathname === '/auth/login') {
      console.log('Middleware: Logged in user accessing login page, redirecting to dashboard.');
      return NextResponse.redirect(new URL('/', request.url)); 
    }
    return NextResponse.next();
  }

  // 2. Nếu KHÔNG có token
  console.log('Middleware: No access token.');

  // Kiểm tra xem path hiện tại có nằm trong danh sách public không
  const isPublicPath = PUBLIC_PATHS.some(publicPath => {
    // Xử lý trường hợp trang chi tiết có dạng động (ví dụ)
    if (publicPath.includes('[') && publicPath.includes(']')) {
      // Tạo regex từ path động, ví dụ: '/newsguestdetail/[slug]' -> /^\/newsguestdetail\/[^/]+\/?$/
      const regexPath = publicPath.replace(/\[\.\.\..+\]/, '.+').replace(/\[[^/]+\]/g, '[^/]+');
      const regex = new RegExp(`^${regexPath}\\/?$`); // Thêm \/? để khớp cả khi có/không có dấu / cuối
      return regex.test(pathname);
    }
    // So khớp chính xác cho các path tĩnh
    return pathname === publicPath;
  });


  if (isPublicPath) {
    // Nếu là trang public, cho phép truy cập
    console.log(`Middleware: Path ${pathname} is public, proceeding without token.`);
    return NextResponse.next();
  } else {
    // Nếu là trang cần bảo vệ và không có token, chuyển hướng về login
    console.log(`Middleware: Path ${pathname} is protected, redirecting to login.`);
    const loginUrl = new URL('/auth/login', request.url);
    loginUrl.searchParams.set('redirectedFrom', pathname); // Giữ lại trang muốn truy cập
    return NextResponse.redirect(loginUrl);
  }
}

// --- Config Matcher được cập nhật ---
// Chỉ loại trừ các tài nguyên thực sự không cần chạy middleware
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * Mọi thứ khác (bao gồm cả trang login, homeguest,...) SẼ chạy qua middleware.
     */
    '/((?!api|_next/static|_next/image|favicon.ico|.*\\..*).*)',
  ],
};