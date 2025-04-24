// src/middleware.ts

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

import { getSession } from '@/lib/auth'; // Giả sử hàm này đồng bộ

// Danh sách các đường dẫn công khai không yêu cầu token
const PUBLIC_PATHS = [
  '/auth/login',
  '/homeguest',
  '/newsguest',
  '/newsguest/[id]', // Đường dẫn động sẽ được xử lý bằng regex
  '/',
];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  console.log(`Middleware triggered for: ${pathname}`);

  // Lấy thông tin session/token
  const accessToken = getSession();

  // 1. Nếu có token, cho phép truy cập mọi trang (trừ trang login)
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
    if (publicPath.includes('[') && publicPath.includes(']')) {
      // Tạo regex từ path động, ví dụ: '/newsguest/[id]' -> /^\/newsguest\/[^/]+\/?$/
      const regexPath = publicPath.replace(/\[[^/]+\]/g, '[^/]+');
      const regex = new RegExp(`^${regexPath}\\/?$`);
      return regex.test(pathname);
    }
    return pathname === publicPath;
  });

  if (isPublicPath) {
    console.log(`Middleware: Path ${pathname} is public, proceeding without token.`);
    return NextResponse.next();
  } else {
    console.log(`Middleware: Path ${pathname} is protected, redirecting to login.`);
    const loginUrl = new URL('/auth/login', request.url);
    loginUrl.searchParams.set('redirectedFrom', pathname);
    return NextResponse.redirect(loginUrl);
  }
}

// Config Matcher: loại trừ các tài nguyên tĩnh không cần middleware
export const config = {
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico|.*\\..*).*)',
  ],
};
