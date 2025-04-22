// src/components/layout/Navbar.tsx (Hoặc đường dẫn tương ứng của bạn)

'use client'; // Đánh dấu đây là Client Component

import Image from "next/image";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import logo from "/public/wqm.jpg"; // Đảm bảo đường dẫn đến logo chính xác
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
    LogOut, Bell, Mail, Menu, X, User, Settings, AlertCircle, Info, Check
} from "lucide-react"; // Import các icon cần dùng
import Link from "next/link";
import { cn } from "@/lib/utils"; // Import tiện ích class names
import { NotificationItem, MessageItem } from "@/types/thongbao"; // Import kiểu dữ liệu thông báo/tin nhắn
import { User as AppUser } from '@/types/user'; // Import kiểu User và đổi tên để tránh trùng với icon

// --- Interface và Data giả lập (Bạn sẽ thay thế bằng fetch API thực tế) ---
const mockNotifications: NotificationItem[] = [
    { id: 'n1', type: 'alert', text: 'Cảnh báo: Chỉ số pH tại Điểm A vượt ngưỡng!', timestamp: new Date(Date.now() - 1000 * 60 * 5), read: false, link: '#' },
    { id: 'n2', type: 'update', text: 'Model Itransformer v1.1 đã huấn luyện xong.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), read: false, link: '#' },
    { id: 'n3', type: 'info', text: 'Báo cáo tháng 4/2025 đã được xuất bản.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), read: true, link: '#' }, // Cập nhật tháng nếu cần
    { id: 'n4', type: 'warning', text: 'Pin yếu tại trạm quan trắc B.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48), read: true },
];

// --- Hết phần data giả lập ---

// Hàm helper để hiển thị thời gian tương đối
function timeAgo(date: Date): string {
    const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    let interval = seconds / 31536000;
    if (interval > 1) return Math.floor(interval) + " năm trước";
    interval = seconds / 2592000;
    if (interval > 1) return Math.floor(interval) + " tháng trước";
    interval = seconds / 86400;
    if (interval > 1) return Math.floor(interval) + " ngày trước";
    interval = seconds / 3600;
    if (interval > 1) return Math.floor(interval) + " giờ trước";
    interval = seconds / 60;
    if (interval > 1) return Math.floor(interval) + " phút trước";
    // Hiển thị "vài giây trước" thay vì số giây cụ thể nếu quá nhỏ
    return seconds < 10 ? "vài giây trước" : Math.floor(seconds) + " giây trước";
}

// === Navbar Component ===
export default function Navbar() {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const router = useRouter();

    // --- State cho Authentication ---
    const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false); // Trạng thái đăng nhập, mặc định là chưa
    const [user, setUser] = useState<AppUser | null>(null);       // Thông tin người dùng, null nếu chưa đăng nhập
    const [isLoading, setIsLoading] = useState<boolean>(true);     // Trạng thái chờ kiểm tra session

    // --- State cho thông báo và tin nhắn ---
    const [notifications, setNotifications] = useState<NotificationItem[]>([]);
    const [unreadNotifications, setUnreadNotifications] = useState(0);

    // --- Load dữ liệu và Kiểm tra Session khi component mount ---
    useEffect(() => {
        // Tải dữ liệu thông báo/tin nhắn (thay bằng fetch API thật)
        setNotifications(mockNotifications);
        setUnreadNotifications(mockNotifications.filter(n => !n.read).length);

        // Kiểm tra session người dùng qua API
        const checkSession = async () => {
            setIsLoading(true); // Bắt đầu kiểm tra
            try {
                const response = await fetch('/api/auth/session'); // Gọi API session
                if (response.ok) {
                    const data: { isLoggedIn: boolean; user: AppUser | null } = await response.json();
                    setIsLoggedIn(data.isLoggedIn);
                    setUser(data.user); // Lưu thông tin user nếu có
                    console.log("Session check success:", data);
                } else {
                    // Xử lý trường hợp API trả về lỗi (vd: 401, 500)
                    console.error("Failed to fetch session status:", response.status, response.statusText);
                    setIsLoggedIn(false);
                    setUser(null);
                }
            } catch (error) {
                console.error("Error checking session:", error);
                // Đảm bảo trạng thái là chưa đăng nhập nếu có lỗi mạng
                setIsLoggedIn(false);
                setUser(null);
            } finally {
                setIsLoading(false); // Kết thúc kiểm tra
            }
        };

        checkSession();
    }, []); // Chỉ chạy một lần khi component mount

    // --- Xử lý Logout ---
    const handleLogout = async () => {
        console.log("Initiating logout...");
        try {
            const response = await fetch('/api/auth/logout', {
                method: 'POST', // Gửi yêu cầu POST đến API logout
            });

            if (response.ok) {
                console.log("Logout successful via API");
                setIsLoggedIn(false);       // Cập nhật trạng thái UI
                setUser(null);            // Xóa thông tin user khỏi state
                setIsMobileMenuOpen(false); // Đóng menu mobile (nếu đang mở)
                router.push("/auth/login"); // Điều hướng về trang đăng nhập
            } else {
                const errorData = await response.json().catch(() => ({ message: 'Logout failed with status ' + response.status }));
                console.error("Logout failed:", response.status, errorData.message);
                // Có thể hiển thị thông báo lỗi cho người dùng
                alert(`Đăng xuất thất bại: ${errorData.message}`);
            }
        } catch (error) {
            console.error("Error during logout request:", error);
            // Có thể hiển thị thông báo lỗi cho người dùng
            alert("Đã xảy ra lỗi trong quá trình đăng xuất. Vui lòng thử lại.");
        }
    };

    // --- Các hàm xử lý Click thông báo/tin nhắn ---
    const handleNotificationClick = (id: string) => {
        // Đánh dấu đã đọc (client-side, cần gọi API để cập nhật DB)
        setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
        setUnreadNotifications(prev => Math.max(0, prev - 1));

        // Tìm và điều hướng nếu có link
        const notification = notifications.find(n => n.id === id);
        if (notification?.link && notification.link !== '#') {
            router.push(notification.link);
            setIsMobileMenuOpen(false); // Đóng menu mobile nếu đang mở
        }
        // TODO: Gọi API để đánh dấu thông báo là đã đọc trên server
        console.log(`Marked notification ${id} as read (client-side)`);
    };

    // --- Hàm lấy Icon cho loại thông báo ---
    const getNotificationIcon = (type: NotificationItem['type']) => {
        switch (type) {
            case 'alert': return <AlertCircle className="h-4 w-4 text-red-500 mr-2 flex-shrink-0" />;
            case 'update': return <Check className="h-4 w-4 text-blue-500 mr-2 flex-shrink-0" />;
            case 'info': return <Info className="h-4 w-4 text-sky-500 mr-2 flex-shrink-0" />;
            case 'warning': return <AlertCircle className="h-4 w-4 text-yellow-500 mr-2 flex-shrink-0" />;
            default: return <Bell className="h-4 w-4 text-gray-400 mr-2 flex-shrink-0" />;
        }
    };

    // --- Hàm render nội dung Dropdown chung (Thông báo/Tin nhắn) ---
    const renderDropdownContent = (
        items: Array<NotificationItem | MessageItem>, // Sử dụng union type
        type: 'notification' | 'message',
        label: string,
        viewAllLink: string,
        itemClickHandler: (id: string) => void
    ) => (
        <DropdownMenuContent align="end" className="w-80 max-h-[80vh] overflow-y-auto shadow-xl">
            <DropdownMenuLabel className="flex justify-between items-center px-3 py-2">
                <span className="font-semibold">{label}</span>
                <Link href={viewAllLink} className="text-xs text-blue-600 hover:underline" onClick={() => setIsMobileMenuOpen(false)}>
                    Xem tất cả
                </Link>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            {items.length === 0 ? (
                <DropdownMenuItem disabled className="text-center text-gray-500 py-4 italic">
                    Không có {label.toLowerCase()} mới
                </DropdownMenuItem>
            ) : (
                items.map((item) => (
                    <DropdownMenuItem
                        key={item.id}
                        // Ngăn dropdown tự đóng, xử lý trong onClick
                        onSelect={(e) => { e.preventDefault(); itemClickHandler(item.id); }}
                        className={cn(
                            "flex items-start p-3 cursor-pointer data-[highlighted]:bg-gray-100 dark:data-[highlighted]:bg-gray-700 transition-colors duration-150 ease-in-out", // Style hover
                            !item.read && "bg-blue-50 dark:bg-blue-900/30 font-medium" // Style chưa đọc
                        )}
                    >
                        {type === 'notification' ? (
                            <>
                                {getNotificationIcon((item as NotificationItem).type)}
                                <div className="flex-1">
                                    <p className="text-sm leading-snug">{(item as NotificationItem).text}</p>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{timeAgo(item.timestamp)}</p>
                                </div>
                            </>
                        ) : ( // type === 'message'
                            <>
                                <Avatar className="h-8 w-8 mr-3 flex-shrink-0">
                                    <AvatarImage src={(item as MessageItem).avatar || '/default-avatar.png'} alt={(item as MessageItem).sender} />
                                    <AvatarFallback>{(item as MessageItem).sender.charAt(0)}</AvatarFallback>
                                </Avatar>
                                <div className="flex-1 overflow-hidden">
                                    <p className="text-sm font-semibold truncate">{(item as MessageItem).sender}</p>
                                    <p className="text-sm text-gray-600 dark:text-gray-300 truncate">{(item as MessageItem).subject}</p>
                                    {/* <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{item.snippet}</p> */}
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{timeAgo(item.timestamp)}</p>
                                </div>
                            </>
                        )}
                        {/* Chấm báo chưa đọc */}
                        {!item.read && (
                            <span className="ml-2 mt-1 h-2 w-2 rounded-full bg-blue-500 flex-shrink-0 self-start"></span>
                        )}
                    </DropdownMenuItem>
                ))
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem className="justify-center py-2" asChild>
                <Link href={viewAllLink} className="text-sm text-blue-600 hover:underline" onClick={() => setIsMobileMenuOpen(false)}>
                    Xem tất cả {label.toLowerCase()}
                </Link>
            </DropdownMenuItem>
        </DropdownMenuContent>
    );

    // === Render JSX ===
    return (
        <nav className="bg-white shadow-md border-b border-gray-200 dark:bg-gray-800 dark:border-gray-700 sticky top-0 left-0 right-0 z-[50]"> {/* sticky để giữ navbar */}
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                <div className="flex h-16 items-center justify-between">
                    {/* Logo Section */}
                    <div className="flex items-center">
                        <Link href={"/"} onClick={() => setIsMobileMenuOpen(false)}>
                            <Image
                                src={logo}
                                alt="Water Monitoring System Logo"
                                width={90} // Có thể điều chỉnh kích thước
                                height={40}
                                priority // Ưu tiên tải logo
                                style={{ objectFit: 'contain' }} // Đảm bảo logo không bị méo
                            />
                        </Link>
                    </div>

                    {/* Desktop Icons and User Section */}
                    <div className="hidden sm:flex items-center space-x-1 md:space-x-2">
                        {/* Hiển thị khi đang tải hoặc đã tải xong */}
                        {isLoading ? (
                            // Placeholder đơn giản khi đang loading
                            <div className="flex items-center space-x-2">
                                <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse"></div>
                                <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse"></div>
                                <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse"></div>
                            </div>
                        ) : isLoggedIn && user ? (
                            // Hiển thị các icon và user dropdown khi đã đăng nhập
                            <>
                                {/* Notifications Dropdown */}
                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        <Button variant="ghost" size="icon" className="relative rounded-full hover:bg-gray-100 dark:hover:bg-gray-700">
                                            {unreadNotifications > 0 && (
                                                <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                                                </span>
                                            )}
                                            <Bell className="h-5 w-5 text-gray-500 dark:text-gray-300" />
                                        </Button>
                                    </DropdownMenuTrigger>
                                    {renderDropdownContent(notifications, 'notification', 'Thông báo', '/notifications', handleNotificationClick)}
                                </DropdownMenu>

                                {/* User Dropdown */}
                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        {/* Thêm hiệu ứng hover cho button avatar */}
                                        <Button variant="ghost" className="relative rounded-full p-0.5 hover:bg-gray-100 dark:hover:bg-gray-700 h-9 w-9 flex items-center justify-center">
                                            <Avatar className="h-8 w-8">
                                                <AvatarImage
                                                    src={user.profilePic || "https://github.com/shadcn.png"}
                                                    alt={user.username || "User Avatar"}
                                                />
                                                <AvatarFallback className="bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200">
                                                    {
                                                        user.firstName?.charAt(0)?.toUpperCase() ||
                                                        user.username?.charAt(0)?.toUpperCase() ||
                                                        user.email?.charAt(0)?.toUpperCase() ||
                                                        'U'
                                                    }
                                                </AvatarFallback>
                                            </Avatar>
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent align="end" className="w-56 shadow-xl">
                                        <DropdownMenuLabel className="font-normal">
                                            <div className="flex flex-col space-y-1">
                                                <p className="text-sm font-medium leading-none">
                                                    {
                                                        `${user.firstName || ''} ${user.lastName || ''}`.trim() ||
                                                        user.username || 'Người dùng'
                                                    }
                                                </p>
                                                <p className="text-xs leading-none text-muted-foreground">
                                                    {user.email}
                                                </p>
                                            </div>
                                        </DropdownMenuLabel>
                                        <DropdownMenuSeparator />
                                        <DropdownMenuItem asChild>
                                            <Link href="/profile" className="cursor-pointer flex items-center">
                                                <User className="mr-2 h-4 w-4" />
                                                <span>Hồ sơ</span>
                                            </Link>
                                        </DropdownMenuItem>
                                        {/* Có thể thêm link Cài đặt nếu cần */}
                                        {/* <DropdownMenuItem asChild>
                                          <Link href="/settings" className="cursor-pointer flex items-center">
                                              <Settings className="mr-2 h-4 w-4" />
                                              <span>Cài đặt</span>
                                          </Link>
                                        </DropdownMenuItem> */}
                                        <DropdownMenuSeparator />
                                        {/* Sử dụng onSelect thay vì onClick để đảm bảo hoạt động đúng trong DropdownMenu */}
                                        <DropdownMenuItem onSelect={(e) => { e.preventDefault(); handleLogout(); }} className="cursor-pointer text-red-600 focus:bg-red-50 focus:text-red-600 dark:focus:bg-red-900/50 dark:focus:text-red-400 flex items-center">
                                            <LogOut className="mr-2 h-4 w-4" />
                                            <span>Đăng xuất</span>
                                        </DropdownMenuItem>
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </>
                        ) : (
                            // Hiển thị nút Đăng nhập khi chưa đăng nhập
                            <Button asChild size="sm">
                                <Link href="/auth/login">Đăng nhập</Link>
                            </Button>
                        )}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex sm:hidden items-center">
                         {/* Nút mở menu mobile */}
                        <Button variant="ghost" size="icon" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="rounded-full">
                            {isMobileMenuOpen ? <X className="block h-6 w-6" /> : <Menu className="block h-6 w-6" />}
                        </Button>
                    </div>
                </div>

                {/* Mobile Menu Dropdown */}
                {isMobileMenuOpen && (
                    // Overlay mờ phía dưới khi mở menu mobile (tùy chọn)
                    // <div className="fixed inset-0 bg-black/30 z-40 sm:hidden" onClick={() => setIsMobileMenuOpen(false)}></div>

                    <div className="sm:hidden absolute top-16 inset-x-0 z-[49] bg-white dark:bg-gray-800 shadow-lg border-t dark:border-gray-700 animate-slide-down"> {/* Thêm animation */}
                        <div className="px-2 pt-2 pb-3 space-y-1">
                            {/* Hiển thị khi đang tải hoặc đã tải xong */}
                            {isLoading ? (
                                <div className="text-center p-4 text-gray-500 dark:text-gray-400 italic">Đang tải...</div>
                            ) : isLoggedIn && user ? (
                                // Menu mobile khi đã đăng nhập
                                <>
                                    {/* Thông tin User ở trên cùng mobile menu */}
                                    <div className="px-3 py-3 border-b dark:border-gray-600 mb-2">
                                        <div className="flex items-center space-x-3">
                                            <Avatar className="h-10 w-10">
                                                <AvatarImage src={user.profilePic || "https://github.com/shadcn.png"} alt={user.username || "User Avatar"}/>
                                                <AvatarFallback className="bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200">
                                                    { user.firstName?.charAt(0)?.toUpperCase() || user.username?.charAt(0)?.toUpperCase() || user.email?.charAt(0)?.toUpperCase() || 'U' }
                                                </AvatarFallback>
                                            </Avatar>
                                            <div>
                                                <p className="text-sm font-medium text-gray-900 dark:text-white">
                                                    {`${user.firstName || ''} ${user.lastName || ''}`.trim() || user.username || 'Người dùng'}
                                                </p>
                                                <p className="text-xs text-gray-500 dark:text-gray-400">
                                                    {user.email}
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Mobile Icons (Notifications/Messages) - Có thể dùng Dropdown hoặc Link trực tiếp */}
                                    <div className="flex justify-around items-center px-2 py-2 border-b dark:border-gray-600">
                                        <DropdownMenu>
                                             <DropdownMenuTrigger asChild>
                                                 <Button variant="ghost" size="icon" className="relative flex flex-col items-center h-auto py-1 px-2 space-y-1">
                                                     {unreadNotifications > 0 && ( <span className="absolute top-0 right-1 flex h-2.5 w-2.5"><span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span></span> )}
                                                     <Bell className="h-5 w-5 text-gray-500 dark:text-gray-300" />
                                                     <span className="text-xs text-gray-600 dark:text-gray-400">Thông báo</span>
                                                 </Button>
                                             </DropdownMenuTrigger>
                                             {renderDropdownContent(notifications, 'notification', 'Thông báo', '/notifications', handleNotificationClick)}
                                         </DropdownMenu>
                                    </div>

                                    {/* Mobile User Actions */}
                                    <Link href="/profile" className="flex items-center px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700" onClick={()=>setIsMobileMenuOpen(false)}>
                                        <User className="mr-3 h-5 w-5" /> Hồ sơ
                                    </Link>
                                    {/* <Link href="/settings" className="flex items-center px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700" onClick={()=>setIsMobileMenuOpen(false)}>
                                        <Settings className="mr-3 h-5 w-5" /> Cài đặt
                                    </Link> */}
                                    <Button variant="ghost" onClick={handleLogout} className="w-full justify-start flex items-center px-3 py-2 rounded-md text-base font-medium text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/50 dark:hover:text-red-400">
                                        <LogOut className="mr-3 h-5 w-5" /> Đăng xuất
                                    </Button>
                                </>
                            ) : (
                                // Menu mobile khi chưa đăng nhập
                                <Link href="/auth/login" className="block w-full px-3 py-2 rounded-md text-base font-medium text-center text-white bg-blue-600 hover:bg-blue-700" onClick={()=>setIsMobileMenuOpen(false)}>
                                    Đăng nhập
                                </Link>
                            )}
                        </div>
                    </div>
                )}
            </div>
            {/* Thêm một lớp CSS để tạo animation cho mobile menu nếu muốn */}
            <style jsx global>{`
                .animate-slide-down {
                    animation: slide-down 0.3s ease-out forwards;
                }
                @keyframes slide-down {
                    from {
                        opacity: 0;
                        transform: translateY(-10px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
            `}</style>
        </nav>
    );
}