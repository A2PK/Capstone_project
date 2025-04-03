'use client'
import Image from "next/image";
// import { redirect } from "next/navigation"; // Không dùng server action ở đây
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import logo from "/public/wqm.jpg";
import { useState, useEffect } from "react"; // Import useEffect
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
    LogOut, Bell, Mail, Menu, X, User, Settings, AlertCircle, Info, Edit3, Check // Thêm icons
} from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils"; // Import cn cho class động
import { NotificationItem, MessageItem } from "@/types/thongbao";
// --- Interface và Data giả lập (đã định nghĩa ở trên) ---
const mockNotifications: NotificationItem[] = [
  { id: 'n1', type: 'alert', text: 'Cảnh báo: Chỉ số pH tại Điểm A vượt ngưỡng!', timestamp: new Date(Date.now() - 1000 * 60 * 5), read: false, link: '#' },
  { id: 'n2', type: 'update', text: 'Model Itransformer v1.1 đã huấn luyện xong.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), read: false, link: '#' },
  { id: 'n3', type: 'info', text: 'Báo cáo tháng 3/2025 đã được xuất bản.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), read: true, link: '#' },
  { id: 'n4', type: 'warning', text: 'Pin yếu tại trạm quan trắc B.', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48), read: true },
];
const mockMessages: MessageItem[] = [
  { id: 'm1', sender: 'An Nguyễn', subject: 'Yêu cầu truy cập dữ liệu', snippet: 'Chào admin, tôi muốn xin quyền truy cập...', timestamp: new Date(Date.now() - 1000 * 60 * 30), read: false, avatar: 'https://randomuser.me/api/portraits/men/32.jpg' },
  { id: 'm2', sender: 'Hệ thống', subject: 'Xác nhận thay đổi mật khẩu', snippet: 'Mật khẩu của bạn đã được thay đổi...', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 5), read: true },
  { id: 'm3', sender: 'Bình Trần', subject: 'Thắc mắc về chỉ số COD', snippet: 'Tôi thấy chỉ số COD gần đây có vẻ hơi cao...', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 26), read: true, avatar: 'https://randomuser.me/api/portraits/women/44.jpg' },
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
  return Math.floor(seconds) + " giây trước";
}


export default function Navbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const router = useRouter();
  const [isLoggedIn, setIsLoggedIn] = useState(true); // Giữ state đăng nhập

  // --- State cho thông báo và tin nhắn ---
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [messages, setMessages] = useState<MessageItem[]>([]);
  const [unreadNotifications, setUnreadNotifications] = useState(0);
  const [unreadMessages, setUnreadMessages] = useState(0);

  // --- Load dữ liệu giả lập khi component mount ---
  useEffect(() => {
    // Trong ứng dụng thực tế, bạn sẽ fetch dữ liệu từ API ở đây
    setNotifications(mockNotifications);
    setMessages(mockMessages);
    setUnreadNotifications(mockNotifications.filter(n => !n.read).length);
    setUnreadMessages(mockMessages.filter(m => !m.read).length);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("loginrole");
    setIsLoggedIn(false); // Cập nhật state
    router.push("/auth/login");
  };

  // Hàm xử lý khi click vào một thông báo (ví dụ: đánh dấu đã đọc)
  const handleNotificationClick = (id: string) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
    setUnreadNotifications(prev => Math.max(0, prev - 1)); // Giảm số lượng chưa đọc
     // Optional: Điều hướng tới link nếu có
     const notification = notifications.find(n => n.id === id);
     if (notification?.link && notification.link !== '#') {
       router.push(notification.link);
     }
  };

  // Hàm xử lý khi click vào một tin nhắn
   const handleMessageClick = (id: string) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, read: true } : m));
    setUnreadMessages(prev => Math.max(0, prev - 1));
     // Optional: Điều hướng tới trang chi tiết tin nhắn
     // router.push(`/messages/${id}`);
     console.log(`Clicked message ${id}`);
   };

  // Hàm render icon cho loại thông báo
   const getNotificationIcon = (type: NotificationItem['type']) => {
        switch (type) {
          case 'alert': return <AlertCircle className="h-4 w-4 text-red-500 mr-2 flex-shrink-0" />;
          case 'update': return <Check className="h-4 w-4 text-blue-500 mr-2 flex-shrink-0" />;
          case 'info': return <Info className="h-4 w-4 text-sky-500 mr-2 flex-shrink-0" />;
          case 'warning': return <AlertCircle className="h-4 w-4 text-yellow-500 mr-2 flex-shrink-0" />;
          default: return <Bell className="h-4 w-4 text-gray-400 mr-2 flex-shrink-0" />;
        }
    };

  // --- Hàm render nội dung dropdown chung ---
  const renderDropdownContent = (
      items: any[],
      type: 'notification' | 'message',
      label: string,
      viewAllLink: string,
      itemClickHandler: (id: string) => void
    ) => (
        <DropdownMenuContent align="end" className="w-80 max-h-[80vh] overflow-y-auto"> {/* Giới hạn chiều cao và scroll */}
          <DropdownMenuLabel className="flex justify-between items-center">
            {label}
            <Link href={viewAllLink} className="text-xs text-blue-600 hover:underline">Xem tất cả</Link>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          {items.length === 0 ? (
             <DropdownMenuItem disabled className="text-center text-gray-500 py-4">Không có {label.toLowerCase()} mới</DropdownMenuItem>
          ) : (
            items.map((item) => (
              <DropdownMenuItem
                key={item.id}
                // Ngăn dropdown tự đóng khi click item, xử lý trong onClick
                onSelect={(e) => { e.preventDefault(); itemClickHandler(item.id); }}
                className={cn(
                  "flex items-start p-3 cursor-pointer data-[highlighted]:bg-gray-100 dark:data-[highlighted]:bg-gray-700", // Style khi hover
                  !item.read && "bg-blue-50 dark:bg-blue-900/30 font-medium" // Style cho item chưa đọc
                )}
              >
                {type === 'notification' ? (
                  <>
                    {getNotificationIcon(item.type)}
                    <div className="flex-1">
                        <p className="text-sm leading-snug">{item.text}</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{timeAgo(item.timestamp)}</p>
                    </div>
                  </>
                ) : ( // type === 'message'
                  <>
                    <Avatar className="h-8 w-8 mr-3 flex-shrink-0">
                      <AvatarImage src={item.avatar || '/default-avatar.png'} alt={item.sender} />
                      <AvatarFallback>{item.sender.charAt(0)}</AvatarFallback>
                    </Avatar>
                    <div className="flex-1 overflow-hidden">
                      <p className="text-sm font-semibold truncate">{item.sender}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-300 truncate">{item.subject}</p>
                      {/* <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{item.snippet}</p> */}
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{timeAgo(item.timestamp)}</p>
                    </div>
                  </>
                )}
                 {/* Chấm báo chưa đọc */}
                 {!item.read && (
                     <span className="ml-2 h-2 w-2 rounded-full bg-blue-500 flex-shrink-0"></span>
                 )}
              </DropdownMenuItem>
            ))
          )}
           <DropdownMenuSeparator />
           <DropdownMenuItem className="justify-center" asChild>
                <Link href={viewAllLink} className="text-sm text-blue-600 hover:underline">Xem tất cả</Link>
           </DropdownMenuItem>
        </DropdownMenuContent>
  );


  return (
    <nav className="bg-white shadow-md border-b border-gray-200 dark:bg-gray-800 dark:border-gray-700 relative z-[50]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo Section */}
          <div className="flex items-center">
            <Link href={isLoggedIn ? "/dashboardofficer/homepage" : "/"}> {/* Link về dashboard nếu đăng nhập */}
                <Image
                src={logo}
                alt="Water Monitoring System Logo"
                width={90}
                height={40}
                priority // Ưu tiên tải logo
                />
            </Link>
          </div>

          {/* Desktop Icons and User Section */}
          <div className="hidden sm:flex items-center space-x-1 md:space-x-2"> {/* Giảm space */}

            {/* Notifications Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="relative">
                  {unreadNotifications > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                        {/* Hoặc hiển thị số lượng nhỏ */}
                        {/* <span className="absolute -top-1 -right-1 inline-flex items-center justify-center px-1.5 py-0.5 text-xs font-bold leading-none text-red-100 transform translate-x-1/2 -translate-y-1/2 bg-red-600 rounded-full">{unreadNotifications}</span> */}
                    </span>
                  )}
                  <Bell className="h-5 w-5 text-gray-500 dark:text-gray-300" />
                </Button>
              </DropdownMenuTrigger>
              {renderDropdownContent(notifications, 'notification', 'Thông báo', '/notifications', handleNotificationClick)}
            </DropdownMenu>

            {/* Messages Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                 <Button variant="ghost" size="icon" className="relative">
                   {unreadMessages > 0 && (
                     <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                    </span>
                   )}
                  <Mail className="h-5 w-5 text-gray-500 dark:text-gray-300" />
                </Button>
              </DropdownMenuTrigger>
              {renderDropdownContent(messages, 'message', 'Tin nhắn', '/messages', handleMessageClick)}
            </DropdownMenu>

            {/* User Dropdown */}
            {isLoggedIn && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                   <Button variant="ghost" className="relative rounded-full p-1">
                    <Avatar className="h-8 w-8"> {/* Size nhỏ hơn */}
                      <AvatarImage src="https://github.com/shadcn.png" alt="User Avatar"/>
                      <AvatarFallback>U</AvatarFallback>
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                   <DropdownMenuLabel>Tài khoản của tôi</DropdownMenuLabel>
                   <DropdownMenuSeparator />
                  <DropdownMenuItem asChild>
                    <Link href="/profile" className="cursor-pointer">
                      <User className="mr-2 h-4 w-4" />
                      <span>Hồ sơ</span>
                    </Link>
                  </DropdownMenuItem>
                   {/* <DropdownMenuItem asChild>
                     <Link href="/settings" className="cursor-pointer">
                       <Settings className="mr-2 h-4 w-4" />
                       <span>Cài đặt</span>
                     </Link>
                   </DropdownMenuItem> */}
                   <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={(e) => {e.preventDefault(); handleLogout(); }} className="cursor-pointer text-red-600 focus:bg-red-50 focus:text-red-700 dark:focus:bg-red-900/50 dark:focus:text-red-400">
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Đăng xuất</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            {!isLoggedIn && (
                 <Button asChild>
                    <Link href="/auth/login">Đăng nhập</Link>
                 </Button>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="flex sm:hidden">
            <Button variant="ghost" size="icon" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
              {isMobileMenuOpen ? <X className="block h-6 w-6" /> : <Menu className="block h-6 w-6" />}
            </Button>
          </div>
        </div>

         {/* Mobile Menu Dropdown */}
        {isMobileMenuOpen && (
          <div className="sm:hidden absolute top-16 inset-x-0 z-50 bg-white dark:bg-gray-800 shadow-lg border-t dark:border-gray-700 z-[50]">
            <div className="px-2 pt-2 pb-3 space-y-1">
                {/* Mobile Icons (sử dụng lại dropdown logic) */}
                <div className="flex justify-around items-center px-2 py-2 border-b dark:border-gray-600">
                     {/* Mobile Notifications */}
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="relative">
                                {unreadNotifications > 0 && ( <span className="absolute top-0 right-0 flex h-2.5 w-2.5"><span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span></span> )}
                                <Bell className="h-6 w-6 text-gray-500 dark:text-gray-300" />
                            </Button>
                        </DropdownMenuTrigger>
                         {renderDropdownContent(notifications, 'notification', 'Thông báo', '/notifications', handleNotificationClick)}
                    </DropdownMenu>

                     {/* Mobile Messages */}
                     <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="relative">
                                {unreadMessages > 0 && ( <span className="absolute top-0 right-0 flex h-2.5 w-2.5"><span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span></span> )}
                                <Mail className="h-6 w-6 text-gray-500 dark:text-gray-300" />
                            </Button>
                        </DropdownMenuTrigger>
                        {renderDropdownContent(messages, 'message', 'Tin nhắn', '/messages', handleMessageClick)}
                     </DropdownMenu>
                </div>


              {/* Mobile User Actions */}
              {isLoggedIn && (
                <>
                  <Link href="/profile" className="flex items-center px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700" onClick={()=>setIsMobileMenuOpen(false)}>
                     <User className="mr-3 h-5 w-5" /> Hồ sơ
                  </Link>
                   <Link href="/settings" className="flex items-center px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700" onClick={()=>setIsMobileMenuOpen(false)}>
                     <Settings className="mr-3 h-5 w-5" /> Cài đặt
                  </Link>
                  <Button variant="ghost" onClick={() => { handleLogout(); setIsMobileMenuOpen(false); }} className="w-full justify-start flex items-center px-3 py-2 rounded-md text-base font-medium text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/50">
                    <LogOut className="mr-3 h-5 w-5" /> Đăng xuất
                  </Button>
                </>
              )}
               {!isLoggedIn && (
                 <Link href="/auth/login" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700" onClick={()=>setIsMobileMenuOpen(false)}>
                    Đăng nhập
                 </Link>
               )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}