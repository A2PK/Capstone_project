// src/components/layout/Navbar.tsx

'use client';

import Image from "next/image";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import logo from "/public/wqm.jpg";
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
    LogOut, Bell, Menu, X, User, Settings, AlertCircle, Info, Check
    // Mail icon might not be needed if only handling Notifications
} from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { Notification as NotificationData } from '@/types/notification'; // Use your actual Notification interface
import { User as AppUser } from '@/types/user';
import { useToast } from "@/hooks/use-toast"; // Import toast
import { findNotificationsByUserId, getAllNotifications, markAsReadNotifications } from '@/lib/notification'; // Import notification API functions
import { parseISO, isValid as isValidDate } from "date-fns";

// --- Helper function for timeAgo (Keep as before) ---
function timeAgo(dateString: string): string {
    try {
        const date = parseISO(dateString);
        if (!isValidDate(date)) return dateString; // Return original string if invalid
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
        return seconds < 10 ? "vài giây trước" : Math.floor(seconds) + " giây trước";
    } catch (e) {
        console.error("Error parsing date for timeAgo:", dateString, e);
        return dateString; // Return original on error
    }
}


// === Navbar Component ===
export default function Navbar() {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const router = useRouter();
    const { toast } = useToast();

    // --- State for Authentication ---
    const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
    const [user, setUser] = useState<AppUser | null>(null);
    const [token, setToken] = useState<string | null>(null); // Store the token
    const [isLoadingSession, setIsLoadingSession] = useState<boolean>(true);

    // --- State for notifications ---
    const [notifications, setNotifications] = useState<NotificationData[]>([]);
    const [isLoadingNotifications, setIsLoadingNotifications] = useState<boolean>(false);
    const [unreadNotificationsCount, setUnreadNotificationsCount] = useState(0);

    // --- Fetch Session and Token ---
    useEffect(() => {
        const checkSession = async () => {
            setIsLoadingSession(true);
            try {
                const response = await fetch('/api/auth/session');
                if (response.ok) {
                    // Expect token to be returned from API now
                    const data: { isLoggedIn: boolean; user: AppUser | null; token: string | null } = await response.json();
                    setIsLoggedIn(data.isLoggedIn);
                    setUser(data.user);
                    setToken(data.token); // Store the token
                    console.log("Session check success:", data.isLoggedIn);
                } else {
                    console.error("Failed to fetch session status:", response.status);
                    setIsLoggedIn(false); setUser(null); setToken(null);
                }
            } catch (error) {
                console.error("Error checking session:", error);
                setIsLoggedIn(false); setUser(null); setToken(null);
            } finally {
                setIsLoadingSession(false);
            }
        };
        checkSession();
    }, []);

    // --- Fetch Notifications when user/token is available ---
    useEffect(() => {
        // Only fetch if logged in, user and token are available
        if (isLoggedIn && user && token) {
            const fetchNotifications = async () => {
                setIsLoadingNotifications(true);
                try {
                    // Fetch latest notifications (e.g., limit 10, sorted by newest)
                    const fetchedNotifications = await getAllNotifications(token, {
                        limit: 10, // Adjust limit as needed
                        sortBy: 'created_at',
                        sortDesc: true
                    });
                    setNotifications(fetchedNotifications || []); // Ensure it's an array
                    setUnreadNotificationsCount(fetchedNotifications?.filter(n => !n.read).length ?? 0);
                } catch (error) {
                    console.error("Failed to fetch notifications:", error);
                    toast({ variant: "destructive", title: "Lỗi", description: "Không thể tải thông báo." });
                    setNotifications([]); // Clear on error
                    setUnreadNotificationsCount(0);
                } finally {
                    setIsLoadingNotifications(false);
                }
            };
            fetchNotifications();
        } else {
            // Clear notifications if not logged in
            setNotifications([]);
            setUnreadNotificationsCount(0);
        }
    }, [isLoggedIn, user, token, toast]); // Rerun when login state, user, or token changes


    // --- Xử lý Logout (Keep as before) ---
    const handleLogout = async () => {
        console.log("Initiating logout...");
         try {
             const response = await fetch('/api/auth/logout', { method: 'POST' });
             if (response.ok) {
                 console.log("Logout successful via API");
                 setIsLoggedIn(false); setUser(null); setToken(null); // Clear token on logout
                 setIsMobileMenuOpen(false);
                 router.push("/auth/login");
             } else {
                 const errorData = await response.json().catch(() => ({ message: 'Logout failed' }));
                 console.error("Logout failed:", response.status, errorData.message);
                 alert(`Đăng xuất thất bại: ${errorData.message}`);
             }
         } catch (error) {
             console.error("Error during logout request:", error);
             alert("Đã xảy ra lỗi trong quá trình đăng xuất.");
         }
    };

    // --- Handle Notification Click ---
    const handleNotificationClick = async (notification: NotificationData) => {
        const { id, read } = notification;
        let needsApiCall = !read; // Only call API if it's currently unread

        // Optimistic UI update: Mark as read immediately on client
        setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
        // Recalculate unread count based on the updated client state
        // Use a function form of setUnreadNotificationsCount to ensure we're working with the *latest* notifications state
        setUnreadNotificationsCount(prevCount => notifications.filter(n => n.id !== id && !n.read).length);


        // TODO: Add navigation logic if notification has a specific link/action
        // if (notification.link) router.push(notification.link);
        // Consider closing dropdown after click? - Handled by onSelect={(e) => { e.preventDefault(); ... }}
        // setIsMobileMenuOpen(false); // Close mobile menu if open - handled below inside onSelect

        // Call API to mark as read on server if needed
        if (needsApiCall && token) {
            try {
                await markAsReadNotifications(token, [id]); // Pass token and array of ID(s)
                console.log(`Marked notification ${id} as read on server.`);
                // Optional: Refetch notifications after marking read, or rely on optimistic update
            } catch (error) {
                console.error(`Failed to mark notification ${id} as read on server:`, error);
                toast({ variant: "destructive", title: "Lỗi", description: "Không thể cập nhật trạng thái thông báo." });
                // Optional: Revert optimistic update if API call fails?
                // setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: false } : n)); // Revert
                // setUnreadNotificationsCount(prev => prev + 1); // Revert count
            }
        }
    };

    // --- Get Notification Icon (Using a default or based on title/desc) ---
    const getNotificationIcon = (notification: NotificationData) => {
        // Simple logic: Use different icons based on keywords in title/description
        const titleLower = notification.title.toLowerCase();
        const descLower = notification.description.toLowerCase();
        if (titleLower.includes('cảnh báo') || descLower.includes('cảnh báo') || titleLower.includes('alert') || titleLower.includes('warning')) {
            return <AlertCircle className="h-4 w-4 text-red-500 mr-2 flex-shrink-0" />;
        }
        if (titleLower.includes('cập nhật') || descLower.includes('cập nhật') || titleLower.includes('update') || titleLower.includes('huấn luyện')) {
            return <Info className="h-4 w-4 text-blue-500 mr-2 flex-shrink-0" />;
        }
         if (titleLower.includes('thành công') || descLower.includes('thành công') || titleLower.includes('hoàn thành')) {
            return <Check className="h-4 w-4 text-green-500 mr-2 flex-shrink-0" />;
         }
        // Default icon
        return <Bell className="h-4 w-4 text-gray-400 mr-2 flex-shrink-0" />;
    };

    // --- Render Dropdown Content ---
    const renderNotificationDropdown = () => (
        <DropdownMenuContent align="end" className="w-80 max-h-[80vh] overflow-y-auto shadow-xl">
            <DropdownMenuLabel className="flex justify-between items-center px-3 py-2">
                <span className="font-semibold">Thông báo</span>
                {/* Maybe link to profile/notifications tab? */}
                <Link href="/profile?tab=notifications" className="text-xs text-blue-600 hover:underline" onClick={() => setIsMobileMenuOpen(false)}>
                    Xem tất cả
                </Link>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            {isLoadingNotifications ? (
                    <DropdownMenuItem disabled className="flex justify-center items-center text-center text-gray-500 py-4 italic"> Đang tải...</DropdownMenuItem>
            ) : notifications.length === 0 ? (
                <DropdownMenuItem disabled className="flex justify-center items-center text-center text-gray-500 py-4 italic">
                    Không có thông báo mới
                </DropdownMenuItem>
            ) : (
                notifications.map((item) => (
                    <DropdownMenuItem
                        key={item.id}
                        // Prevent default close, handle in onClick
                        onSelect={(e) => {
                            e.preventDefault();
                            handleNotificationClick(item);
                            // Optionally close mobile menu if needed here, though dropdown should handle its own closure
                            // setIsMobileMenuOpen(false);
                         }}
                        className={cn(
                            "flex items-start p-3 cursor-pointer data-[highlighted]:bg-gray-100 dark:data-[highlighted]:bg-gray-700 transition-colors duration-150 ease-in-out",
                            !item.read && "bg-blue-50 dark:bg-blue-900/30" // Style unread - removed font-medium to avoid layout shifts maybe
                        )}
                    >
                        {getNotificationIcon(item)}
                        <div className="flex-1 min-w-0"> {/* Added min-w-0 for better text wrapping */}
                            {/* Display Title or Description */}
                            <p className={cn("text-sm leading-snug", !item.read && "font-semibold")}>{item.title}</p> {/* Apply font-semibold conditionally */}
                            <p className="text-xs text-gray-600 dark:text-gray-300 mt-0.5 truncate">{item.description}</p> {/* Added truncate */}
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{timeAgo(item.createdAt)}</p>
                        </div>
                        {/* Unread indicator dot */}
                        {!item.read && (
                            <span className="ml-2 mt-1 h-2 w-2 rounded-full bg-blue-500 flex-shrink-0 self-start"></span>
                        )}
                    </DropdownMenuItem>
                ))
            )}
            <DropdownMenuSeparator />
            {/* Use asChild for proper Link behavior within DropdownMenuItem */}
            <DropdownMenuItem className="justify-center py-2" asChild>
                <Link href="/profile?tab=notifications" className="text-sm text-blue-600 hover:underline w-full flex justify-center items-center" onClick={() => setIsMobileMenuOpen(false)}>
                    Xem tất cả thông báo
                </Link>
            </DropdownMenuItem>
        </DropdownMenuContent>
    );

    // --- Render JSX ---
    return (
        <nav className="bg-white shadow-md border-b border-gray-200 dark:bg-gray-800 dark:border-gray-700 sticky top-0 left-0 right-0 z-[50]">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                <div className="flex h-16 items-center justify-between">
                    {/* Logo Section */}
                    <div className="flex items-center">
                        <Link href={"/"} onClick={() => setIsMobileMenuOpen(false)} className="flex-shrink-0">
                            {/* Ensure Link doesn't cause hydration issues if wrapping Image directly isn't needed */}
                            <Image src={logo} alt="Logo" width={90} height={40} priority style={{ objectFit: 'contain' }} />
                        </Link>
                    </div>

                    {/* Desktop Icons and User Section */}
                    <div className="hidden sm:flex sm:items-center sm:space-x-1 md:space-x-2">
                        {isLoadingSession ? (
                            // Simple placeholder while checking session
                            <div className="flex items-center space-x-2">
                                <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse"></div>
                                <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse"></div>
                            </div>
                        ) : isLoggedIn && user ? (
                            // Logged-in state
                            <>
                                {/* Notifications Dropdown */}
                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        {/* Single Button child - This is correct */}
                                        <Button variant="ghost" size="icon" className="relative rounded-full hover:bg-gray-100 dark:hover:bg-gray-700">
                                            {unreadNotificationsCount > 0 && (
                                                <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                                                </span>
                                            )}
                                            <Bell className="h-5 w-5 text-gray-500 dark:text-gray-300" />
                                            <span className="sr-only">View notifications</span>
                                        </Button>
                                    </DropdownMenuTrigger>
                                    {renderNotificationDropdown()}
                                </DropdownMenu>

                                {/* User Dropdown */}
                                <DropdownMenu>
                                      <DropdownMenuTrigger asChild>
                                        {/* Single Button child - This is correct */}
                                        <Button variant="ghost" className="relative rounded-full p-0.5 hover:bg-gray-100 dark:hover:bg-gray-700 h-9 w-9 flex items-center justify-center">
                                              <span className="sr-only">Open user menu</span>
                                              <Avatar className="h-8 w-8">
                                                  <AvatarImage src={user.profilePic || undefined} alt={user.username || "User Avatar"} />
                                                   <AvatarFallback className="bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 text-xs font-medium">
                                                        { user.firstName?.charAt(0)?.toUpperCase() || user.username?.charAt(0)?.toUpperCase() || user.email?.charAt(0)?.toUpperCase() || 'U' }
                                                   </AvatarFallback>
                                              </Avatar>
                                         </Button>
                                      </DropdownMenuTrigger>
                                      <DropdownMenuContent align="end" className="w-56 shadow-xl">
                                          <DropdownMenuLabel className="font-normal">
                                              <div className="flex flex-col space-y-1">
                                                  <p className="text-sm font-medium leading-none truncate"> { `${user.firstName || ''} ${user.lastName || ''}`.trim() || user.username || 'Người dùng' } </p>
                                                  <p className="text-xs leading-none text-muted-foreground truncate"> {user.email} </p>
                                              </div>
                                          </DropdownMenuLabel>
                                          <DropdownMenuSeparator />
                                          {/* Use asChild for proper Link behavior */}
                                          <DropdownMenuItem asChild>
                                               {/* Single Link child - This is correct */}
                                               <Link href="/profile" className="cursor-pointer flex items-center w-full">
                                                   <User className="mr-2 h-4 w-4" />
                                                   <span>Hồ sơ</span>
                                               </Link>
                                          </DropdownMenuItem>
                                          {/* <DropdownMenuItem asChild> <Link href="/settings" className="cursor-pointer flex items-center w-full"> <Settings className="mr-2 h-4 w-4" /> <span>Cài đặt</span> </Link> </DropdownMenuItem> */}
                                          <DropdownMenuSeparator />
                                          {/* Use onSelect for actions within DropdownMenuItem */}
                                          <DropdownMenuItem
                                               onSelect={(e) => { e.preventDefault(); handleLogout(); }}
                                               className="cursor-pointer text-red-600 focus:bg-red-50 focus:text-red-600 dark:focus:bg-red-900/50 dark:focus:text-red-400 flex items-center w-full"
                                           >
                                               <LogOut className="mr-2 h-4 w-4" />
                                               <span>Đăng xuất</span>
                                           </DropdownMenuItem>
                                      </DropdownMenuContent>
                                </DropdownMenu>
                            </>
                        ) : (
                             // Logged-out state
                             // Use asChild for proper Link behavior
                            <Button asChild size="sm">
                                {/* Single Link child - This is correct */}
                                <Link href="/auth/login">Đăng nhập</Link>
                            </Button>
                        )}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex sm:hidden items-center">
                        <Button variant="ghost" size="icon" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="rounded-full">
                            <span className="sr-only">Open main menu</span>
                            {isMobileMenuOpen ? <X className="block h-6 w-6" aria-hidden="true" /> : <Menu className="block h-6 w-6" aria-hidden="true" />}
                        </Button>
                    </div>
                </div>

                {/* Mobile Menu Dropdown */}
                {isMobileMenuOpen && (
                    <div className="sm:hidden absolute top-16 inset-x-0 z-[49] bg-white dark:bg-gray-800 shadow-lg border-t dark:border-gray-700 animate-slide-down">
                        <div className="px-2 pt-2 pb-3 space-y-1">
                           {isLoadingSession ? (
                                <div className="text-center p-4 text-gray-500 dark:text-gray-400 italic">Đang tải...</div>
                           ) : isLoggedIn && user ? (
                                <>
                                    {/* Mobile User Info */}
                                    <div className="px-3 py-3 border-b dark:border-gray-600 mb-2">
                                        <div className="flex items-center space-x-3">
                                            <Avatar className="h-10 w-10 flex-shrink-0">
                                                <AvatarImage src={user.profilePic || undefined} alt={user.username || "Avatar"}/>
                                                <AvatarFallback className="bg-gray-300 dark:bg-gray-600 text-sm font-medium">
                                                    { user.firstName?.charAt(0)?.toUpperCase() || user.username?.charAt(0)?.toUpperCase() || user.email?.charAt(0)?.toUpperCase() || 'U' }
                                                </AvatarFallback>
                                            </Avatar>
                                            <div className="min-w-0 flex-1">
                                                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{`${user.firstName || ''} ${user.lastName || ''}`.trim() || user.username || 'Người dùng'}</p>
                                                <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{user.email}</p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Mobile Icons (Notification only for now) */}
                                    <div className="flex justify-start items-center px-2 py-2 border-b dark:border-gray-600">
                                        <DropdownMenu>
                                            <DropdownMenuTrigger asChild>
                                                {/* Single Button child - This is correct */}
                                                <Button variant="ghost" size="sm" className="relative flex items-center justify-start text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md w-full px-3 py-2">
                                                    {unreadNotificationsCount > 0 && (
                                                        <span className="absolute top-1 right-2 flex h-2.5 w-2.5">
                                                             <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                                             <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                                                        </span>
                                                    )}
                                                    <Bell className="mr-3 h-5 w-5 flex-shrink-0" />
                                                    <span className="text-sm font-medium">Thông báo</span>
                                                </Button>
                                            </DropdownMenuTrigger>
                                            {renderNotificationDropdown()}
                                        </DropdownMenu>
                                        {/* Add other icons/links here if needed */}
                                    </div>

                                    {/* Mobile Links */}
                                    <Link
                                        href="/profile"
                                        className="flex items-center px-3 py-2 rounded-md text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                        onClick={()=>setIsMobileMenuOpen(false)}
                                    >
                                        <User className="mr-3 h-5 w-5 flex-shrink-0" /> Hồ sơ
                                    </Link>
                                    {/* Mobile Logout Button */}
                                    <Button
                                        variant="ghost"
                                        onClick={() => { handleLogout(); setIsMobileMenuOpen(false); }} // Close menu on click
                                        className="w-full justify-start flex items-center px-3 py-2 rounded-md text-sm font-medium text-red-600 hover:bg-red-50 hover:text-red-700 dark:hover:bg-red-900/30 dark:hover:text-red-400"
                                     >
                                        <LogOut className="mr-3 h-5 w-5 flex-shrink-0" /> Đăng xuất
                                    </Button>
                                </>
                           ) : (
                                // Mobile Logged Out Button
                                <Link
                                    href="/auth/login"
                                    className="block w-full px-3 py-2 rounded-md text-base font-medium text-center text-white bg-blue-600 hover:bg-blue-700"
                                    onClick={()=>setIsMobileMenuOpen(false)}
                                >
                                    Đăng nhập
                                </Link>
                           )}
                        </div>
                    </div>
                )}
            </div>
             {/* Animation Style */}
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