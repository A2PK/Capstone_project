// src/pages/profile.tsx (hoặc đường dẫn tương ứng)
"use client";

import React, { useEffect, useState, ChangeEvent, FormEvent, useCallback } from "react";
import Link from "next/link";
import { format, parseISO, isValid as isValidDate } from 'date-fns';
import { useSearchParams, useRouter } from "next/navigation";
import PageLoader from "@/components/pageloader";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";

import { UserCircle, Mail, MapPin, Briefcase, Calendar, AlertCircle, Loader2, Bell, Info, Check, Camera } from "lucide-react";

// --- Types ---
import type { User } from '@/types/user';
import type { Notification as NotificationData } from '@/types/notification';
import type { FileInfo } from "@/types/file";

// --- API Libs ---
import { updateUser, getUserById } from '@/lib/user'; // Import getUserById
import { findNotificationsByUserId, markAsReadNotifications, getAllNotifications } from '@/lib/notification';
import { uploadFile } from "@/lib/file";
// Giả sử generateProxyUrl được export từ đây hoặc một file utils khác
// Nếu nó ở lib/article, bạn có thể cần di chuyển nó đến vị trí chung hơn
import { generateProxyUrl } from "@/lib/article"; // <<--- ĐỔI PATH NẾU CẦN

// --- Hooks & Utils ---
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

// --- Components ---
function ErrorDisplay({ message }: { message: string }) {
  return (
    <div className="container mx-auto py-10 flex justify-center items-center">
      <Card className="w-full max-w-md border-red-500">
        <CardHeader className="flex flex-row items-center gap-2 text-red-600">
          <AlertCircle className="h-6 w-6" />
          <CardTitle>Lỗi</CardTitle>
        </CardHeader>
        <CardContent>
          <p>{message}</p>
          <p className="mt-2 text-sm text-muted-foreground">Vui lòng thử lại sau hoặc liên hệ quản trị viên.</p>
        </CardContent>
        <CardFooter>
          <Button variant="outline" onClick={() => window.location.reload()}>Tải lại trang</Button>
        </CardFooter>
      </Card>
    </div>
  );
}

function timeAgo(dateString: string): string {
  try {
    const date = parseISO(dateString);
    if (!isValidDate(date)) return dateString;
    const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    let interval = seconds / 31536000; if (interval > 1) return `${Math.floor(interval)} năm`;
    interval = seconds / 2592000; if (interval > 1) return `${Math.floor(interval)} tháng`;
    interval = seconds / 86400; if (interval > 1) return `${Math.floor(interval)} ngày`;
    interval = seconds / 3600; if (interval > 1) return `${Math.floor(interval)} giờ`;
    interval = seconds / 60; if (interval > 1) return `${Math.floor(interval)} phút`;
    return seconds < 10 ? `vài giây` : `${Math.floor(seconds)} giây`;
  } catch { return dateString; }
}

// --- Form Data Interfaces ---
interface ProfileFormData {
  firstName: string;
  lastName: string;
  address: string;
  age: string;
}

interface AccountFormData {
  email: string;
  phone: string;
}

// --- Main Component ---
export default function ProfilePage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { toast } = useToast();

  // --- State Variables ---
  const [isLoading, setIsLoading] = useState(true); // Loading ban đầu
  const [isRefetching, setIsRefetching] = useState(false); // Loading khi refetch sau update
  const initialTab = searchParams.get('tab') || 'profile';
  const [activeTab, setActiveTab] = useState(initialTab);
  const [isSaving, setIsSaving] = useState(false); // Trạng thái đang lưu (upload, gọi API update)
  const [userData, setUserData] = useState<User | null>(null); // Dữ liệu user đầy đủ
  const [userId, setUserId] = useState<string | null>(null); // Lưu userId từ session
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null); // Token xác thực
  const [isMarkingAllAsRead, setIsMarkingAllAsRead] = useState(false);
  // Profile Form State
  const [profileForm, setProfileForm] = useState<ProfileFormData>({ firstName: '', lastName: '', address: '', age: '' });
  const [profilePicFile, setProfilePicFile] = useState<File | null>(null);

  // Account Form State
  const [accountForm, setAccountForm] = useState<AccountFormData>({ email: '', phone: '' });

  // Security Form State
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // Notifications State
  const [notifications, setNotifications] = useState<NotificationData[]>([]);
  const [isLoadingNotifications, setIsLoadingNotifications] = useState(false);
  const [notificationError, setNotificationError] = useState<string | null>(null);

  // --- Function to load full user data and set states ---
  const loadAndSetUserData = useCallback(async (id: string, authToken: string) => {
    setIsLoading(true); // Bật loading khi bắt đầu fetch user data
    setError(null);
    try {
      console.log(`Workspaceing full user data for ID: ${id}`);
      const fullUserData = await getUserById(id);
      setUserData(fullUserData);
      console.log(fullUserData)
      // Initialize forms with fetched data
      setProfileForm({
        firstName: fullUserData.firstName ?? '',
        lastName: fullUserData.lastName ?? '',
        address: fullUserData.address ?? '',
        age: fullUserData.age?.toString() ?? ''
      });
      setAccountForm({
        email: fullUserData.email ?? '',
        phone: fullUserData.phone ?? ''
      });
      console.log("Full user data loaded and forms initialized.");
    } catch (e: any) {
      console.error("ProfilePage: Failed to fetch user data by ID:", e);
      setError(e.message || "Đã xảy ra lỗi khi tải thông tin chi tiết người dùng.");
      setUserData(null); // Reset userData nếu lỗi
    } finally {
      setIsLoading(false); // Tắt loading khi fetch xong (kể cả lỗi)
    }
  }, []); // Không có dependencies vì id và token được truyền vào

  // --- Refetch Function (sau khi update) ---
  const refetchAndSetUserData = useCallback(async () => {
    // Dùng userId và token từ state
    if (!userId || !token) {
      console.warn("Cannot refetch user data: user ID or token missing.");
      toast({ title: "Lỗi", description: "Thiếu thông tin để tải lại dữ liệu.", variant: "destructive" });
      return;
    }
    setIsRefetching(true);
    try {
      console.log(`Refetching user data for ID: ${userId}`);
      const freshUserData = await getUserById(userId);
      setUserData(freshUserData);

      // Re-initialize forms with fresh data
      setProfileForm({
        firstName: freshUserData.firstName ?? '', lastName: freshUserData.lastName ?? '', address: freshUserData.address ?? '', age: freshUserData.age?.toString() ?? ''
      });
      setAccountForm({ email: freshUserData.email ?? '', phone: freshUserData.phone ?? '' });
      setProfilePicFile(null); // Reset file input

      console.log("User data refetched and state updated after modification.");
    } catch (refetchError: any) {
      console.error("ProfilePage: Failed to refetch user data after update:", refetchError);
      toast({
        title: "Lỗi đồng bộ",
        description: "Đã cập nhật thành công, nhưng không thể tải lại dữ liệu mới nhất. Dữ liệu hiển thị có thể chưa chính xác.",
        variant: "warning",
      });
    } finally {
      setIsRefetching(false);
    }
  }, [userId, token, toast]); // Dependencies: userId, token, toast

  // --- Effects ---
  // 1. Fetch Session Info (Token and User ID) on Mount
  useEffect(() => {
    const fetchSession = async () => {
      setIsLoading(true); // Bắt đầu loading tổng thể
      setError(null);
      setToken(null);
      setUserId(null);
      setUserData(null); // Reset user data

      try {
        const response = await fetch('/api/auth/session');
        if (!response.ok) {
          if (response.status === 401) { // Handle unauthorized specifically
            setError("Phiên đăng nhập hết hạn hoặc không hợp lệ. Vui lòng đăng nhập lại.");
          } else {
            throw new Error(`Lỗi mạng khi lấy session: ${response.status}`);
          }
          return; // Stop if session fetch fails critically
        }
        const data: { isLoggedIn: boolean; user: { id: string } | null; token: string | null } = await response.json();

        if (data.isLoggedIn && data.user?.id && data.token) {
          console.log("Session loaded successfully. Token and User ID obtained.");
          setToken(data.token);
          setUserId(data.user.id);
          // *** Gọi hàm để fetch dữ liệu user đầy đủ ***
          await loadAndSetUserData(data.user.id, data.token);
        } else {
          setError("Bạn chưa đăng nhập."); // Hoặc "Không thể lấy thông tin phiên."
        }
      } catch (e: any) {
        console.error("ProfilePage: Failed to fetch session info:", e);
        setError(e.message || "Đã xảy ra lỗi khi kiểm tra phiên đăng nhập.");
      } finally {
        // setIsLoading(false); // Loading sẽ được tắt bởi loadAndSetUserData hoặc ở đây nếu loadAndSetUserData không được gọi
        if (!userId) setIsLoading(false); // Tắt loading nếu không lấy được userId để gọi loadAndSetUserData
      }
    };
    fetchSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Chỉ chạy một lần khi mount

  // 2. Fetch Notifications when UserData and Token are available
  useEffect(() => {
    if (userData && token) { // Chỉ fetch khi có user data đầy đủ và token
      const fetchNotifications = async () => {
        setIsLoadingNotifications(true);
        setNotificationError(null);
        try {
          const fetchedNotifications = await getAllNotifications(token, { limit: 50, sortBy: 'created_at', sortDesc: true });
          setNotifications(fetchedNotifications || []);
        } catch (err: any) {
          console.error("ProfilePage: Failed to fetch notifications:", err);
          setNotificationError(err.message || "Không thể tải danh sách thông báo.");
          setNotifications([]);
        } finally {
          setIsLoadingNotifications(false);
        }
      };
      fetchNotifications();
    } else {
      setNotifications([]); // Clear notifications if no user or token
    }
  }, [userData, token]); // Chạy lại khi userData hoặc token thay đổi

  // --- Event Handlers ---
  const handleProfileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setProfileForm(prev => ({ ...prev, [id]: value }));
  };
  const handleProfilePicChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setProfilePicFile(e.target.files[0]);
    } else {
      setProfilePicFile(null);
    }
  };
  const handleAccountChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setAccountForm(prev => ({ ...prev, [id]: value }));
  };
  const handlePasswordChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    if (id === 'newPassword') setNewPassword(value);
    else if (id === 'confirmPassword') setConfirmPassword(value);
  };
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    const newUrl = `/profile?tab=${value}`;
    router.push(newUrl, { scroll: false });
  };

  // --- Form Submission Handlers ---
  const handleSaveProfile = async (e: FormEvent) => {
    e.preventDefault();
    // Cần userId và token để thực hiện update
    if (!userId || !token) {
      toast({ title: "Lỗi", description: "Thiếu thông tin xác thực để cập nhật.", variant: "destructive" });
      return;
    }
    setIsSaving(true);
    // Lấy URL hiện tại từ state userData (đã được fetch bởi getUserById)
    let currentProfilePicUrl: string | null = userData?.profilePic ?? null;
    let newProfilePicUrl = currentProfilePicUrl; // Bắt đầu với URL hiện tại

    try {
      // 1. Upload ảnh (nếu có)
      if (profilePicFile) {
        console.log("Uploading new profile picture...");
        const picFormData = { type: 'profile_picture', name: profilePicFile.name, file: profilePicFile };
        try {
          const uploadedFileInfo: FileInfo = await uploadFile(token, picFormData);
          newProfilePicUrl = uploadedFileInfo.url; // Cập nhật URL nếu upload thành công
          console.log("Profile picture uploaded:", newProfilePicUrl);
          toast({ title: "Thông báo", description: "Ảnh đại diện đã được tải lên.", variant: "default" }); // Dùng info hoặc success
        } catch (uploadError: any) {
          console.error("ProfilePage: Failed to upload profile picture:", uploadError);
          toast({
            title: "Lỗi tải ảnh lên",
            description: uploadError.message || "Không thể tải ảnh đại diện mới. Thông tin khác vẫn sẽ được lưu.",
            variant: "destructive",
          });
          // Không dừng lại, vẫn tiếp tục lưu thông tin khác với ảnh cũ (newProfilePicUrl = currentProfilePicUrl)
        }
      }

      // 2. Chuẩn bị data update
      const updateData: Partial<User> = {
        firstName: profileForm.firstName.trim(),
        lastName: profileForm.lastName.trim(),
        address: profileForm.address.trim(),
        age: profileForm.age ? parseInt(profileForm.age, 10) || null : null,
        profilePic: newProfilePicUrl, // Luôn gửi URL (mới hoặc cũ/null)
      };

      // 3. Gọi API update
      console.log("Updating user profile data:", updateData);
      const status = await updateUser(updateData, userId); // Dùng userId từ state

      if (status >= 200 && status < 300) {
        toast({ title: "Thành công", description: "Thông tin cá nhân đã được cập nhật.", variant: "success" });
        // 4. Refetch dữ liệu đầy đủ
        await refetchAndSetUserData();
        window.location.reload();
      } else {
        throw new Error(`Lỗi cập nhật thông tin từ server: ${status}`);
      }
    } catch (err: any) {
      console.error("ProfilePage: Failed to save profile:", err);
      toast({
        title: "Lỗi cập nhật",
        description: err.message || "Không thể cập nhật thông tin cá nhân.",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };
  const handleMarkAllAsRead = async () => {
    if (!token) {
      toast({ variant: "destructive", title: "Lỗi", description: "Thiếu thông tin xác thực để thực hiện hành động này." });
      return;
    }

    const unreadNotifications = notifications.filter(n => !n.read);
    if (unreadNotifications.length === 0) {
      // Không có gì để đánh dấu
      return;
    }

    const unreadIds = unreadNotifications.map(n => n.id);

    setIsMarkingAllAsRead(true); // Bắt đầu loading

    // --- Optimistic UI Update ---
    // Lưu trạng thái hiện tại để có thể hoàn tác nếu lỗi
    const originalNotifications = [...notifications];
    // Cập nhật state ngay lập tức để UI phản hồi nhanh
    setNotifications(prev =>
      prev.map(n => (unreadIds.includes(n.id) ? { ...n, read: true } : n))
    );

    try {
      console.log(`Attempting to mark ${unreadIds.length} notifications as read.`);
      await markAsReadNotifications(token, unreadIds); // Gọi API
      toast({ variant: "success", title: "Thành công", description: "Tất cả thông báo chưa đọc đã được đánh dấu." });
      // Thành công, không cần làm gì thêm vì UI đã được cập nhật
      window.location.reload();
    } catch (error: any) {
      console.error("Failed to mark all notifications as read:", error);
      toast({ variant: "destructive", title: "Lỗi", description: error.message || "Không thể đánh dấu tất cả thông báo là đã đọc. Vui lòng thử lại." });
      // --- Revert Optimistic Update ---
      // Hoàn tác lại trạng thái trên UI nếu API gọi thất bại
      setNotifications(originalNotifications);
    } finally {
      setIsMarkingAllAsRead(false); // Kết thúc loading
    }
  };
  const handleSaveAccount = async (e: FormEvent) => {
    e.preventDefault();
    if (!userId || !token) { /* validation */ return; }
    setIsSaving(true);
    const updateData: Partial<User> = {
      email: accountForm.email.trim(),
      phone: accountForm.phone.trim(),
    };
    try {
      const status = await updateUser(updateData, userId);
      if (status >= 200 && status < 300) {
        toast({ title: "Thành công", description: "Thông tin tài khoản đã được cập nhật.", variant: "success" });
        await refetchAndSetUserData(); // Refetch
      } else {
        throw new Error(`Lỗi từ server: ${status}`);
      }
    } catch (err: any) {
      console.error("ProfilePage: Failed to update account:", err);
      toast({ title: "Lỗi cập nhật", description: err.message || "Không thể cập nhật thông tin tài khoản.", variant: "destructive" });
    } finally {
      setIsSaving(false);
    }
  };

  const handleUpdatePassword = async (e: FormEvent) => {
    e.preventDefault();
    if (!userId || !token) { /* validation */ return; }
    if (!newPassword || newPassword !== confirmPassword) { /* validation */
      if (!newPassword) { toast({ title: "Thiếu thông tin", description: "Vui lòng nhập mật khẩu mới.", variant: "destructive" }); return; }
      if (newPassword !== confirmPassword) { toast({ title: "Xác nhận không khớp", description: "Mật khẩu mới và mật khẩu xác nhận không giống nhau.", variant: "destructive" }); return; }
      return;
    }
    setIsSaving(true);
    const updateData: Partial<User> = { password: newPassword };
    try {
      const status = await updateUser(updateData, userId);
      if (status >= 200 && status < 300) {
        toast({ title: "Thành công", description: "Mật khẩu của bạn đã được cập nhật.", variant: "success" });
        setNewPassword('');
        setConfirmPassword('');
        await refetchAndSetUserData(); // Refetch
      } else {
        throw new Error(`Lỗi từ server: ${status}`);
      }
    } catch (err: any) {
      console.error("ProfilePage: Failed to update password:", err);
      toast({ title: "Lỗi cập nhật mật khẩu", description: err.message || "Không thể cập nhật mật khẩu. Vui lòng thử lại.", variant: "destructive" });
    } finally {
      setIsSaving(false);
    }
  };

  // --- Notification Handlers ---
  const handleMarkOneAsRead = async (notificationId: string) => {
    if (!token) return;
    const notification = notifications.find(n => n.id === notificationId);
    if (!notification || notification.read) return;
    setNotifications(prev => prev.map(n => n.id === notificationId ? { ...n, read: true } : n)); // Optimistic update
    try {
      await markAsReadNotifications(token, [notificationId]);
    } catch (error) {
      console.error(`Failed to mark notification ${notificationId} as read on server:`, error);
      toast({ variant: "destructive", title: "Lỗi", description: "Không thể cập nhật trạng thái thông báo." });
      setNotifications(prev => prev.map(n => n.id === notificationId ? { ...n, read: false } : n)); // Revert on error
    }
  };
  const getNotificationIcon = (notification: NotificationData) => {
    const titleLower = notification.title.toLowerCase(); const descLower = notification.description.toLowerCase();
    if (titleLower.includes('cảnh báo') || descLower.includes('cảnh báo') || titleLower.includes('alert') || titleLower.includes('warning')) return <AlertCircle className="h-5 w-5 text-red-500 mr-3 flex-shrink-0" />;
    if (titleLower.includes('cập nhật') || descLower.includes('cập nhật') || titleLower.includes('update') || titleLower.includes('huấn luyện')) return <Info className="h-5 w-5 text-blue-500 mr-3 flex-shrink-0" />;
    if (titleLower.includes('thành công') || descLower.includes('thành công') || titleLower.includes('hoàn thành')) return <Check className="h-5 w-5 text-green-500 mr-3 flex-shrink-0" />;
    return <Bell className="h-5 w-5 text-gray-400 mr-3 flex-shrink-0" />;
  };

  // --- Render Logic ---
  if (isLoading) return <PageLoader message="Đang tải thông tin người dùng..." />; // Cập nhật message
  if (error) return <ErrorDisplay message={error} />;
  // Sau khi loading xong, nếu không có user data (do lỗi fetch hoặc chưa đăng nhập)
  if (!userData) {
    return (
      <div className="container mx-auto py-10 text-center">
        {/* Hiển thị lỗi nếu có, hoặc thông báo đăng nhập */}
        {error ? <ErrorDisplay message={error} /> : (
          <p>Vui lòng <Link href="/auth/login" className="text-blue-600 hover:underline">đăng nhập</Link> để xem thông tin.</p>
        )}
      </div>
    );
  }

  // --- Derived data for display (khi userData đã tồn tại) ---
  const fallbackName = `${userData.firstName?.[0] ?? ''}${userData.lastName?.[0] ?? ''}`.toUpperCase() || 'U';
  const joinDate = userData.createdAt ? format(new Date(userData.createdAt), 'dd/MM/yyyy') : 'N/A';
  const displayRole = userData.role === 'manager' ? 'Tổng quản trị' : (userData.role === 'admin' ? 'Quản trị viên' : (userData.role === 'officer' ? 'Nhân viên' : userData.role));
  // Sử dụng generateProxyUrl cho ảnh đại diện (nếu có)
  const profilePicSrc = generateProxyUrl(userData.profilePic ? userData.profilePic : "")
  // Xác định URL cho ảnh preview trong form
  console.log("profilePicSrc", profilePicSrc, "mmm")
  const profilePicPreviewSrc = profilePicFile ? URL.createObjectURL(profilePicFile) : profilePicSrc;

  // --- JSX ---
  return (
    <div className="container mx-auto py-10 relative">
      {/* Overlay loading khi đang refetch sau update */}
      {isRefetching && (
        <div className="absolute inset-0 bg-white/70 dark:bg-black/70 flex items-center justify-center z-50 rounded-lg">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      )}

      {/* Làm mờ UI khi đang refetch */}
      <div className={cn("flex flex-col gap-8 md:flex-row", isRefetching && "opacity-50 pointer-events-none")}>
        {/* Left Sidebar - User Info */}
        <div className="w-full md:w-1/3 lg:w-1/4">
          <Card>
            <CardHeader className="flex flex-col items-center space-y-4 pt-6">
              <Avatar className="h-24 w-24 border">
                {/* Dùng profilePicSrc đã qua generateProxyUrl */}
                <AvatarImage src={generateProxyUrl(userData?.profilePic ? userData.profilePic : "")} alt={`Ảnh đại diện ${userData.firstName}`} />
                <AvatarFallback className="text-3xl">{fallbackName}</AvatarFallback>
              </Avatar>
              <div className="space-y-1 text-center">
                <h2 className="text-2xl font-semibold">{`${userData.firstName || ''} ${userData.lastName || ''}`.trim() || userData.username}</h2>
                <p className="text-sm text-muted-foreground">@{userData.username}</p>
              </div>
            </CardHeader>
            <CardContent className="mt-4 space-y-3 text-sm">
              <div className="flex items-start gap-3">
                <Mail className="h-4 w-4 flex-shrink-0 mt-1 text-muted-foreground" />
                <span className="break-all">{userData.email}</span>
              </div>
              {userData.phone && (
                <div className="flex items-start gap-3">
                  {/* Nên dùng icon Phone nếu có */}
                  <Mail className="h-4 w-4 flex-shrink-0 mt-1 text-muted-foreground" />
                  <span>{userData.phone}</span>
                </div>
              )}
              {userData.address && (
                <div className="flex items-start gap-3">
                  <MapPin className="h-4 w-4 flex-shrink-0 mt-1 text-muted-foreground" />
                  <span>{userData.address}</span>
                </div>
              )}
              <div className="flex items-start gap-3">
                <Briefcase className="h-4 w-4 flex-shrink-0 mt-1 text-muted-foreground" />
                <span>{displayRole}</span>
              </div>
              <div className="flex items-start gap-3">
                <Calendar className="h-4 w-4 flex-shrink-0 mt-1 text-muted-foreground" />
                <span>Tham gia ngày {joinDate}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Content Area - Tabs */}
        <div className="flex-1">
          <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="profile">Thông tin cá nhân</TabsTrigger>
              <TabsTrigger value="account">Tài khoản</TabsTrigger>
              <TabsTrigger value="notifications">Thông báo</TabsTrigger>
              <TabsTrigger value="security">Bảo mật</TabsTrigger>
            </TabsList>

            {/* Profile Tab */}
            <TabsContent value="profile" className="mt-6">
              <form onSubmit={handleSaveProfile}>
                <Card>
                  <CardHeader>
                    <CardTitle>Thông tin cá nhân</CardTitle>
                    <CardDescription>Cập nhật thông tin cá nhân và ảnh đại diện của bạn.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    {/* Profile Picture Upload */}
                    <div className="space-y-2">
                      <Label htmlFor="profilePicInput">Ảnh đại diện</Label>
                      <div className="flex items-center gap-4">
                        <Avatar className="h-16 w-16 border">
                          {/* Dùng profilePicPreviewSrc cho ảnh xem trước */}
                          <AvatarImage src={profilePicPreviewSrc} alt="Ảnh xem trước" />
                          <AvatarFallback><Camera className="h-6 w-6 text-muted-foreground" /></AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                          <Input
                            id="profilePicInput"
                            type="file"
                            accept="image/*"
                            onChange={handleProfilePicChange}
                            disabled={isSaving || isRefetching} // Disable khi đang lưu hoặc refetch
                            className="h-13 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary-foreground file:text-primary hover:file:bg-muted"
                          />
                          {profilePicFile && <p className="mt-1 text-sm text-muted-foreground">Đã chọn: {profilePicFile.name}</p>}
                          {!profilePicFile && userData.profilePic && <p className="mt-1 text-xs text-gray-500">Ảnh hiện tại. Chọn file mới để thay thế.</p>}
                          {!profilePicFile && !userData.profilePic && <p className="mt-1 text-xs text-gray-500">Chưa có ảnh đại diện. Chọn file để tải lên.</p>}
                        </div>
                      </div>
                    </div>

                    {/* Other Profile Fields */}
                    <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label htmlFor="firstName">Tên</Label>
                        <Input id="firstName" value={profileForm.firstName} onChange={handleProfileChange} disabled={isSaving || isRefetching} placeholder="Nhập tên" />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="lastName">Họ</Label>
                        <Input id="lastName" value={profileForm.lastName} onChange={handleProfileChange} disabled={isSaving || isRefetching} placeholder="Nhập họ" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="username">Username</Label>
                      <Input id="username" defaultValue={userData.username} readOnly disabled />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="role">Vai trò</Label>
                      <Input id="role" defaultValue={displayRole} readOnly disabled />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="address">Địa chỉ</Label>
                      <Input id="address" value={profileForm.address} onChange={handleProfileChange} disabled={isSaving || isRefetching} placeholder="Nhập địa chỉ" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="age">Tuổi</Label>
                      <Input id="age" type="number" value={profileForm.age} onChange={handleProfileChange} disabled={isSaving || isRefetching} placeholder="Nhập tuổi" min="0" />
                    </div>
                  </CardContent>
                  <CardFooter className="border-t px-6 py-4">
                    <Button type="submit" disabled={isSaving || isRefetching} className="ml-auto">
                      {(isSaving || isRefetching) && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      {isSaving ? 'Đang lưu...' : (isRefetching ? 'Đang đồng bộ...' : 'Lưu thay đổi')}
                    </Button>
                  </CardFooter>
                </Card>
              </form>
            </TabsContent>

            {/* Account Tab */}
            <TabsContent value="account" className="mt-6">
              <form onSubmit={handleSaveAccount}>
                <Card>
                  <CardHeader>
                    <CardTitle>Tài khoản</CardTitle>
                    <CardDescription>Quản lý thông tin liên lạc của bạn.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="space-y-2">
                      <Label htmlFor="email">Email</Label>
                      <Input id="email" type="email" value={accountForm.email} onChange={handleAccountChange} disabled={isSaving || isRefetching} placeholder="your@email.com" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="phone">Số điện thoại</Label>
                      <Input id="phone" value={accountForm.phone} onChange={handleAccountChange} disabled={isSaving || isRefetching} placeholder="Nhập số điện thoại" />
                    </div>
                  </CardContent>
                  <CardFooter className="border-t px-6 py-4">
                    <Button type="submit" disabled={isSaving || isRefetching} className="ml-auto">
                      {(isSaving || isRefetching) && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      {isSaving ? 'Đang lưu...' : (isRefetching ? 'Đang đồng bộ...' : 'Lưu thay đổi')}
                    </Button>
                  </CardFooter>
                </Card>
              </form>
            </TabsContent>

            {/* Notifications Tab */}
            <TabsContent value="notifications" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Thông báo của bạn</CardTitle>
                  <CardDescription>Danh sách các thông báo gần đây.</CardDescription>
                </CardHeader>
                <CardContent>
                  {isLoadingNotifications ? (
                    <div className="flex justify-center items-center p-6">
                      <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                    </div>
                  ) : notificationError ? (
                    <p className="text-red-600 text-center">{notificationError}</p>
                  ) : notifications.length === 0 ? (
                    <p className="text-muted-foreground text-center italic py-4">Không có thông báo nào.</p>
                  ) : (
                    <ScrollArea className="h-[400px] pr-4"> {/* Adjust height as needed */}
                      <ul className="space-y-4">
                        {notifications.map((noti) => (
                          <li
                            key={noti.id}
                            className={cn(
                              "flex items-start gap-3 p-3 border rounded-md transition-colors",
                              noti.read ? "border-gray-200 dark:border-gray-700" : "border-blue-300 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-800"
                            )}
                          >
                            <div className="mt-1">
                              {getNotificationIcon(noti)}
                            </div>
                            <div className="flex-1">
                              <p className={cn("font-semibold text-sm", !noti.read && "text-blue-800 dark:text-blue-200")}>{noti.title}</p>
                              <p className="text-sm text-muted-foreground mt-0.5">{noti.description}</p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1.5">{timeAgo(noti.createdAt)} trước</p>
                            </div>
                            {!noti.read && (
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-auto px-2 py-1 text-xs"
                                onClick={() => handleMarkOneAsRead(noti.id)}
                                title="Đánh dấu đã đọc"
                              >
                                <Check className="h-4 w-4" />
                              </Button>
                            )}
                          </li>
                        ))}
                      </ul>
                    </ScrollArea>
                  )}
                </CardContent>
                <CardFooter className="border-t px-6 py-4">
                  <Button
                    variant="outline"
                    size="sm"
                    // Vô hiệu hóa nếu đang tải thông báo, không có thông báo chưa đọc, hoặc đang trong quá trình đánh dấu tất cả
                    disabled={isLoadingNotifications || notifications.filter(n => !n.read).length === 0 || isMarkingAllAsRead}
                    onClick={handleMarkAllAsRead} // Gắn hàm xử lý
                  >
                    {isMarkingAllAsRead ? ( // Hiển thị loader nếu đang xử lý
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Đang xử lý...
                      </>
                    ) : (
                      'Đánh dấu tất cả đã đọc' // Văn bản mặc định
                    )}
                  </Button>
                </CardFooter>
              </Card>
            </TabsContent>

            {/* Security Tab */}
            <TabsContent value="security" className="mt-6">
              <form onSubmit={handleUpdatePassword}>
                <Card>
                  <CardHeader>
                    <CardTitle>Bảo mật</CardTitle>
                    <CardDescription>Thay đổi mật khẩu đăng nhập của bạn.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="space-y-2">
                      <Label htmlFor="newPassword">Mật khẩu mới</Label>
                      <Input
                        id="newPassword"
                        type="password"
                        minLength={8}
                        value={newPassword}
                        onChange={handlePasswordChange}
                        disabled={isSaving || isRefetching}
                        placeholder="Nhập mật khẩu mới"
                        autoComplete="new-password"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="confirmPassword">Nhập lại mật khẩu mới</Label>
                      <Input
                        id="confirmPassword"
                        type="password"
                        minLength={8}
                        value={confirmPassword}
                        onChange={handlePasswordChange}
                        disabled={isSaving || isRefetching}
                        placeholder="Xác nhận mật khẩu mới"
                        autoComplete="new-password"
                      />
                    </div>
                  </CardContent>
                  <CardFooter className="border-t px-6 py-4">
                    <Button type="submit" disabled={isSaving || isRefetching || !newPassword || !confirmPassword} className="ml-auto">
                      {(isSaving || isRefetching) && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      {isSaving ? 'Đang cập nhật...' : (isRefetching ? 'Đang đồng bộ...' : 'Cập nhật mật khẩu')}
                    </Button>
                  </CardFooter>
                </Card>
              </form>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}