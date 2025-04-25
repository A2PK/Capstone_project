"use client";

import React, { useEffect, useState, ChangeEvent, FormEvent } from "react";
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
import { Badge } from "@/components/ui/badge";

import { UserCircle, Mail, MapPin, Briefcase, Calendar, AlertCircle, Loader2, Bell, Info, Check } from "lucide-react";

import type { User } from '@/types/user';
import { Notification as NotificationData } from '@/types/notification';
import { updateUser } from '@/lib/user';
import { findNotificationsByUserId, markAsReadNotifications, getAllNotifications } from '@/lib/notification';
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

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

export default function ProfilePage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const initialTab = searchParams.get('tab') || 'profile';
  const [activeTab, setActiveTab] = useState(initialTab);
  const [isSaving, setIsSaving] = useState(false);
  const [userData, setUserData] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const { toast } = useToast();

  const [profileForm, setProfileForm] = useState<ProfileFormData>({ firstName: '', lastName: '', address: '', age: '' });
  const [accountForm, setAccountForm] = useState<AccountFormData>({ email: '', phone: '' });
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [notifications, setNotifications] = useState<NotificationData[]>([]);
  const [isLoadingNotifications, setIsLoadingNotifications] = useState(false);
  const [notificationError, setNotificationError] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserDataAndToken = async () => {
      setIsLoading(true);
      setError(null);
      setNotificationError(null);
      try {
        const response = await fetch('/api/auth/session');
        if (!response.ok) throw new Error(`Lỗi mạng: ${response.status}`);
        const data: { isLoggedIn: boolean; user: User | null; token: string | null } = await response.json();

        if (data.isLoggedIn && data.user && data.token) {
          setUserData(data.user);
          setToken(data.token);
          setProfileForm({ firstName: data.user.firstName ?? '', lastName: data.user.lastName ?? '', address: data.user.address ?? '', age: data.user.age?.toString() ?? '' });
          setAccountForm({ email: data.user.email ?? '', phone: data.user.phone ?? '' });
        } else {
          //console.warn("ProfilePage: User not logged in or user/token data not found.");
          setError("Bạn chưa đăng nhập hoặc không tìm thấy thông tin người dùng/token.");
        }
      } catch (e: any) {
        //console.error("ProfilePage: Failed to fetch user data/token:", e);
        setError(e.message || "Đã xảy ra lỗi khi tải thông tin.");
      } finally {
        setIsLoading(false);
      }
    };
    fetchUserDataAndToken();
  }, []);
  useEffect(() => {
    if (userData && token) {
      const fetchNotifications = async () => {
        setIsLoadingNotifications(true);
        setNotificationError(null);
        try {
          //console.log(token, userData)
          const fetchedNotifications = await getAllNotifications(token, {
            limit: 50,
            sortBy: 'created_at',
            sortDesc: true,
          });
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
    }
  }, [userData, token]);

  const handleProfileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setProfileForm(prev => ({ ...prev, [id]: value }));
  };
  const handlePasswordChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    if (id === 'newPassword') {
      setNewPassword(value);
    } else if (id === 'confirmPassword') {
      setConfirmPassword(value);
    }
  };
  const handleAccountChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setAccountForm(prev => ({ ...prev, [id]: value }));
  };
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    const newUrl = `/profile?tab=${value}`;
    router.push(newUrl, { scroll: false });
  };
  const handleUpdatePassword = async (e: FormEvent) => {
    e.preventDefault(); // Ngăn form submit theo cách truyền thống
    if (!userData) return; // Đảm bảo userData tồn tại

    // --- Validation ---
    if (!newPassword) {
      toast({
        title: "Thiếu thông tin",
        description: "Vui lòng nhập mật khẩu mới.",
        variant: "destructive",
      });
      return;
    }
    if (newPassword !== confirmPassword) {
      toast({
        title: "Xác nhận không khớp",
        description: "Mật khẩu mới và mật khẩu xác nhận không giống nhau.",
        variant: "destructive",
      });
      return;
    }
    // Optional: Thêm kiểm tra độ phức tạp mật khẩu ở đây nếu muốn
    // Ví dụ: if (newPassword.length < 8) { ... }

    setIsSaving(true); // Bắt đầu trạng thái loading

    const updateData: Partial<User> = {
      password: newPassword // Chỉ gửi trường password
    };

    try {
      // Gọi API updateUser chỉ với trường password
      const status = await updateUser(updateData, userData.id);

      if (status >= 200 && status < 300) {
        toast({
          title: "Thành công",
          description: "Mật khẩu của bạn đã được cập nhật.",
          variant: "success",
        });
        // Reset các trường mật khẩu sau khi thành công
        setNewPassword('');
        setConfirmPassword('');
      } else {
        // Cố gắng đưa ra thông báo lỗi cụ thể hơn dựa trên status code
        let errorMsg = `Lỗi từ server: ${status}`;
        if (status === 400) { // Bad Request (ví dụ: mật khẩu không đủ mạnh theo quy tắc backend)
          errorMsg = "Mật khẩu mới không hợp lệ hoặc không đủ mạnh.";
        } else if (status === 401 || status === 403) { // Unauthorized or Forbidden
          errorMsg = "Bạn không có quyền thực hiện hành động này.";
        }
        throw new Error(errorMsg);
      }
    } catch (err: any) {
      console.error("ProfilePage: Failed to update password:", err);
      toast({
        title: "Lỗi cập nhật mật khẩu",
        description: err.message || "Không thể cập nhật mật khẩu. Vui lòng thử lại.",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false); // Kết thúc trạng thái loading
    }
  };

  const handleSaveProfile = async (e: FormEvent) => {
    e.preventDefault();
    if (!userData) return;
    setIsSaving(true);
    const updateData: Partial<User> = {
      firstName: profileForm.firstName.trim(),
      lastName: profileForm.lastName.trim(),
      address: profileForm.address.trim(),
      age: profileForm.age ? parseInt(profileForm.age, 10) || null : null,
    };

    try {
      const status = await updateUser(updateData, userData.id);
      if (status >= 200 && status < 300) {
        toast({
          title: "Thành công",
          description: "Thông tin cá nhân đã được cập nhật.",
          variant: "success",
        });
        setUserData(prev => prev ? { ...prev, ...updateData } : null);
      } else {
        throw new Error(`Lỗi từ server: ${status}`);
      }
    } catch (err: any) {
      //console.error("ProfilePage: Failed to update profile:", err);
      toast({
        title: "Lỗi cập nhật",
        description: err.message || "Không thể cập nhật thông tin cá nhân.",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };
  const handleSaveAccount = async (e: FormEvent) => {
    e.preventDefault();
    if (!userData) return;
    setIsSaving(true);
    const updateData: Partial<User> = {
      email: accountForm.email.trim(),
      phone: accountForm.phone.trim(),
    };

    try {
      const status = await updateUser(updateData, userData.id);
      if (status >= 200 && status < 300) {
        toast({
          title: "Thành công",
          description: "Thông tin tài khoản đã được cập nhật.",
          variant: "success",
        });
        setUserData(prev => prev ? { ...prev, ...updateData } : null);
      } else {
        throw new Error(`Lỗi từ server: ${status}`);
      }
    } catch (err: any) {
      console.error("ProfilePage: Failed to update account:", err);
      toast({
        title: "Lỗi cập nhật",
        description: err.message || "Không thể cập nhật thông tin tài khoản.",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };
  const handleMarkOneAsRead = async (notificationId: string) => {
    if (!token) return;
    // Find the notification client-side
    const notification = notifications.find(n => n.id === notificationId);
    if (!notification || notification.read) return; // Already read or not found
    setNotifications(prev => prev.map(n => n.id === notificationId ? { ...n, read: true } : n));
    try {
      await markAsReadNotifications(token, [notificationId]);
      //console.log(`Marked notification ${notificationId} as read on server.`);
    } catch (error) {
      //console.error(`Failed to mark notification ${notificationId} as read on server:`, error);
      toast({ variant: "destructive", title: "Lỗi", description: "Không thể cập nhật trạng thái thông báo." });
      setNotifications(prev => prev.map(n => n.id === notificationId ? { ...n, read: false } : n));
    }
  };
  const getNotificationIcon = (notification: NotificationData) => {
    const titleLower = notification.title.toLowerCase(); const descLower = notification.description.toLowerCase();
    if (titleLower.includes('cảnh báo') || descLower.includes('cảnh báo') || titleLower.includes('alert') || titleLower.includes('warning')) return <AlertCircle className="h-5 w-5 text-red-500 mr-3 flex-shrink-0" />;
    if (titleLower.includes('cập nhật') || descLower.includes('cập nhật') || titleLower.includes('update') || titleLower.includes('huấn luyện')) return <Info className="h-5 w-5 text-blue-500 mr-3 flex-shrink-0" />;
    if (titleLower.includes('thành công') || descLower.includes('thành công') || titleLower.includes('hoàn thành')) return <Check className="h-5 w-5 text-green-500 mr-3 flex-shrink-0" />;
    return <Bell className="h-5 w-5 text-gray-400 mr-3 flex-shrink-0" />;
  };
  if (isLoading) return <PageLoader message="Đang tải trang thông tin cá nhân..." />;
  if (error) return <ErrorDisplay message={error} />;
  if (!userData) {
    return (
      <div className="container mx-auto py-10 text-center">
        <p>Vui lòng <Link href="/auth/login" className="text-blue-600 hover:underline">đăng nhập</Link> để xem thông tin.</p>
      </div>
    );
  }

  const fallbackName = `${userData.firstName?.[0] ?? ''}${userData.lastName?.[0] ?? ''}`.toUpperCase() || 'U';
  const joinDate = userData.createdAt ? format(new Date(userData.createdAt), 'dd/MM/yyyy') : 'N/A';
  const displayRole = userData.role === 'manager' ? 'Tổng quản trị' : (userData.role === 'admin' ? 'Quản trị viên' : (userData.role === 'officer' ? 'Nhân viên' : userData.role)); // Hiển thị role thân thiện

  return (
    <div className="container mx-auto py-10">
      <div className="flex flex-col gap-8 md:flex-row">
        <div className="w-full md:w-1/3 lg:w-1/4">
          <Card>
            <CardHeader className="flex flex-col items-center space-y-4 pt-6">
              <Avatar className="h-24 w-24 border">
                <AvatarImage src={userData.profilePic ?? undefined} alt={`Ảnh đại diện ${userData.firstName}`} />
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
                  <Mail className="h-4 w-4 flex-shrink-0 mt-1 text-muted-foreground" /> {/* Có thể thay icon Phone */}
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
        <div className="flex-1">
          <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="profile">Thông tin cá nhân</TabsTrigger>
              <TabsTrigger value="account">Tài khoản</TabsTrigger>
              <TabsTrigger value="notifications">Thông báo</TabsTrigger>
              <TabsTrigger value="security">Bảo mật</TabsTrigger>
            </TabsList>
            <TabsContent value="profile" className="mt-6">
              <form onSubmit={handleSaveProfile}>
                <Card>
                  <CardHeader>
                    <CardTitle>Thông tin cá nhân</CardTitle>
                    <CardDescription>Cập nhật họ tên, địa chỉ, tuổi của bạn.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label htmlFor="firstName">Tên</Label>
                        <Input id="firstName" value={profileForm.firstName} onChange={handleProfileChange} disabled={isSaving} placeholder="Nhập tên" />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="lastName">Họ</Label>
                        <Input id="lastName" value={profileForm.lastName} onChange={handleProfileChange} disabled={isSaving} placeholder="Nhập họ" />
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
                      <Input id="address" value={profileForm.address} onChange={handleProfileChange} disabled={isSaving} placeholder="Nhập địa chỉ" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="age">Tuổi</Label>
                      <Input id="age" type="number" value={profileForm.age} onChange={handleProfileChange} disabled={isSaving} placeholder="Nhập tuổi" min="0" />
                    </div>
                  </CardContent>
                  <CardFooter className="border-t px-6 py-4">
                    <Button type="submit" disabled={isSaving} className="ml-auto">
                      {isSaving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      {isSaving ? 'Đang lưu...' : 'Lưu thay đổi'}
                    </Button>
                  </CardFooter>
                </Card>
              </form>
            </TabsContent>
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
                  <Button variant="outline" size="sm">Đánh dấu tất cả đã đọc</Button>
                </CardFooter>
              </Card>
            </TabsContent>
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
                      <Input id="email" type="email" value={accountForm.email} onChange={handleAccountChange} disabled={isSaving} placeholder="your@email.com" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="phone">Số điện thoại</Label>
                      <Input id="phone" value={accountForm.phone} onChange={handleAccountChange} disabled={isSaving} placeholder="Nhập số điện thoại" />
                    </div>
                  </CardContent>
                  <CardFooter className="border-t px-6 py-4">
                    <Button type="submit" disabled={isSaving} className="ml-auto">
                      {isSaving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      {isSaving ? 'Đang lưu...' : 'Lưu thay đổi'}
                    </Button>
                  </CardFooter>
                </Card>
              </form>
            </TabsContent>
            <TabsContent value="security" className="mt-6">
              {/* Thêm thẻ form và onSubmit */}
              <form onSubmit={handleUpdatePassword}>
                <Card>
                  <CardHeader>
                    <CardTitle>Bảo mật</CardTitle>
                    <CardDescription>Thay đổi mật khẩu đăng nhập của bạn.</CardDescription> {/* Cập nhật mô tả */}
                  </CardHeader>
                  <CardContent className="space-y-5">
                    {/* Xóa Input Mật khẩu hiện tại */}
                    {/*
                    <div className="space-y-2">
                      <Label htmlFor="currentPassword">Mật khẩu hiện tại</Label>
                      <Input id="currentPassword" type="password" disabled={isSaving} />
                    </div>
                    */}

                    <div className="space-y-2">
                      <Label htmlFor="newPassword">Mật khẩu mới</Label>
                      {/* Kết nối Input với state và handler */}
                      <Input
                        id="newPassword"
                        type="password"
                        value={newPassword}
                        onChange={handlePasswordChange}
                        disabled={isSaving}
                        placeholder="Nhập mật khẩu mới" // Thêm placeholder
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="confirmPassword">Nhập lại mật khẩu mới</Label>
                      {/* Kết nối Input với state và handler */}
                      <Input
                        id="confirmPassword"
                        type="password"
                        value={confirmPassword}
                        onChange={handlePasswordChange}
                        disabled={isSaving}
                        placeholder="Xác nhận mật khẩu mới" // Thêm placeholder
                      />
                    </div>
                  </CardContent>
                  <CardFooter className="border-t px-6 py-4">
                    {/* Kích hoạt nút và thêm trạng thái loading */}
                    <Button type="submit" disabled={isSaving || !newPassword || !confirmPassword} className="ml-auto">
                      {isSaving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      {isSaving ? 'Đang cập nhật...' : 'Cập nhật mật khẩu'}
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