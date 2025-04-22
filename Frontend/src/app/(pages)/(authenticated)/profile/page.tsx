// src/app/dashboard/profile/page.tsx (hoặc đường dẫn tương ứng)

"use client"

import { useEffect, useState, ChangeEvent, FormEvent } from "react";
import Link from "next/link";
import { format } from 'date-fns'; // Để format ngày tháng

// Components UI (Shadcn)
import PageLoader from "@/components/pageloader";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Icons (lucide-react)
import { UserCircle, Mail, MapPin, Briefcase, Calendar, AlertCircle, Loader2 } from "lucide-react";

// Logic & Types
import type { User } from '@/types/user'; // Import kiểu User
import { updateUser } from '@/lib/user';   // *** Import hàm cập nhật user ***
import { useToast } from "@/hooks/use-toast"; // *** Import hook toast từ đường dẫn custom ***

// --- Component hiển thị lỗi ---
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

// --- Kiểu dữ liệu cho state form input ---
interface ProfileFormData {
    firstName: string;
    lastName: string;
    address: string;
    age: string; // Dùng string cho input, chuyển đổi khi gửi
}

interface AccountFormData {
    email: string;
    phone: string;
}

// --- Component Chính: ProfilePage ---
export default function ProfilePage() {
  // ---- State ----
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false); // Trạng thái đang lưu
  const [userData, setUserData] = useState<User | null>(null); // Dữ liệu user fetch về
  const [error, setError] = useState<string | null>(null); // Lỗi fetch hoặc lưu
  const { toast } = useToast(); // Hook để hiển thị thông báo

  // State cho các form input (quản lý giá trị nhập liệu)
  const [profileForm, setProfileForm] = useState<ProfileFormData>({
    firstName: '',
    lastName: '',
    address: '',
    age: '',
  });
  const [accountForm, setAccountForm] = useState<AccountFormData>({
    email: '',
    phone: '',
  });

  // ---- Data Fetching ----
  useEffect(() => {
    const fetchUserData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/auth/session'); // Gọi API lấy session/user data
        if (!response.ok) {
          throw new Error(`Lỗi mạng: ${response.status}`);
        }
        const data = await response.json();

        if (data.isLoggedIn && data.user) {
          setUserData(data.user);
          // Khởi tạo giá trị ban đầu cho các state form từ userData
          setProfileForm({
              firstName: data.user.firstName ?? '',
              lastName: data.user.lastName ?? '',
              address: data.user.address ?? '',
              age: data.user.age?.toString() ?? '', // Chuyển number sang string
          });
          setAccountForm({
              email: data.user.email ?? '',
              phone: data.user.phone ?? '',
          });
        } else {
          // Trường hợp không đăng nhập hoặc API không trả về user
          console.warn("ProfilePage: User not logged in or user data not found from API.");
          setError("Bạn chưa đăng nhập hoặc không tìm thấy thông tin người dùng.");
          // Optional: Chuyển hướng về trang login
          // import { useRouter } from 'next/navigation';
          // const router = useRouter();
          // router.push('/auth/login');
        }
      } catch (e: any) {
        console.error("ProfilePage: Failed to fetch user data:", e);
        setError(e.message || "Đã xảy ra lỗi khi tải thông tin.");
      } finally {
        setIsLoading(false); // Kết thúc loading
      }
    };

    fetchUserData();
  }, []); // Chạy 1 lần khi component mount

  // ---- Input Change Handlers ----
  const handleProfileChange = (e: ChangeEvent<HTMLInputElement>) => {
      const { id, value } = e.target;
      setProfileForm(prev => ({ ...prev, [id]: value }));
  };

  const handleAccountChange = (e: ChangeEvent<HTMLInputElement>) => {
      const { id, value } = e.target;
      setAccountForm(prev => ({ ...prev, [id]: value }));
  };

  // ---- Save Handlers ----
  // Lưu thông tin cá nhân
  const handleSaveProfile = async (e: FormEvent) => {
      e.preventDefault(); // Ngăn form submit và tải lại trang
      if (!userData) return; // Đảm bảo có userData trước khi lưu
      setIsSaving(true);

      // Chuẩn bị dữ liệu cần gửi đi (chỉ những trường thuộc form này)
      const updateData: Partial<User> = {
          firstName: profileForm.firstName.trim(),
          lastName: profileForm.lastName.trim(),
          address: profileForm.address.trim(),
          // Chuyển age từ string về number, nếu rỗng/invalid thì thành null
          age: profileForm.age ? parseInt(profileForm.age, 10) || null : null,
      };

      try {
          // Gọi hàm updateUser từ lib
          const status = await updateUser(updateData, userData.id);

          if (status >= 200 && status < 300) { // Kiểm tra status thành công
              toast({
                  title: "Thành công",
                  description: "Thông tin cá nhân đã được cập nhật.",
                  variant: "success", // hoặc 'success'
              });
              // Cập nhật lại state userData chính để UI đồng bộ
              setUserData(prev => prev ? { ...prev, ...updateData } : null);
          } else {
             // Trường hợp API trả về lỗi nhưng không throw (ít xảy ra với axios)
             throw new Error(`Lỗi từ server: ${status}`);
          }
      } catch (err: any) {
          console.error("ProfilePage: Failed to update profile:", err);
          toast({
              title: "Lỗi cập nhật",
              description: err.message || "Không thể cập nhật thông tin cá nhân.",
              variant: "destructive",
          });
      } finally {
          setIsSaving(false); // Kết thúc trạng thái lưu
      }
  };

  // Lưu thông tin tài khoản
  const handleSaveAccount = async (e: FormEvent) => {
      e.preventDefault();
      if (!userData) return;
      setIsSaving(true);

      // Chuẩn bị dữ liệu (chỉ email và phone)
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
               // Cập nhật lại state userData chính
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

  // ---- Conditional Rendering ----
  if (isLoading) return <PageLoader message="Đang tải trang thông tin cá nhân..." />;
  if (error) return <ErrorDisplay message={error} />; // Hiển thị lỗi nếu có
  if (!userData) {
      // Trường hợp không loading, không lỗi nhưng không có user (vd: chưa đăng nhập)
       return (
         <div className="container mx-auto py-10 text-center">
           <p>Vui lòng <Link href="/auth/login" className="text-blue-600 hover:underline">đăng nhập</Link> để xem thông tin.</p>
         </div>
       );
  }

  // ---- Main Render ----
  // Chuẩn bị một số giá trị hiển thị từ userData
  const fallbackName = `${userData.firstName?.[0] ?? ''}${userData.lastName?.[0] ?? ''}`.toUpperCase() || 'U';
  const joinDate = userData.createdAt ? format(new Date(userData.createdAt), 'dd/MM/yyyy') : 'N/A'; // Format ngày VN
  const displayRole = userData.role === 'admin' ? 'Quản trị viên' : (userData.role === 'officer' ? 'Nhân viên' : userData.role); // Hiển thị role thân thiện

  return (
    <div className="container mx-auto py-10">
      {/* Nhớ đảm bảo <Toaster /> đã có trong layout cha */}
      <div className="flex flex-col gap-8 md:flex-row">

        {/* ---- Cột Sidebar Thông tin tóm tắt (dữ liệu từ state `userData`) ---- */}
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
            {/* CardFooter có thể thêm nút khác nếu cần */}
          </Card>
        </div>

        {/* ---- Cột Nội dung chính (Tabs) ---- */}
        <div className="flex-1">
          <Tabs defaultValue="profile" className="w-full">
            {/* Tabs List */}
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="profile">Thông tin cá nhân</TabsTrigger>
              <TabsTrigger value="account">Tài khoản</TabsTrigger>
              <TabsTrigger value="security">Bảo mật</TabsTrigger>
            </TabsList>

            {/* ---- Tab Content: Thông tin cá nhân ---- */}
            <TabsContent value="profile" className="mt-6">
              <form onSubmit={handleSaveProfile}>
                  <Card>
                    <CardHeader>
                      <CardTitle>Thông tin cá nhân</CardTitle>
                      <CardDescription>Cập nhật họ tên, địa chỉ, tuổi của bạn.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-5">
                      {/* First Name / Last Name */}
                      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                        <div className="space-y-2">
                          <Label htmlFor="firstName">Tên</Label>
                          <Input id="firstName" value={profileForm.firstName} onChange={handleProfileChange} disabled={isSaving} placeholder="Nhập tên"/>
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="lastName">Họ</Label>
                          <Input id="lastName" value={profileForm.lastName} onChange={handleProfileChange} disabled={isSaving} placeholder="Nhập họ"/>
                        </div>
                      </div>
                      {/* Username (Readonly) */}
                      <div className="space-y-2">
                        <Label htmlFor="username">Username</Label>
                        <Input id="username" defaultValue={userData.username} readOnly disabled />
                      </div>
                      {/* Role (Readonly) */}
                      <div className="space-y-2">
                        <Label htmlFor="role">Vai trò</Label>
                        <Input id="role" defaultValue={displayRole} readOnly disabled />
                      </div>
                      {/* Address */}
                      <div className="space-y-2">
                        <Label htmlFor="address">Địa chỉ</Label>
                        <Input id="address" value={profileForm.address} onChange={handleProfileChange} disabled={isSaving} placeholder="Nhập địa chỉ"/>
                      </div>
                      {/* Age */}
                      <div className="space-y-2">
                        <Label htmlFor="age">Tuổi</Label>
                        <Input id="age" type="number" value={profileForm.age} onChange={handleProfileChange} disabled={isSaving} placeholder="Nhập tuổi" min="0"/>
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

             {/* ---- Tab Content: Tài khoản ---- */}
            <TabsContent value="account" className="mt-6">
               <form onSubmit={handleSaveAccount}>
                  <Card>
                    <CardHeader>
                      <CardTitle>Tài khoản</CardTitle>
                      <CardDescription>Quản lý thông tin liên lạc của bạn.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-5">
                      {/* Email */}
                      <div className="space-y-2">
                        <Label htmlFor="email">Email</Label>
                        <Input id="email" type="email" value={accountForm.email} onChange={handleAccountChange} disabled={isSaving} placeholder="your@email.com"/>
                      </div>
                      {/* Phone */}
                      <div className="space-y-2">
                        <Label htmlFor="phone">Số điện thoại</Label>
                        <Input id="phone" value={accountForm.phone} onChange={handleAccountChange} disabled={isSaving} placeholder="Nhập số điện thoại"/>
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

             {/* ---- Tab Content: Bảo mật ---- */}
            <TabsContent value="security" className="mt-6">
              {/* Form này cần logic và API riêng để đổi mật khẩu */}
              <Card>
                <CardHeader>
                  <CardTitle>Bảo mật</CardTitle>
                  <CardDescription>Thay đổi mật khẩu của bạn.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                  <div className="space-y-2">
                    <Label htmlFor="currentPassword">Mật khẩu hiện tại</Label>
                    <Input id="currentPassword" type="password" disabled={isSaving} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="newPassword">Mật khẩu mới</Label>
                    <Input id="newPassword" type="password" disabled={isSaving}/>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword">Nhập lại mật khẩu mới</Label>
                    <Input id="confirmPassword" type="password" disabled={isSaving}/>
                  </div>
                </CardContent>
                <CardFooter className="border-t px-6 py-4">
                   {/* TODO: Implement update password logic (cần API endpoint riêng) */}
                  <Button disabled={true} className="ml-auto">Cập nhật mật khẩu</Button>
                  {/* Tạm thời disable nút này */}
                </CardFooter>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}