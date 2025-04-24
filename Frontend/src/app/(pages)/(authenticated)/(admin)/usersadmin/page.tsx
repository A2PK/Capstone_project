// src/app/(admin)/users/page.tsx (Ví dụ đường dẫn)
"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableCaption,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast"; // Đảm bảo đường dẫn đúng
import PageLoader from "@/components/pageloader"; // Đảm bảo đường dẫn đúng
import { Pagination } from "@/components/pagination"; // Import Pagination
import { PlusCircle, Edit, Trash2, Users, Search, UserCog, UserCheck, UserX, ShieldCheck, ShieldAlert, User } from "lucide-react";
import { cn } from "@/lib/utils"; // Đảm bảo đường dẫn đúng
import { createUser, deleteUser, getAllUsers, updateUser } from "@/lib/user"; // Import API functions (getUserById, searchUser có thể dùng nếu cần)
import { getUsers, User as AppUser} from "@/types/user"; // Import AppUser type
import { PaginParam } from "@/types/station2"; // Import PaginParam type

// --- Constants ---
const ITEMS_PER_PAGE = 10; // Số lượng user trên mỗi trang
const USER_ROLES = ["admin", "officer"]; // Cập nhật nếu role từ API khác, "viewer"
const USER_ACTIVE_STATUSES = ["active", "inactive"];

// --- Form Data Type ---
// Bao gồm các trường có thể chỉnh sửa/tạo mới trong form
type UserFormData = Omit<AppUser, 'id' | 'createdAt' | 'updatedAt' | 'isActive'> & {
    isActive: "active" | "inactive"; // Dùng string trong form cho Select
    password: string; 
};

// --- Component chính ---
export default function UsersAdminPage() {
  const { toast } = useToast();
  const [users, setUsers] = useState<AppUser[]>([]);
  const [paginationInfo, setPaginationInfo] = useState<PaginParam | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isAlertOpen, setIsAlertOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState<AppUser | null>(null); // User đang được sửa
  const [userToDeleteId, setUserToDeleteId] = useState<string | null>(null);

  // State cho form trong Dialog
  const [formData, setFormData] = useState<UserFormData>({
    username: "",
    email: "",
    firstName: "",
    lastName: "",
    role: "viewer", // Default role - Cập nhật nếu cần
    isActive: "active", // Default status
    phone: "",
    address: "",
    age: undefined, // Hoặc null nếu API cho phép
    profilePic: "",
    password: "",
  });

  // --- Fetch Users Function ---
  const fetchUsers = useCallback(async (page: number = 1) => {
    setIsLoading(true);
    try {
      const options = {
        limit: ITEMS_PER_PAGE,
        offset: (page - 1) * ITEMS_PER_PAGE,
        // Thêm filters, sortBy, sortDesc nếu API hỗ trợ và UI có
        // filters: { email: searchTerm } // Ví dụ nếu muốn search API theo email
      };
      const response = await getAllUsers("nothing",options); // Gọi API thật
      setUsers(response.users);
      setPaginationInfo(response.paginationInfo);
      setCurrentPage(page); // Cập nhật trang hiện tại
    } catch (error) {
      console.error("Failed to fetch users:", error);
      toast({
        title: "Lỗi",
        description: "Không thể tải danh sách người dùng. Vui lòng thử lại.",
        variant: "destructive",
      });
      setUsers([]); // Reset users nếu lỗi
      setPaginationInfo(null);
    } finally {
      setIsLoading(false);
    }
  }, [toast]); // Thêm toast vào dependency array

  // Load dữ liệu ban đầu và khi trang thay đổi
  useEffect(() => {
    fetchUsers(currentPage);
  }, [currentPage, fetchUsers]);

  // Lọc user phía client dựa trên searchTerm (áp dụng cho danh sách đã tải về)
  const filteredUsers = users.filter(
    (user) =>
      (user.firstName && user.firstName.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (user.lastName && user.lastName.toLowerCase().includes(searchTerm.toLowerCase())) ||
      user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // --- Xử lý mở Dialog ---
  const handleOpenDialog = (user: AppUser | null = null) => {
    setCurrentUser(user);
    if (user) {
      // Nếu sửa, điền form với dữ liệu user hiện tại
      setFormData({
        username: user.username,
        email: user.email,
        firstName: user.firstName || "",
        lastName: user.lastName || "",
        role: user.role, // Đảm bảo role này có trong USER_ROLES
        isActive: user.isActive ? "active" : "inactive", // Chuyển boolean sang string
        phone: user.phone || "",
        address: user.address || "",
        age: user.age || undefined,
        profilePic: user.profilePic || "",
        password: "",
      });
    } else {
      // Nếu thêm mới, reset form về giá trị mặc định
      setFormData({
        username: "", email: "", firstName: "", lastName: "",
        role: "officer", isActive: "active", phone: "", address: "", age: undefined, profilePic: "",password: "StrongP@ssw0rd!"
      });
    }
    setIsDialogOpen(true);
  };

  // --- Xử lý thay đổi Input trong Form ---
  const handleFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? (value === '' ? undefined : Number(value)) : value, // Xử lý input number
    }));
  };

  // --- Xử lý thay đổi Select trong Form ---
  const handleSelectChange = (name: keyof UserFormData, value: string) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // --- Xử lý Lưu User (Thêm mới hoặc Cập nhật) ---
  const handleSaveUser = async () => {
    // --- Input Validation cơ bản ---
    if (!formData.username || !formData.email || !formData.firstName || !formData.lastName) {
      toast({ title: "Lỗi", description: "Vui lòng nhập Username, Email, Tên và Họ.", variant: "destructive"});
      return;
    }
    if (!/\S+@\S+\.\S+/.test(formData.email)) {
      toast({ title: "Lỗi", description: "Địa chỉ email không hợp lệ.", variant: "destructive" });
      return;
    }
    // Thêm validation khác nếu cần

    setIsLoading(true);
    try {
        // Chuẩn bị dữ liệu gửi đi, chuyển isActive về boolean
        const userDataToSave: Partial<AppUser> = {
            ...formData,
            isActive: formData.isActive === "active", // Chuyển về boolean
            // Đảm bảo không gửi trường không cần thiết hoặc null/undefined nếu API yêu cầu
            phone: formData.phone || null,
            address: formData.address || null,
            age: formData.age || null,
            profilePic: formData.profilePic || null,
        };

        let status: number;
        if (currentUser) {
            // --- Cập nhật User ---
            // Chỉ gửi những trường đã thay đổi nếu cần tối ưu, hoặc gửi tất cả form data
            status = await updateUser(userDataToSave, currentUser.id); // API updateUser nhận Partial<AppUser>
            if (status >= 200 && status < 300) {
                toast({ title: "Thành công", description: `Đã cập nhật người dùng ${formData.username}.`, variant: "success"});
            } else {
                 throw new Error(`API trả về status ${status}`);
            }
        } else {
            // --- Thêm User mới ---
            // Loại bỏ các trường không cần thiết khi tạo mới nếu API yêu cầu
            // Ví dụ: không cần gửi id, createdAt, updatedAt
             // Cần thêm password nếu API yêu cầu khi tạo mới
            // const { id, createdAt, updatedAt, ...createData } = userDataToSave; // Nếu cần loại bỏ trường
             // Giả sử API createUser nhận đủ các trường này (trừ id, timestamp)
            status = await createUser(userDataToSave as AppUser); // Ép kiểu nếu chắc chắn đủ trường cần thiết cho API createUser
             if (status >= 200 && status < 300) {
                toast({ title: "Thành công", description: `Đã thêm người dùng ${formData.username}.`, variant: "success"});
            } else {
                 throw new Error(`API trả về status ${status}`);
            }
        }

        setIsDialogOpen(false); // Đóng Dialog
        setCurrentUser(null); // Reset currentUser
        await fetchUsers(currentPage); // Tải lại danh sách user trang hiện tại

    } catch (error) {
      console.error("Failed to save user:", error);
      toast({
        title: "Lỗi",
        description: `Không thể lưu người dùng. ${error instanceof Error ? error.message : 'Lỗi không xác định.'}`,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };


  // --- Xử lý mở Alert Xóa ---
  const handleDeleteClick = (userId: string) => {
    setUserToDeleteId(userId);
    setIsAlertOpen(true);
  };

  // --- Xử lý Xác nhận Xóa ---
  const handleConfirmDelete = async () => {
    if (!userToDeleteId) return;

    setIsLoading(true);
    try {
        // Gọi API deleteUser (mặc định hardDelete=true)
        const status = await deleteUser(userToDeleteId);

        if (status >= 200 && status < 300) {
            toast({ title: "Đã xóa", description: "Người dùng đã được xóa khỏi hệ thống." , variant: "destructive"});
            setUserToDeleteId(null);
            setIsAlertOpen(false); // Đóng Alert

            // Tải lại dữ liệu trang hiện tại hoặc trang trước đó nếu trang hiện tại trống sau khi xóa
            const remainingUsers = users.filter(u => u.id !== userToDeleteId);
            if (remainingUsers.length === 0 && currentPage > 1) {
                await fetchUsers(currentPage - 1);
            } else {
                 await fetchUsers(currentPage);
            }

        } else {
             throw new Error(`API trả về status ${status}`);
        }

    } catch (error) {
        console.error("Failed to delete user:", error);
        toast({
            title: "Lỗi",
            description: `Không thể xóa người dùng. ${error instanceof Error ? error.message : 'Lỗi không xác định.'}`,
            variant: "destructive",
        });
        setIsAlertOpen(false); // Đóng Alert ngay cả khi lỗi
    } finally {
        setIsLoading(false);
    }
  };

  // --- Xử lý chuyển trang ---
  const handlePageChange = (page: number) => {
    if (page !== currentPage) {
        setCurrentPage(page);
        // useEffect sẽ tự động gọi fetchUsers khi currentPage thay đổi
    }
  };

  // --- Tính toán số trang ---
  const totalPages = paginationInfo?.totalItems
    ? Math.ceil(Number(paginationInfo.totalItems) / ITEMS_PER_PAGE)
    : 0;


 // --- Helper Render Badges ---
  const renderRoleBadge = (role: string) => { // Dùng string để linh hoạt hơn
    const roleMap: Record<string, { text: string; className: string; icon: React.ReactNode }> = {
      admin: { text: "Quản trị viên", className: "border-red-500/50 bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300", icon: <ShieldAlert className="mr-1 h-3 w-3" /> },
      officer: { text: "Viên chức", className: "border-blue-500/50 bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300", icon: <UserCog className="mr-1 h-3 w-3" /> },
      viewer: { text: "Người xem", className: "border-gray-500/50 bg-gray-50 text-gray-700 dark:bg-gray-700 dark:text-gray-300", icon: <User className="mr-1 h-3 w-3" /> },
      // Thêm các role khác nếu cần
      default: { text: role, className: "border-gray-500/50 bg-gray-50 text-gray-700 dark:bg-gray-700 dark:text-gray-300", icon: <User className="mr-1 h-3 w-3" /> },
    };
    const roleInfo = roleMap[role] || roleMap.default;
    return <Badge variant="outline" className={cn("text-xs font-normal", roleInfo.className)}>{roleInfo.icon}{roleInfo.text}</Badge>;
  };

  // Cập nhật để nhận boolean
  const renderStatusBadge = (isActive: boolean) => {
    const statusMap: Record<string, { text: string; className: string; icon: React.ReactNode }> = {
       true: { text: "Hoạt động", className: "border-green-500/50 bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-300", icon: <UserCheck className="mr-1 h-3 w-3" /> },
       false: { text: "Ngừng hoạt động", className: "border-gray-400/50 bg-gray-50 text-gray-500 dark:bg-gray-700 dark:text-gray-400", icon: <UserX className="mr-1 h-3 w-3" /> },
    };
     const statusInfo = statusMap[String(isActive)]; // Chuyển boolean thành key string
     return <Badge variant="outline" className={cn("text-xs font-normal", statusInfo.className)}>{statusInfo.icon}{statusInfo.text}</Badge>;
  };


  // --- Render ---
  // Hiển thị loader toàn trang chỉ khi đang tải lần đầu
  if (isLoading && users.length === 0) return <PageLoader message="Đang tải danh sách người dùng..." />;

  return (
    <div className="container mx-auto py-6 px-4 md:px-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
        <h1 className="text-3xl font-bold tracking-tight flex items-center">
            <Users className="mr-3 h-7 w-7"/>
            Quản lý người dùng
        </h1>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button onClick={() => handleOpenDialog(null)}>
              <PlusCircle className="mr-2 h-4 w-4" /> Thêm người dùng mới
            </Button>
          </DialogTrigger>
          {/* --- Dialog Content (Form Thêm/Sửa) --- */}
          <DialogContent className="sm:max-w-[580px]"> {/* Tăng chiều rộng nếu cần */}
            <DialogHeader>
              <DialogTitle>{currentUser ? "Chỉnh sửa người dùng" : "Thêm người dùng mới"}</DialogTitle>
              <DialogDescription>
                {currentUser ? "Cập nhật thông tin và quyền hạn." : "Nhập thông tin cho người dùng mới."}
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4 max-h-[65vh] overflow-y-auto pr-6">
                {/* Username */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="username" className="text-right">Username <span className="text-red-500">*</span></Label>
                    <Input id="username" name="username" value={formData.username} onChange={handleFormChange} className="col-span-3" placeholder="Tên đăng nhập" disabled={!!currentUser} /> {/* Không cho sửa username? */}
                </div>
                {/* Email */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="email" className="text-right">Email <span className="text-red-500">*</span></Label>
                    <Input id="email" name="email" type="email" value={formData.email} onChange={handleFormChange} className="col-span-3" placeholder="vidu@example.com" />
                </div>
                {/* First Name */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="firstName" className="text-right">Tên <span className="text-red-500">*</span></Label>
                    <Input id="firstName" name="firstName" value={formData.firstName} onChange={handleFormChange} className="col-span-3" placeholder="Nhập tên" />
                </div>
                 {/* Last Name */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="lastName" className="text-right">Họ <span className="text-red-500">*</span></Label>
                    <Input id="lastName" name="lastName" value={formData.lastName} onChange={handleFormChange} className="col-span-3" placeholder="Nhập họ" />
                </div>
                 {/* Role */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="role" className="text-right">Quyền hạn <span className="text-red-500">*</span></Label>
                    <Select value={formData.role} onValueChange={(value) => handleSelectChange('role', value)}>
                        <SelectTrigger id="role" className="col-span-3">
                            <SelectValue placeholder="Chọn quyền hạn" />
                        </SelectTrigger>
                        <SelectContent>
                            {USER_ROLES.map((role) => (
                                <SelectItem key={role} value={role}>
                                    {renderRoleBadge(role)}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
                 {/* Active Status */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="isActive" className="text-right">Trạng thái <span className="text-red-500">*</span></Label>
                     <Select value={formData.isActive} onValueChange={(value) => handleSelectChange('isActive', value as "active" | "inactive")}>
                        <SelectTrigger id="isActive" className="col-span-3">
                            <SelectValue placeholder="Chọn trạng thái" />
                        </SelectTrigger>
                        <SelectContent>
                            {USER_ACTIVE_STATUSES.map((status) => (
                            <SelectItem key={status} value={status}>
                                {renderStatusBadge(status === 'active')}
                            </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
                 {/* Phone */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="phone" className="text-right">Số điện thoại</Label>
                    <Input id="phone" name="phone" value={formData.phone || ''} onChange={handleFormChange} className="col-span-3" placeholder="(Tùy chọn)" />
                </div>
                 {/* Address */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="address" className="text-right">Địa chỉ</Label>
                    <Input id="address" name="address" value={formData.address || ''} onChange={handleFormChange} className="col-span-3" placeholder="(Tùy chọn)" />
                </div>
                 {/* Age */}
                 <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="age" className="text-right">Tuổi</Label>
                    <Input id="age" name="age" type="number" value={formData.age == null ? '' : formData.age} onChange={handleFormChange} className="col-span-3" placeholder="(Tùy chọn)" />
                </div>
                {/* Avatar URL */}
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="profilePic" className="text-right">Avatar URL</Label>
                    <Input id="profilePic" name="profilePic" value={formData.profilePic || ''} onChange={handleFormChange} className="col-span-3" placeholder="(Tùy chọn) URL ảnh đại diện" />
                </div>

                {/* Thêm trường mật khẩu nếu cần thiết cho việc tạo mới */}
                 {!currentUser && (
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="password" className="text-right">Mật khẩu <span className="text-red-500">*</span></Label>
                        <Input id="password" name="password" type="text" value={formData.password || ''} onChange={handleFormChange} className="col-span-3" placeholder="Mật khẩu mặc định (có thể sửa)" />
                    </div>
                 )}
            </div>
            <DialogFooter>
              <DialogClose asChild>
                  <Button type="button" variant="outline" disabled={isLoading}>Hủy</Button>
              </DialogClose>
              <Button type="button" onClick={handleSaveUser} disabled={isLoading}>
                {isLoading ? "Đang lưu..." : "Lưu thay đổi"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

        {/* --- Thanh Search (Client-side filtering) --- */}
        <div className="mb-4">
            <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                    type="search"
                    placeholder="Tìm theo tên, username, email..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-8 w-full sm:w-[300px]"
                />
            </div>
        </div>


      {/* --- Bảng Danh sách Người dùng --- */}
      <div className="border rounded-lg overflow-hidden">
        <Table>
          {/* Hiển thị caption phù hợp hơn khi có hoặc không có dữ liệu */}
          <TableCaption>
              {filteredUsers.length > 0
                ? `Hiển thị ${filteredUsers.length} trên tổng số ${paginationInfo?.totalItems || 0} người dùng.`
                : "Danh sách người dùng trong hệ thống."}
          </TableCaption>
          <TableHeader className="bg-muted/50 dark:bg-muted/30">
            <TableRow>
              <TableHead className="w-[80px]">Avatar</TableHead>
              <TableHead>Tên</TableHead>
              <TableHead>Username</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Quyền hạn</TableHead>
              <TableHead>Trạng thái</TableHead>
              <TableHead>Cập nhật lần cuối</TableHead> {/* Thay lastLogin thành updatedAt */}
              <TableHead className="text-right w-[120px]">Hành động</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading && users.length > 0 && ( // Hiển thị loading nhẹ khi tải lại
                 <TableRow>
                    <TableCell colSpan={8} className="h-24 text-center text-muted-foreground">
                       Đang tải dữ liệu...
                    </TableCell>
                 </TableRow>
            )}
            {!isLoading && filteredUsers.length > 0 ? (
              filteredUsers.map((user) => (
                <TableRow key={user.id} className="hover:bg-muted/40 dark:hover:bg-muted/20">
                  <TableCell>
                    <Avatar className="h-9 w-9">
                      <AvatarImage src={user.profilePic || undefined} alt={`${user.firstName} ${user.lastName}`} />
                      <AvatarFallback>{`${user.firstName?.charAt(0) ?? ''}${user.lastName?.charAt(0) ?? ''}`.toUpperCase()}</AvatarFallback>
                    </Avatar>
                  </TableCell>
                  <TableCell className="font-medium">{`${user.firstName || ''} ${user.lastName || ''}`}</TableCell>
                  <TableCell className="text-muted-foreground">{user.username}</TableCell>
                  <TableCell className="text-muted-foreground">{user.email}</TableCell>
                  <TableCell>{renderRoleBadge(user.role)}</TableCell>
                  <TableCell>{renderStatusBadge(user.isActive)}</TableCell>
                  <TableCell className="text-muted-foreground text-sm">
                     {/* Format lại ngày tháng nếu cần */}
                     {user.updatedAt ? new Date(user.updatedAt).toLocaleString() : "Chưa có"}
                  </TableCell>
                  <TableCell className="text-right">
                    {/* Edit Button */}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-muted-foreground hover:text-primary h-8 w-8"
                      onClick={() => handleOpenDialog(user)}
                      disabled={isLoading || user.role === 'admin'}
                      title={user.role === 'admin' ? "Không thể sửa quản trị viên" : "Sửa người dùng"}
                    >
                      <Edit className="h-4 w-4" />
                      <span className="sr-only">Sửa</span>
                    </Button>
                    {/* Delete Button + Alert */}
                    <AlertDialog open={isAlertOpen && userToDeleteId === user.id} onOpenChange={(open) => {if (!open) setUserToDeleteId(null); setIsAlertOpen(open);}}>
                        <AlertDialogTrigger asChild>
                             <Button
                                variant="ghost"
                                size="icon"
                                className="text-muted-foreground hover:text-destructive h-8 w-8"
                                onClick={(e) => {
                                  // Ngăn chặn mở dialog nếu nút bị vô hiệu hóa (đề phòng)
                                  if (user.role === 'admin') {
                                    e.preventDefault();
                                    return;
                                  }
                                  handleDeleteClick(user.id);
                                }}
                                 // ---- THAY ĐỔI Ở ĐÂY ----
                                disabled={isLoading || user.role === 'admin'}
                                title={user.role === 'admin' ? "Không thể xóa quản trị viên" : "Xóa người dùng"}
                            >
                                <Trash2 className="h-4 w-4" />
                                <span className="sr-only">Xóa</span>
                            </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                        <AlertDialogHeader>
                            <AlertDialogTitle>Xác nhận xóa người dùng?</AlertDialogTitle>
                            <AlertDialogDescription>
                            Bạn có chắc chắn muốn xóa người dùng <strong>{user.username}</strong> ({user.firstName} {user.lastName})? Hành động này không thể hoàn tác.
                            </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                            <AlertDialogCancel disabled={isLoading}>Hủy</AlertDialogCancel>
                            <AlertDialogAction onClick={handleConfirmDelete} disabled={isLoading} className={cn("bg-destructive hover:bg-destructive/90", isLoading && "opacity-50 cursor-not-allowed")}>
                            {isLoading ? "Đang xóa..." : "Xác nhận xóa"}
                            </AlertDialogAction>
                        </AlertDialogFooter>
                        </AlertDialogContent>
                    </AlertDialog>
                  </TableCell>
                </TableRow>
              ))
            ) : (
              !isLoading && ( // Chỉ hiển thị "Không tìm thấy" khi không loading
                <TableRow>
                  <TableCell colSpan={8} className="h-24 text-center text-muted-foreground">
                    {searchTerm ? "Không tìm thấy người dùng phù hợp." : "Chưa có người dùng nào."}
                  </TableCell>
                </TableRow>
              )
            )}
          </TableBody>
        </Table>
      </div>

        {/* --- Pagination Controls --- */}
        {totalPages > 1 && (
            <div className="flex justify-center mt-6">
                <Pagination
                    currentPage={currentPage}
                    totalPages={totalPages}
                    onPageChange={handlePageChange}
                />
            </div>
        )}

    </div>
  );
}