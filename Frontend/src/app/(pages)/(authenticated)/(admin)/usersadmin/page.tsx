"use client"; // Đánh dấu đây là Client Component

import React, { useState, useEffect } from "react";
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
  DialogClose, // Import DialogClose
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
import { useToast } from "@/hooks/use-toast"; // Sử dụng hook toast
import PageLoader from "@/components/pageloader"; // Giả sử bạn có component này
import { PlusCircle, Edit, Trash2, Users, Search, UserCog, UserCheck, UserX, ShieldCheck, ShieldAlert, User } from "lucide-react";
import { cn } from "@/lib/utils";

// --- Định nghĩa kiểu User ---
type UserRole = "admin" | "manager" | "viewer";
type UserStatus = "active" | "inactive";

interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  status: UserStatus;
  lastLogin?: string; // Format: "YYYY-MM-DD HH:mm" hoặc Date object
  avatar?: string; // URL ảnh avatar
}

// --- Dữ liệu giả lập ---
const mockUsers: User[] = [
  { id: "usr_001", name: "Admin Chính", email: "admin@example.com", role: "admin", status: "active", lastLogin: "2025-04-03 15:30", avatar: "https://randomuser.me/api/portraits/men/1.jpg" },
  { id: "usr_002", name: "Quản lý A", email: "manager.a@example.com", role: "manager", status: "active", lastLogin: "2025-04-03 09:15", avatar: "https://randomuser.me/api/portraits/women/2.jpg" },
  { id: "usr_003", name: "Người xem B", email: "viewer.b@example.com", role: "viewer", status: "active", lastLogin: "2025-04-02 11:00" },
  { id: "usr_004", name: "Người xem C (Inactive)", email: "viewer.c@example.com", role: "viewer", status: "inactive", avatar: "https://randomuser.me/api/portraits/men/3.jpg" },
  { id: "usr_005", name: "Quản lý B (Inactive)", email: "manager.b@example.com", role: "manager", status: "inactive", lastLogin: "2025-03-20 18:00", avatar: "https://randomuser.me/api/portraits/women/4.jpg" },
];

const USER_ROLES: UserRole[] = ["admin", "manager", "viewer"];
const USER_STATUSES: UserStatus[] = ["active", "inactive"];

// --- Component chính ---
export default function UsersAdminPage() {
  const { toast } = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isAlertOpen, setIsAlertOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState<User | null>(null); // User đang được sửa, null nếu là thêm mới
  const [userToDeleteId, setUserToDeleteId] = useState<string | null>(null);

  // State cho form trong Dialog
  const [formData, setFormData] = useState<Omit<User, 'id' | 'lastLogin'>>({
    name: "",
    email: "",
    role: "viewer", // Default role
    status: "active", // Default status
    avatar: "",
  });

  // Load dữ liệu ban đầu
  useEffect(() => {
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setUsers(mockUsers);
      setIsLoading(false);
    }, 1000);
  }, []);

  // Lọc user dựa trên searchTerm
  const filteredUsers = users.filter(
    (user) =>
      user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // --- Xử lý mở Dialog ---
  const handleOpenDialog = (user: User | null = null) => {
    setCurrentUser(user);
    if (user) {
      // Nếu sửa, điền form với dữ liệu user hiện tại
      setFormData({
        name: user.name,
        email: user.email,
        role: user.role,
        status: user.status,
        avatar: user.avatar || "",
      });
    } else {
      // Nếu thêm mới, reset form
      setFormData({ name: "", email: "", role: "viewer", status: "active", avatar: "" });
    }
    setIsDialogOpen(true);
  };

  // --- Xử lý thay đổi Input trong Form ---
  const handleFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // --- Xử lý thay đổi Select trong Form ---
  const handleSelectChange = (name: keyof typeof formData, value: string) => {
    setFormData((prev) => ({ ...prev, [name]: value as UserRole | UserStatus })); // Ép kiểu nếu cần
  };

  // --- Xử lý Lưu User (Thêm mới hoặc Cập nhật) ---
  const handleSaveUser = () => {
    // --- Input Validation cơ bản ---
    if (!formData.name || !formData.email) {
        toast({
            title: "Lỗi",
            description: "Vui lòng nhập Tên và Email.",
            variant: "destructive",
        });
        return;
    }
    // Email validation đơn giản
    if (!/\S+@\S+\.\S+/.test(formData.email)) {
         toast({
            title: "Lỗi",
            description: "Địa chỉ email không hợp lệ.",
            variant: "destructive",
        });
        return;
    }

    setIsLoading(true); // Bật loading giả lập
    setTimeout(() => { // Giả lập gọi API
        if (currentUser) {
        // --- Cập nhật User ---
        setUsers((prevUsers) =>
            prevUsers.map((user) =>
            user.id === currentUser.id ? { ...currentUser, ...formData } : user
            )
        );
        toast({
            title: "Thành công",
            description: `Đã cập nhật người dùng ${formData.name}.`,
            variant: "success",
        });
        } else {
        // --- Thêm User mới ---
        const newUser: User = {
            id: `usr_${Date.now().toString().slice(-5)}`, // Tạo ID tạm
            ...formData,
            lastLogin: undefined, // Người mới chưa login
        };
        setUsers((prevUsers) => [newUser, ...prevUsers]); // Thêm vào đầu danh sách
        toast({
            title: "Thành công",
            description: `Đã thêm người dùng ${newUser.name}.`,
            variant: "success",
        });
        }
        setIsLoading(false); // Tắt loading
        setIsDialogOpen(false); // Đóng Dialog
        setCurrentUser(null); // Reset currentUser
    }, 500);
  };


  // --- Xử lý mở Alert Xóa ---
  const handleDeleteClick = (userId: string) => {
    setUserToDeleteId(userId);
    setIsAlertOpen(true);
  };

  // --- Xử lý Xác nhận Xóa ---
  const handleConfirmDelete = () => {
    if (!userToDeleteId) return;

    setIsLoading(true); // Bật loading giả lập
    setTimeout(() => { // Giả lập gọi API
        setUsers((prevUsers) => prevUsers.filter((user) => user.id !== userToDeleteId));
        toast({
            title: "Đã xóa",
            description: "Người dùng đã được xóa khỏi hệ thống.",
            // variant: "destructive", // Có thể dùng destructive hoặc success tùy ý
        });
        setIsLoading(false); // Tắt loading
        setIsAlertOpen(false); // Đóng Alert
        setUserToDeleteId(null); // Reset userToDeleteId
    }, 500);
  };

  // --- Helper Render Badges ---
  const renderRoleBadge = (role: UserRole) => {
    const roleMap: Record<UserRole, { text: string; className: string; icon: React.ReactNode }> = {
      admin: { text: "Quản trị viên", className: "border-red-500/50 bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300", icon: <ShieldAlert className="mr-1 h-3 w-3" /> },
      manager: { text: "Quản lý", className: "border-blue-500/50 bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300", icon: <UserCog className="mr-1 h-3 w-3" /> },
      viewer: { text: "Người xem", className: "border-gray-500/50 bg-gray-50 text-gray-700 dark:bg-gray-700 dark:text-gray-300", icon: <User className="mr-1 h-3 w-3" /> },
    };
    const { text, className, icon } = roleMap[role];
    return <Badge variant="outline" className={cn("text-xs font-normal", className)}>{icon}{text}</Badge>;
  };

  const renderStatusBadge = (status: UserStatus) => {
    const statusMap: Record<UserStatus, { text: string; className: string; icon: React.ReactNode }> = {
      active: { text: "Hoạt động", className: "border-green-500/50 bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-300", icon: <UserCheck className="mr-1 h-3 w-3" /> },
      inactive: { text: "Ngừng hoạt động", className: "border-gray-400/50 bg-gray-50 text-gray-500 dark:bg-gray-700 dark:text-gray-400", icon: <UserX className="mr-1 h-3 w-3" /> },
    };
     const { text, className, icon } = statusMap[status];
    return <Badge variant="outline" className={cn("text-xs font-normal", className)}>{icon}{text}</Badge>;
  };

  // --- Render ---
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
          <DialogContent className="sm:max-w-[480px]">
            <DialogHeader>
              <DialogTitle>{currentUser ? "Chỉnh sửa người dùng" : "Thêm người dùng mới"}</DialogTitle>
              <DialogDescription>
                {currentUser ? "Cập nhật thông tin và quyền hạn." : "Nhập thông tin cho người dùng mới."}
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
               {/* Input Tên */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">
                  Tên
                </Label>
                <Input
                  id="name"
                  name="name" // Thêm name để handleFormChange nhận diện
                  value={formData.name}
                  onChange={handleFormChange}
                  className="col-span-3"
                  placeholder="Nhập họ và tên"
                />
              </div>
              {/* Input Email */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="email" className="text-right">
                  Email
                </Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleFormChange}
                  className="col-span-3"
                  placeholder="vidu@example.com"
                />
              </div>
              {/* Select Quyền hạn (Role) */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="role" className="text-right">
                  Quyền hạn
                </Label>
                 <Select
                    value={formData.role}
                    onValueChange={(value) => handleSelectChange('role', value)}
                  >
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
              {/* Select Trạng thái (Status) */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="status" className="text-right">
                  Trạng thái
                </Label>
                 <Select
                    value={formData.status}
                    onValueChange={(value) => handleSelectChange('status', value)}
                  >
                    <SelectTrigger id="status" className="col-span-3">
                      <SelectValue placeholder="Chọn trạng thái" />
                    </SelectTrigger>
                    <SelectContent>
                       {USER_STATUSES.map((status) => (
                        <SelectItem key={status} value={status}>
                         {renderStatusBadge(status)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
              </div>
               {/* Input Avatar URL (Optional) */}
               <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="avatar" className="text-right">
                  Avatar URL
                </Label>
                <Input
                  id="avatar"
                  name="avatar"
                  value={formData.avatar}
                  onChange={handleFormChange}
                  className="col-span-3"
                  placeholder="(Tùy chọn) Nhập URL ảnh đại diện"
                />
              </div>
            </div>
            <DialogFooter>
              <DialogClose asChild>
                  <Button type="button" variant="outline">Hủy</Button>
              </DialogClose>
              <Button type="button" onClick={handleSaveUser} disabled={isLoading}>
                {isLoading ? "Đang lưu..." : "Lưu thay đổi"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

        {/* --- Thanh Search --- */}
        <div className="mb-4">
            <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                    type="search"
                    placeholder="Tìm kiếm theo tên hoặc email..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-8 w-full sm:w-[300px]"
                />
            </div>
        </div>


      {/* --- Bảng Danh sách Người dùng --- */}
      <div className="border rounded-lg overflow-hidden">
        <Table>
          <TableCaption>Danh sách người dùng trong hệ thống.</TableCaption>
          <TableHeader className="bg-muted/50 dark:bg-muted/30">
            <TableRow>
              <TableHead className="w-[80px]">Avatar</TableHead>
              <TableHead>Tên</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Quyền hạn</TableHead>
              <TableHead>Trạng thái</TableHead>
              <TableHead>Lần đăng nhập cuối</TableHead>
              <TableHead className="text-right w-[120px]">Hành động</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredUsers.length > 0 ? (
              filteredUsers.map((user) => (
                <TableRow key={user.id} className="hover:bg-muted/40 dark:hover:bg-muted/20">
                  <TableCell>
                    <Avatar className="h-9 w-9">
                      <AvatarImage src={user.avatar} alt={user.name} />
                      <AvatarFallback>{user.name.charAt(0).toUpperCase()}</AvatarFallback>
                    </Avatar>
                  </TableCell>
                  <TableCell className="font-medium">{user.name}</TableCell>
                  <TableCell className="text-muted-foreground">{user.email}</TableCell>
                  <TableCell>{renderRoleBadge(user.role)}</TableCell>
                  <TableCell>{renderStatusBadge(user.status)}</TableCell>
                  <TableCell className="text-muted-foreground text-sm">
                    {user.lastLogin || "Chưa đăng nhập"}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-muted-foreground hover:text-primary h-8 w-8"
                      onClick={() => handleOpenDialog(user)}
                      disabled={isLoading} // Disable khi đang loading
                    >
                      <Edit className="h-4 w-4" />
                      <span className="sr-only">Sửa</span>
                    </Button>
                    <AlertDialog open={isAlertOpen && userToDeleteId === user.id} onOpenChange={(open) => {if (!open) setUserToDeleteId(null); setIsAlertOpen(open);}}>
                      <AlertDialogTrigger asChild>
                         <Button
                           variant="ghost"
                           size="icon"
                           className="text-muted-foreground hover:text-destructive h-8 w-8"
                           onClick={() => handleDeleteClick(user.id)}
                           disabled={isLoading}
                         >
                          <Trash2 className="h-4 w-4" />
                          <span className="sr-only">Xóa</span>
                         </Button>
                      </AlertDialogTrigger>
                       {/* --- Alert Dialog Content (Xác nhận Xóa) --- */}
                       <AlertDialogContent>
                        <AlertDialogHeader>
                            <AlertDialogTitle>Xác nhận xóa người dùng?</AlertDialogTitle>
                            <AlertDialogDescription>
                            Bạn có chắc chắn muốn xóa người dùng <strong>{user.name}</strong>? Hành động này không thể hoàn tác.
                            </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                            <AlertDialogCancel disabled={isLoading}>Hủy</AlertDialogCancel>
                            <AlertDialogAction onClick={handleConfirmDelete} disabled={isLoading} className="bg-destructive hover:bg-destructive/90">
                                {isLoading ? "Đang xóa..." : "Xác nhận xóa"}
                            </AlertDialogAction>
                        </AlertDialogFooter>
                       </AlertDialogContent>
                    </AlertDialog>
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={7} className="h-24 text-center text-muted-foreground">
                  {searchTerm ? "Không tìm thấy người dùng phù hợp." : "Chưa có người dùng nào."}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

       {/* Optional: Pagination controls can be added here */}

    </div>
  );
}