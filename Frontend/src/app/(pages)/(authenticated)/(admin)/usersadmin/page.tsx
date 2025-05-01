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
import { PlusCircle, Edit, Trash2, Users, Search, UserCog, UserCheck, UserX, ShieldCheck, ShieldAlert, User, ShieldX, AlertCircle } from "lucide-react"; // Thêm icons nếu cần
import { cn } from "@/lib/utils"; // Đảm bảo đường dẫn đúng
import { createUser, deleteUser, getAllUsers, updateUser } from "@/lib/user"; // Import API functions
import { getUsers, User as AppUser } from "@/types/user";
import { PaginParam } from "@/types/station2";
import { useAuth } from '@/hooks/useAuth'; // *** IMPORT HOOK useAuth THỰC TẾ ***

// --- Constants ---
const ITEMS_PER_PAGE = 10;
const ALL_USER_ROLES = ["manager", "admin", "officer"] as const;
const FORM_USER_ROLES = ["admin", "officer"] as const;
const USER_ACTIVE_STATUSES = ["active", "inactive"] as const;

type UserRole = typeof ALL_USER_ROLES[number];

// --- Form Data Type ---
type UserFormData = Omit<AppUser, 'id' | 'createdAt' | 'updatedAt' | 'isActive' | 'role'> & {
    isActive: "active" | "inactive";
    role: typeof FORM_USER_ROLES[number];
    password?: string;
};

// --- Component chính ---
export default function UsersAdminPage() {
    const { toast } = useToast();
    // *** SỬ DỤNG HOOK useAuth THỰC TẾ ***
    const {
        user: loggedInUser, // đổi tên 'user' từ hook thành 'loggedInUser' cho nhất quán với code cũ
        isLoading: isAuthLoading, // đổi tên 'isLoading' từ hook để tránh xung đột
        isLoggedIn,
        error: authError
    } = useAuth();

    const [users, setUsers] = useState<AppUser[]>([]);
    const [paginationInfo, setPaginationInfo] = useState<PaginParam | null>(null);
    const [currentPage, setCurrentPage] = useState(1);
    const [isLoadingData, setIsLoadingData] = useState(true); // State riêng cho loading dữ liệu bảng
    const [searchTerm, setSearchTerm] = useState("");
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [isAlertOpen, setIsAlertOpen] = useState(false);
    const [currentUser, setCurrentUser] = useState<AppUser | null>(null);
    const [userToDeleteId, setUserToDeleteId] = useState<string | null>(null);

    const [formData, setFormData] = useState<UserFormData>({
        username: "", email: "", firstName: "", lastName: "",
        role: "officer", isActive: "active", phone: "", address: "", age: undefined, profilePic: "", password: ""
    });

    // --- Fetch Users Function ---
    const fetchUsers = useCallback(async (page: number = 1) => {
        setIsLoadingData(true); // Sử dụng state loading riêng
        try {
            const options = {
                limit: ITEMS_PER_PAGE,
                offset: (page - 1) * ITEMS_PER_PAGE,
            };
            // Note: API getAllUsers có thể cần token xác thực. Đảm bảo API client (axios) được cấu hình để gửi token nếu cần.
            // Ví dụ, bạn có thể cần cấu hình interceptor cho axios để tự động đính kèm token từ useAuth().token
            const response = await getAllUsers("nothing", options);
            setUsers(response.users);
            setPaginationInfo(response.paginationInfo);
            setCurrentPage(page);
        } catch (error) {
            console.error("Failed to fetch users:", error);
            toast({
                title: "Lỗi tải dữ liệu",
                description: "Không thể tải danh sách người dùng. Vui lòng thử lại.",
                variant: "destructive",
            });
            setUsers([]);
            setPaginationInfo(null);
        } finally {
            setIsLoadingData(false);
        }
    }, [toast]); // Thêm toast vào dependency array

    // Load dữ liệu user chỉ khi đã xác thực và có quyền (là manager hoặc admin)
    useEffect(() => {
        if (!isAuthLoading && isLoggedIn && loggedInUser && (loggedInUser.role === 'manager' || loggedInUser.role === 'admin')) {
             fetchUsers(currentPage);
        } else if (!isAuthLoading && (!isLoggedIn || !loggedInUser)) {
             // Nếu đã load xong auth mà không đăng nhập, reset state bảng
             setUsers([]);
             setPaginationInfo(null);
             setIsLoadingData(false);
        }
         // Nếu loggedInUser là officer, cũng reset state vì họ không xem được trang này
         else if (!isAuthLoading && loggedInUser && loggedInUser.role === 'officer') {
              setUsers([]);
              setPaginationInfo(null);
              setIsLoadingData(false);
         }

    }, [currentPage, fetchUsers, isAuthLoading, isLoggedIn, loggedInUser]); // Thêm các dependency liên quan đến auth

    // Lọc user phía client dựa trên searchTerm
    const filteredUsers = users.filter(
        (user) =>
            (user.firstName && user.firstName.toLowerCase().includes(searchTerm.toLowerCase())) ||
            (user.lastName && user.lastName.toLowerCase().includes(searchTerm.toLowerCase())) ||
            user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
            user.email.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // --- Helper Function: Kiểm tra quyền Edit/Delete ---
     // (Giữ nguyên như trước, sử dụng loggedInUser từ hook)
    const canEditOrDeleteUser = (loggedInRole: UserRole | undefined | null, targetUserRole: UserRole): boolean => {
        if (!loggedInRole) return false;
        switch (loggedInRole) {
            case 'manager':
                return targetUserRole === 'admin' || targetUserRole === 'officer';
            case 'admin':
                return targetUserRole === 'officer';
            case 'officer':
                return false;
            default:
                return false;
        }
    };

     // --- Helper Function: Kiểm tra quyền thay đổi Role ---
    const canChangeRole = (loggedInRole: UserRole | undefined | null, currentTargetRole: UserRole, newTargetRole: UserRole): boolean => {
         if (!loggedInRole) return false;
         switch (loggedInRole) {
             case 'manager':
                 return (currentTargetRole === 'admin' || currentTargetRole === 'officer') &&
                        (newTargetRole === 'admin' || newTargetRole === 'officer');
             case 'admin':
                  return currentTargetRole === 'officer' && (newTargetRole === 'admin' || newTargetRole === 'officer');
             default:
                 return false;
         }
     };

     // --- Helper Function: Lấy tooltip cho nút Edit/Delete ---
      // (Giữ nguyên như trước, sử dụng loggedInUser từ hook)
     const getActionTooltip = (action: 'edit' | 'delete', currentUser: AppUser | null, targetUser: AppUser): string => {
        if (!currentUser) return "Đang tải thông tin người dùng..."; // Hoặc "Bạn cần đăng nhập"
        if (!isLoggedIn) return "Bạn cần đăng nhập";
        if (action === 'delete' && currentUser.id === targetUser.id) return "Không thể tự xóa chính mình";

        const canAct = canEditOrDeleteUser(currentUser.role as UserRole, targetUser.role as UserRole);

        if (canAct) {
            return action === 'edit' ? `Chỉnh sửa ${targetUser.username}` : `Xóa ${targetUser.username}`;
        } else {
             // Lý do không thể thực hiện hành động
            if (currentUser.role === 'officer') return "Bạn không có quyền thực hiện hành động này";
            if (currentUser.role === 'admin') {
                if (targetUser.role === 'admin') return "Không thể tác động đến Quản trị viên khác";
                if (targetUser.role === 'manager') return "Không thể tác động đến Tổng quản trị";
            }
             if (currentUser.role === 'manager') {
                  if (targetUser.role === 'manager') return "Không thể tác động đến Tổng quản trị khác (hoặc chính mình)";
             }
            // Kiểm tra trường hợp target là chính mình khi edit (nếu muốn disable)
            // if (action === 'edit' && currentUser.id === targetUser.id) return "Không thể tự sửa thông tin tại đây";
            return "Không có quyền thực hiện hành động này";
        }
     };


    // --- Xử lý mở Dialog ---
    const handleOpenDialog = (user: AppUser | null = null) => {
        // Chỉ cho phép mở nếu đã đăng nhập và có quyền (manager/admin)
         if (!isLoggedIn || !loggedInUser || (loggedInUser.role !== 'manager' && loggedInUser.role !== 'admin')) {
             toast({ title: "Không có quyền", description: "Bạn không có quyền thực hiện thao tác này.", variant: "destructive"});
             return;
         }
        // Kiểm tra quyền khi sửa
        if (user && !canEditOrDeleteUser(loggedInUser?.role as UserRole, user.role as UserRole)) {
             toast({ title: "Không có quyền", description: "Bạn không có quyền chỉnh sửa người dùng này.", variant: "destructive"});
             return;
        }


        setCurrentUser(user);
        if (user) {
            setFormData({
                username: user.username,
                email: user.email,
                firstName: user.firstName || "",
                lastName: user.lastName || "",
                role: (user.role === 'admin' || user.role === 'officer') ? user.role : 'officer',
                isActive: user.isActive ? "active" : "inactive",
                phone: user.phone || "",
                address: user.address || "",
                age: user.age || undefined,
                profilePic: user.profilePic || "",
                password: "",
            });
        } else {
            setFormData({
                username: "", email: "", firstName: "", lastName: "",
                role: "officer", isActive: "active", phone: "", address: "", age: undefined, profilePic: "", password: "DefaultPassword123"
            });
        }
        setIsDialogOpen(true);
    };

    // --- Xử lý thay đổi Input/Select trong Form ---
    const handleFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value, type } = e.target;
        setFormData((prev) => ({
            ...prev,
            [name]: type === 'number' ? (value === '' ? undefined : Number(value)) : value,
        }));
    };

    const handleSelectChange = (name: keyof UserFormData, value: string) => {
        if (name === 'role') {
             setFormData((prev) => ({ ...prev, [name]: value as typeof FORM_USER_ROLES[number] }));
        } else if (name === 'isActive') {
             setFormData((prev) => ({ ...prev, [name]: value as "active" | "inactive" }));
        } else {
             setFormData((prev) => ({ ...prev, [name]: value }));
        }
    };

    // --- Xử lý Lưu User (Thêm mới hoặc Cập nhật) ---
    const handleSaveUser = async () => {
        // Kiểm tra đăng nhập và role cơ bản
        if (!isLoggedIn || !loggedInUser || (loggedInUser.role !== 'manager' && loggedInUser.role !== 'admin')) {
             toast({ title: "Không có quyền", description: "Bạn không có quyền thực hiện thao tác này.", variant: "destructive"});
             return;
         }
        // --- Input Validation ---
        if (!formData.username || !formData.email || !formData.firstName || !formData.lastName || !formData.role) {
            toast({ title: "Thiếu thông tin", description: "Vui lòng nhập Username, Email, Tên, Họ và chọn Quyền hạn.", variant: "destructive" });
            return;
        }
         if (!currentUser && !formData.password) {
             toast({ title: "Thiếu thông tin", description: "Vui lòng nhập mật khẩu cho người dùng mới.", variant: "destructive" });
             return;
         }
        if (!/\S+@\S+\.\S+/.test(formData.email)) {
            toast({ title: "Lỗi", description: "Địa chỉ email không hợp lệ.", variant: "destructive" });
            return;
        }

        // *** Kiểm tra quyền trước khi gọi API ***
        if (currentUser && loggedInUser) {
            if (!canEditOrDeleteUser(loggedInUser.role as UserRole, currentUser.role as UserRole)) {
                 toast({ title: "Không có quyền", description: "Bạn không có quyền chỉnh sửa người dùng này.", variant: "destructive" });
                 return;
            }
             if (formData.role !== currentUser.role && !canChangeRole(loggedInUser.role as UserRole, currentUser.role as UserRole, formData.role)) {
                toast({ title: "Không có quyền", description: "Bạn không có quyền thay đổi vai trò của người dùng này thành vai trò đó.", variant: "destructive" });
                return;
             }
        }

        setIsLoadingData(true); // Bắt đầu loading
        try {
            const userDataToSave: Partial<AppUser> & { password?: string } = {
                username: formData.username,
                email: formData.email,
                firstName: formData.firstName,
                lastName: formData.lastName,
                role: formData.role,
                isActive: formData.isActive === "active",
                phone: formData.phone || null,
                address: formData.address || null,
                age: formData.age || null,
                profilePic: formData.profilePic || null,
                ...(formData.password && { password: formData.password }),
            };


            let status: number;
            if (currentUser) {
                 // --- Cập nhật User ---
                 if (!formData.password) {
                     delete userDataToSave.password;
                 }
                 delete userDataToSave.username;

                 // API updateUser cần được cấu hình để gửi token nếu cần
                status = await updateUser(userDataToSave, currentUser.id);
                if (status >= 200 && status < 300) {
                    toast({ title: "Thành công", description: `Đã cập nhật người dùng ${formData.username}.`, variant: "success" });
                } else {
                     // Cố gắng lấy thông báo lỗi từ API nếu có
                     const errorData = status === 403 ? "Bạn không có quyền thực hiện hành động này." : `API trả về status ${status}`;
                    throw new Error(errorData);
                }
            } else {
                // --- Thêm User mới ---
                 if (!userDataToSave.password) {
                      throw new Error("Mật khẩu là bắt buộc khi tạo người dùng mới.");
                 }
                 // API createUser cần được cấu hình để gửi token nếu cần
                status = await createUser(userDataToSave as AppUser);
                if (status >= 200 && status < 300) {
                    toast({ title: "Thành công", description: `Đã thêm người dùng ${formData.username}.`, variant: "success" });
                } else {
                     const errorData = status === 403 ? "Bạn không có quyền thực hiện hành động này." : `API trả về status ${status}`;
                    throw new Error(errorData);
                }
            }

            setIsDialogOpen(false);
            setCurrentUser(null);
            await fetchUsers(currentPage); // Tải lại danh sách

        } catch (error) {
            console.error("Failed to save user:", error);
            toast({
                title: "Lỗi",
                description: `Không thể lưu người dùng. ${error instanceof Error ? error.message : 'Lỗi không xác định.'}`,
                variant: "destructive",
            });
        } finally {
            setIsLoadingData(false); // Kết thúc loading
        }
    };


    // --- Xử lý mở Alert Xóa ---
    const handleDeleteClick = (user: AppUser) => {
         // Kiểm tra quyền trước khi mở
         if (!loggedInUser || !canEditOrDeleteUser(loggedInUser.role as UserRole, user.role as UserRole) || loggedInUser.id === user.id) {
             toast({ title: "Không có quyền", description: getActionTooltip('delete', loggedInUser ?? null, user), variant: "destructive" });
             return;
         }
        setUserToDeleteId(user.id);
        setIsAlertOpen(true);
    };

    // --- Xử lý Xác nhận Xóa ---
    const handleConfirmDelete = async () => {
        if (!userToDeleteId) return;
         // Kiểm tra đăng nhập và quyền lần cuối
         const userTarget = users.find(u => u.id === userToDeleteId);
         if (!isLoggedIn || !loggedInUser || !userTarget || !canEditOrDeleteUser(loggedInUser.role as UserRole, userTarget.role as UserRole) || loggedInUser.id === userTarget.id) {
             toast({ title: "Hành động bị chặn", description: "Không đủ quyền hoặc thông tin người dùng không hợp lệ.", variant: "destructive" });
             setIsAlertOpen(false);
             setUserToDeleteId(null);
             return;
         }

        setIsLoadingData(true); // Bắt đầu loading
        try {
             // API deleteUser cần được cấu hình để gửi token nếu cần
            const status = await deleteUser(userToDeleteId);

            if (status >= 200 && status < 300) {
                toast({ title: "Đã xóa", description: "Người dùng đã được xóa.", variant: "success" });
                setUserToDeleteId(null);
                setIsAlertOpen(false);

                // Tải lại dữ liệu
                const remainingUsers = users.filter(u => u.id !== userToDeleteId);
                if (remainingUsers.length === 0 && currentPage > 1) {
                    setCurrentPage(currentPage - 1); // Chuyển về trang trước nếu trang hiện tại trống
                     // useEffect sẽ tự gọi fetchUsers khi currentPage thay đổi
                } else {
                    await fetchUsers(currentPage); // Tải lại trang hiện tại
                }
            } else {
                const errorData = status === 403 ? "Bạn không có quyền thực hiện hành động này." : `API trả về status ${status}`;
                 throw new Error(errorData);
            }
        } catch (error) {
            console.error("Failed to delete user:", error);
            toast({
                title: "Lỗi",
                description: `Không thể xóa người dùng. ${error instanceof Error ? error.message : 'Lỗi không xác định.'}`,
                variant: "destructive",
            });
            setIsAlertOpen(false);
        } finally {
            setIsLoadingData(false); // Kết thúc loading
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
    const renderRoleBadge = (role: UserRole | string) => {
        const roleMap: Record<UserRole | 'default', { text: string; className: string; icon: React.ReactNode }> = {
            manager: { text: "Tổng quản trị", className: "border-purple-500/50 bg-purple-50 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300", icon: <ShieldCheck className="mr-1 h-3 w-3" /> },
            admin: { text: "Quản trị viên", className: "border-red-500/50 bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300", icon: <ShieldAlert className="mr-1 h-3 w-3" /> },
            officer: { text: "Viên chức", className: "border-blue-500/50 bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300", icon: <UserCog className="mr-1 h-3 w-3" /> },
            default: { text: String(role), className: "border-gray-500/50 bg-gray-50 text-gray-700 dark:bg-gray-700 dark:text-gray-300", icon: <User className="mr-1 h-3 w-3" /> },
        };
        const roleInfo = roleMap[role as UserRole] || roleMap.default;
        return <Badge variant="outline" className={cn("text-xs font-normal", roleInfo.className)}>{roleInfo.icon}{roleInfo.text}</Badge>;
    };

    const renderStatusBadge = (isActive: boolean) => {
        const statusMap: Record<string, { text: string; className: string; icon: React.ReactNode }> = {
            true: { text: "Hoạt động", className: "border-green-500/50 bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-300", icon: <UserCheck className="mr-1 h-3 w-3" /> },
            false: { text: "Ngừng hoạt động", className: "border-gray-400/50 bg-gray-50 text-gray-500 dark:bg-gray-700 dark:text-gray-400", icon: <UserX className="mr-1 h-3 w-3" /> },
        };
        const statusInfo = statusMap[String(isActive)];
        return <Badge variant="outline" className={cn("text-xs font-normal", statusInfo.className)}>{statusInfo.icon}{statusInfo.text}</Badge>;
    };


    // --- Render ---

    // 1. Xử lý trạng thái loading auth ban đầu
    if (isAuthLoading) {
        return <PageLoader message="Đang kiểm tra phiên đăng nhập..." />;
    }

    // 2. Xử lý lỗi auth hoặc chưa đăng nhập
    if (authError) {
        return <div className="container mx-auto py-10 text-center text-red-600">
                    <AlertCircle className="mx-auto h-12 w-12 mb-4" />
                    <h2 className="text-xl font-semibold mb-2">Lỗi xác thực</h2>
                    <p>{authError}</p>
                    {/* Có thể thêm nút để thử lại hoặc đăng nhập */}
                </div>;
    }
    if (!isLoggedIn || !loggedInUser) {
         return <div className="container mx-auto py-10 text-center text-orange-600">
                     <AlertCircle className="mx-auto h-12 w-12 mb-4" />
                     <h2 className="text-xl font-semibold mb-2">Yêu cầu đăng nhập</h2>
                     <p>Bạn cần đăng nhập để truy cập trang này.</p>
                     {/* Thêm Link hoặc Button để chuyển hướng đến trang đăng nhập */}
                     {/* Ví dụ: <Link href="/login"><Button>Đăng nhập</Button></Link> */}
                 </div>;
    }

    // 3. Xử lý người dùng không có quyền truy cập (ví dụ: officer)
     if (loggedInUser.role === 'officer') {
          return <div className="container mx-auto py-10 text-center text-yellow-600">
                     <ShieldX className="mx-auto h-12 w-12 mb-4" />
                     <h2 className="text-xl font-semibold mb-2">Truy cập bị từ chối</h2>
                     <p>Bạn không có quyền truy cập vào trang quản lý người dùng.</p>
                 </div>;
     }

     // 4. Render nội dung chính nếu đã đăng nhập và có quyền (manager/admin)
    return (
        <div className="container mx-auto py-6 px-4 md:px-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
                <h1 className="text-3xl font-bold tracking-tight flex items-center">
                    <Users className="mr-3 h-7 w-7" />
                    Quản lý người dùng
                </h1>
                {/* Nút thêm chỉ hiển thị cho manager/admin */}
                 <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                     <DialogTrigger asChild>
                         <Button onClick={() => handleOpenDialog(null)} disabled={isLoadingData}>
                             <PlusCircle className="mr-2 h-4 w-4" /> Thêm người dùng mới
                         </Button>
                     </DialogTrigger>
                     <DialogContent className="sm:max-w-[580px]">
                            <DialogHeader>
                                <DialogTitle>{currentUser ? "Chỉnh sửa người dùng" : "Thêm người dùng mới"}</DialogTitle>
                                <DialogDescription>
                                    {currentUser ? `Cập nhật thông tin cho ${currentUser.username}.` : "Nhập thông tin cho người dùng mới (Admin hoặc Officer)."}
                                </DialogDescription>
                            </DialogHeader>
                             {/* Nội dung Form Dialog giữ nguyên như trước */}
                            <div className="grid gap-4 py-4 max-h-[65vh] overflow-y-auto pr-6">
                                {/* Username (Không cho sửa) */}
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="username" className="text-right">Username <span className="text-red-500">*</span></Label>
                                    <Input id="username" name="username" value={formData.username} onChange={handleFormChange} className="col-span-3" placeholder="Tên đăng nhập" disabled={!!currentUser} required />
                                </div>
                                {/* Email */}
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="email" className="text-right">Email <span className="text-red-500">*</span></Label>
                                    <Input id="email" name="email" type="email" value={formData.email} onChange={handleFormChange} className="col-span-3" placeholder="vidu@example.com" required />
                                </div>
                                {/* First Name */}
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="firstName" className="text-right">Tên <span className="text-red-500">*</span></Label>
                                    <Input id="firstName" name="firstName" value={formData.firstName} onChange={handleFormChange} className="col-span-3" placeholder="Nhập tên" required />
                                </div>
                                {/* Last Name */}
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="lastName" className="text-right">Họ <span className="text-red-500">*</span></Label>
                                    <Input id="lastName" name="lastName" value={formData.lastName} onChange={handleFormChange} className="col-span-3" placeholder="Nhập họ" required />
                                </div>
                                {/* Role (Chỉ Admin/Officer) */}
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="role" className="text-right">Quyền hạn <span className="text-red-500">*</span></Label>
                                    <Select
                                        value={formData.role}
                                        onValueChange={(value) => handleSelectChange('role', value)}
                                        disabled={
                                             isLoadingData || // Disable khi đang lưu
                                             (!!currentUser && loggedInUser && !canChangeRole(loggedInUser.role as UserRole, currentUser.role as UserRole, formData.role))
                                        }
                                    >
                                        <SelectTrigger id="role" className="col-span-3">
                                            <SelectValue placeholder="Chọn quyền hạn" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {FORM_USER_ROLES.map((role) => (
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
                                    <Select value={formData.isActive} onValueChange={(value) => handleSelectChange('isActive', value as "active" | "inactive")} disabled={isLoadingData}>
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
                                    <Input id="phone" name="phone" value={formData.phone || ''} onChange={handleFormChange} className="col-span-3" placeholder="(Tùy chọn)" disabled={isLoadingData} />
                                </div>
                                {/* Address */}
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="address" className="text-right">Địa chỉ</Label>
                                    <Input id="address" name="address" value={formData.address || ''} onChange={handleFormChange} className="col-span-3" placeholder="(Tùy chọn)" disabled={isLoadingData}/>
                                </div>
                                {/* Age */}
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="age" className="text-right">Tuổi</Label>
                                    <Input id="age" name="age" type="number" min={18} value={formData.age == null ? '' : formData.age} onChange={handleFormChange} className="col-span-3" placeholder="(Tùy chọn)" disabled={isLoadingData}/>
                                </div>
                                {/* Avatar URL */}
                                {/* <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="profilePic" className="text-right">Avatar URL</Label>
                                    <Input id="profilePic" name="profilePic" value={formData.profilePic || ''} onChange={handleFormChange} className="col-span-3" placeholder="(Tùy chọn) URL ảnh đại diện" disabled={isLoadingData}/>
                                </div> */}

                                {/* Password */}
                                {!currentUser && (
                                    <div className="grid grid-cols-4 items-center gap-4">
                                        <Label htmlFor="password" className="text-right">Mật khẩu <span className="text-red-500">*</span></Label>
                                        <Input id="password" name="password" minLength={8} type="text" value={formData.password || ''} onChange={handleFormChange} className="col-span-3" placeholder="Nhập mật khẩu" required disabled={isLoadingData}/>
                                    </div>
                                )}
                                {currentUser && (
                                     <div className="grid grid-cols-4 items-center gap-4">
                                         <Label htmlFor="password" className="text-right">Đặt lại mật khẩu</Label>
                                         <Input id="password" name="password" minLength={8} type="text" value={formData.password || ''} onChange={handleFormChange} className="col-span-3" placeholder="(Để trống nếu không đổi)" disabled={isLoadingData}/>
                                     </div>
                                )}

                            </div>
                            <DialogFooter>
                                <DialogClose asChild>
                                    <Button type="button" variant="outline" disabled={isLoadingData}>Hủy</Button>
                                </DialogClose>
                                <Button type="button" onClick={handleSaveUser} disabled={isLoadingData}>
                                    {isLoadingData ? "Đang lưu..." : (currentUser ? "Lưu thay đổi" : "Tạo người dùng")}
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
                        placeholder="Tìm theo tên, username, email..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-8 w-full sm:w-[300px]"
                         disabled={isLoadingData} // Disable search khi đang load/lưu/xóa
                    />
                </div>
            </div>


            {/* --- Bảng Danh sách Người dùng --- */}
            <div className="border rounded-lg overflow-hidden">
                <Table>
                    <TableCaption>
                         {/* Chỉ hiển thị caption nếu không có lỗi và đã đăng nhập */}
                         {isLoggedIn && (isLoadingData
                             ? "Đang tải danh sách người dùng..."
                             : filteredUsers.length > 0
                                 ? `Hiển thị ${filteredUsers.length} trên tổng số ${paginationInfo?.totalItems || 0} người dùng.`
                                 : "Không tìm thấy người dùng nào."
                         )}
                    </TableCaption>
                    <TableHeader className="bg-muted/50 dark:bg-muted/30">
                        <TableRow>
                            <TableHead className="w-[80px]">Avatar</TableHead>
                            <TableHead>Tên</TableHead>
                            <TableHead>Username</TableHead>
                            <TableHead>Email</TableHead>
                            <TableHead>Quyền hạn</TableHead>
                            <TableHead>Trạng thái</TableHead>
                            <TableHead>Cập nhật lần cuối</TableHead>
                            <TableHead className="text-right w-[120px]">Hành động</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {isLoadingData && ( // Hiển thị loading khi đang fetch/save/delete
                            <TableRow>
                                <TableCell colSpan={8} className="h-24 text-center text-muted-foreground">
                                    Đang xử lý...
                                </TableCell>
                            </TableRow>
                        )}
                        {!isLoadingData && filteredUsers.length > 0 ? (
                            filteredUsers.map((user) => {
                                // Tính toán trạng thái nút dựa trên loggedInUser lấy từ hook
                                const canAct = loggedInUser ? canEditOrDeleteUser(loggedInUser.role as UserRole, user.role as UserRole) : false;
                                const isSelf = loggedInUser?.id === user.id;
                                // Luôn disable nút khi đang có hành động khác diễn ra (isLoadingData)
                                const disableEdit = isLoadingData || !canAct || isSelf;
                                const disableDelete = isLoadingData || !canAct || isSelf;
                                const editTooltip = getActionTooltip('edit', loggedInUser, user);
                                const deleteTooltip = getActionTooltip('delete', loggedInUser, user);

                                return (
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
                                        <TableCell>{renderRoleBadge(user.role as UserRole)}</TableCell>
                                        <TableCell>{renderStatusBadge(user.isActive)}</TableCell>
                                        <TableCell className="text-muted-foreground text-sm">
                                            {user.updatedAt ? new Date(user.updatedAt).toLocaleString() : "Chưa có"}
                                        </TableCell>
                                        <TableCell className="text-right">
                                            {/* Edit Button */}
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="text-muted-foreground hover:text-primary h-8 w-8"
                                                onClick={() => { if (!disableEdit) handleOpenDialog(user); }}
                                                disabled={disableEdit}
                                                title={editTooltip}
                                            >
                                                <Edit className="h-4 w-4" />
                                                <span className="sr-only">Sửa</span>
                                            </Button>

                                            {/* Delete Button + Alert */}
                                            <AlertDialog open={isAlertOpen && userToDeleteId === user.id} onOpenChange={(open) => { if (!open) setUserToDeleteId(null); setIsAlertOpen(open); }}>
                                                <AlertDialogTrigger asChild>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="text-muted-foreground hover:text-destructive h-8 w-8"
                                                        onClick={(e) => {
                                                             if (disableDelete) {
                                                                 e.preventDefault();
                                                                  toast({ title: "Thông báo", description: deleteTooltip, variant: "default" });
                                                                 return;
                                                             }
                                                            handleDeleteClick(user);
                                                        }}
                                                        disabled={disableDelete}
                                                        title={deleteTooltip}
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
                                                         {/* Disable nút khi đang loading */}
                                                        <AlertDialogCancel disabled={isLoadingData}>Hủy</AlertDialogCancel>
                                                        <AlertDialogAction onClick={handleConfirmDelete} disabled={isLoadingData} className={cn("bg-destructive hover:bg-destructive/90", isLoadingData && "opacity-50 cursor-not-allowed")}>
                                                            {isLoadingData ? "Đang xóa..." : "Xác nhận xóa"}
                                                        </AlertDialogAction>
                                                    </AlertDialogFooter>
                                                </AlertDialogContent>
                                            </AlertDialog>
                                        </TableCell>
                                    </TableRow>
                                );
                            })
                        ) : (
                             !isLoadingData && ( // Chỉ hiển thị "Không tìm thấy" khi không loading data
                                <TableRow>
                                    <TableCell colSpan={8} className="h-24 text-center text-muted-foreground">
                                        {searchTerm ? "Không tìm thấy người dùng phù hợp." : "Chưa có người dùng nào trong hệ thống."}
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
                        // Có thể thêm prop disabled cho Pagination nếu isLoadingData = true
                         //isDisabled={isLoadingData}
                    />
                </div>
            )}

        </div>
    );
}