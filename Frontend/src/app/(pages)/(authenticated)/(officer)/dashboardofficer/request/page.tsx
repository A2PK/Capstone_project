// src/app/(pages)/(authenticated)/(officer)/dashboardofficer/request/page.tsx

"use client"

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation"; // Import nếu cần
import { Request } from "@/types/requestofficial"; // Đảm bảo type đúng
import { FileInfo } from "@/types/file"; // Kiểu FileInfo
import { User } from "@/types/user"; // Import kiểu User

// UI Components
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertCircle, FileText, Loader2, Eye, Trash2 } from "lucide-react"; // Icons (Trash2 cho nút xóa)
import {
    AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog"; // Alert cho xác nhận xóa
import {
    Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose,
} from "@/components/ui/dialog"; // Dialog cho xem trước file

// Libraries
import * as XLSX from "xlsx"; // Để đọc file Excel/CSV
import { format, parseISO, isValid } from 'date-fns'; // Để định dạng ngày
import axios from "axios"; // Để fetch nội dung file
import { cn } from "@/lib/utils"; // Tiện ích classnames

// Hooks & API Functions
import { useToast } from "@/hooks/use-toast";
// Đảm bảo import đúng các hàm API request và file
import { findRequestsByUserParticipant, deleteRequest } from "@/lib/request";
import { getMultipleFilesByFileIds } from "@/lib/file";
import { generateProxyUrl } from "@/lib/article"; // Đảm bảo đường dẫn đúng

// Interface kết hợp Request và chi tiết file
interface RequestWithFileDetails extends Request {
    fileDetails?: FileInfo[] | null; // Mảng chứa thông tin chi tiết file
}

// Helper lấy style badge trạng thái
const getStatusTailwindClasses = (status: Request['status']): { border: string; } => {
    switch (status) {
        case "pending": return { border: "border-yellow-500" };
        case "approved": return { border: "border-green-500" };
        case "rejected": return { border: "border-red-500" };
        case "cancelled": return { border: "border-gray-500" }; // Vẫn giữ nếu backend có trạng thái này
        default: return { border: "border-gray-400" };
    }
};

// Helper định dạng ngày
function formatRequestDate(dateString: string): string {
    try {
        const date = parseISO(dateString);
        return isValid(date) ? format(date, 'dd/MM/yyyy HH:mm') : 'Ngày không hợp lệ';
    } catch {
        return 'Ngày không hợp lệ';
    }
}

// --- Component Chính ---
export default function RequestsPage() {
    const { toast } = useToast();
    const [requests, setRequests] = useState<RequestWithFileDetails[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [deletingId, setDeletingId] = useState<string | null>(null); // State cho biết ID đang được xóa
    const [filePreviewError, setFilePreviewError] = useState<string | null>(null);
    // Auth state
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [isSessionLoading, setIsSessionLoading] = useState(true);

    // STATE CHO FILE PREVIEW
    const [selectedFileContent, setSelectedFileContent] = useState<any[]>([]);
    const [fileHeaders, setFileHeaders] = useState<string[]>([]);
    const [fileContentLoading, setFileContentLoading] = useState(false);
    const [previewingFileInfo, setPreviewingFileInfo] = useState<FileInfo | null>(null);
    const [fileDialogOpen, setFileDialogOpen] = useState(false);

    // --- Fetch Session Info ---
    useEffect(() => {
        const fetchSession = async () => {
            setIsSessionLoading(true);
            try {
                const response = await fetch('/api/auth/session');
                if (!response.ok) throw new Error('Lỗi fetch session');
                const data: { isLoggedIn: boolean; user: User | null; token: string | null } = await response.json();
                if (data.isLoggedIn && data.user && data.token) {
                    setUser(data.user);
                    setToken(data.token);
                    console.log("Officer session loaded:", data.user.id);
                } else {
                    setError("Vui lòng đăng nhập để xem yêu cầu.");
                    setUser(null); // Đảm bảo user là null nếu không đăng nhập
                    setToken(null);
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : "Lỗi tải thông tin phiên.");
                console.error("Session fetch error:", err);
                setUser(null);
                setToken(null);
            } finally {
                setIsSessionLoading(false);
            }
        };
        fetchSession();
    }, []);

    // --- Fetch Requests và File Details ---
    useEffect(() => {
        if (user && token) { // Chỉ fetch khi có user và token
            const loadRequestsAndFiles = async () => {
                setLoading(true);
                setError(null);
                try {
                    console.log("Fetching requests for user:", user.id);
                    // Officer xem các yêu cầu họ đã gửi
                    const fetchedRequests = await findRequestsByUserParticipant(token, user.id, { sortBy: 'created_at', sortDesc: true });
                    console.log("Fetched raw requests:", fetchedRequests);

                    // Fetch chi tiết file cho mỗi yêu cầu
                    const requestsWithDetails = await Promise.all(
                        (fetchedRequests || []).map(async (req): Promise<RequestWithFileDetails> => {
                            if (req.fileIds && req.fileIds.length > 0 && token) {
                                try {
                                    console.log(`Workspaceing file details for request ${req.id}, file IDs:`, req.fileIds);
                                    const filesInfo = await getMultipleFilesByFileIds(token, { ids: req.fileIds });
                                    console.log(`Workspaceed file details for request ${req.id}:`, filesInfo);
                                    return { ...req, fileDetails: filesInfo || null };
                                } catch (fileError) {
                                    console.error(`Failed to fetch file details for request ${req.id}:`, fileError);
                                    return { ...req, fileDetails: null }; // Gán null nếu lỗi fetch file
                                }
                            }
                            return { ...req, fileDetails: null }; // Gán null nếu không có fileIds
                        })
                    );

                    console.log("Requests with details:", requestsWithDetails);
                    setRequests(requestsWithDetails);

                } catch (err) {
                    setError(err instanceof Error ? err.message : "Không thể tải danh sách yêu cầu.");
                    console.error("Error loading requests:", err);
                    setRequests([]); // Set mảng rỗng nếu lỗi
                } finally {
                    setLoading(false);
                }
            };
            loadRequestsAndFiles();
        } else if (!isSessionLoading && !user) {
             // Nếu load session xong mà không có user thì báo lỗi và dừng loading
             setError("Vui lòng đăng nhập để xem yêu cầu.");
             setLoading(false);
        }
    }, [user, token, isSessionLoading]); // Dependencies: user, token, isSessionLoading

    // --- *** HÀM XÓA YÊU CẦU (ĐÃ SỬA LOGIC) *** ---
    const handleDeleteRequest = async (id: string) => {
        if (!token) {
            toast({ variant: "destructive", title: "Lỗi", description: "Mất phiên đăng nhập." });
            return;
        }
        setDeletingId(id); // Đặt ID đang xóa để disable nút
        try {
            // Gọi API deleteRequest (thực hiện hard delete trên backend)
            const status = await deleteRequest(token, id);

            if (status >= 200 && status < 300) { // Kiểm tra status thành công từ API
                // *** THAY ĐỔI CẬP NHẬT STATE: Dùng filter để xóa khỏi danh sách ***
                setRequests(prev => prev.filter((req) => req.id !== id));

                // *** THAY ĐỔI THÔNG BÁO TOAST ***
                toast({ variant: "success", title: "Thành công", description: "Đã xoá yêu cầu." });

            } else {
                // Xử lý trường hợp API trả về status lỗi nhưng không throw error
                if (status === 404) {
                     toast({ variant: "warning", title: "Thông báo", description: "Yêu cầu này có thể đã được xóa trước đó." });
                     // Vẫn xóa khỏi danh sách trên UI nếu còn tồn tại
                      setRequests(prev => prev.filter((req) => req.id !== id));
                } else {
                    // Ném lỗi nếu là status code lỗi khác
                    throw new Error(`Lỗi xoá yêu cầu: Status code ${status}`);
                }
            }
        } catch (error) {
            console.error("Xoá yêu cầu thất bại.", error);
            toast({ variant: "destructive", title: "Lỗi", description: error instanceof Error ? error.message : "Không thể xoá yêu cầu." });
        } finally {
            setDeletingId(null); // Reset state disable nút
        }
    };

    // --- Xem Trước File ---
    const handleFilePreview = async (request: RequestWithFileDetails, fileId: string) => {
        if (!token || !fileId) {
            toast({ title: "Lỗi", description: "Thiếu thông tin cần thiết.", variant: "destructive"});
            return;
        }
        const fileInfo = request.fileDetails?.find(f => f.id === fileId);
        if (!fileInfo) {
            toast({ title: "Lỗi", description: `Không tìm thấy chi tiết file ID: ${fileId}.`, variant: "destructive"});
            return;
        }

        // Reset state cho dialog mới
        setPreviewingFileInfo(fileInfo);
        setSelectedFileContent([]);
        setFileHeaders([]);
        setFileContentLoading(true);
        setFileDialogOpen(true);
        setFilePreviewError(null);

        try {
            // Kiểm tra loại file
            const fileNameLower = fileInfo.name?.toLowerCase() || '';
            const isSupportedPreviewType = fileNameLower.endsWith('.xlsx') || fileNameLower.endsWith('.xls') || fileNameLower.endsWith('.csv') || fileInfo.type?.includes('spreadsheet') || fileInfo.type?.includes('csv');

            if (!isSupportedPreviewType) {
                throw new Error(`Xem trước chỉ hỗ trợ file Excel/CSV. File này có vẻ là loại khác.`);
            }

            // Fetch và parse file Excel/CSV
            const proxyUrl = generateProxyUrl(fileInfo.url);
            const response = await axios.get(proxyUrl, { responseType: 'arraybuffer' });
            const arrayBuffer: ArrayBuffer = response.data;
            const workbook = XLSX.read(arrayBuffer, { type: 'array' });
            const firstSheetName = workbook.SheetNames[0];
            if (!firstSheetName) throw new Error("File không có sheet nào.");
            const worksheet = workbook.Sheets[firstSheetName];
            const data: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" }); // Lấy header

            if (!data || data.length < 1) {
                setSelectedFileContent([]); // File rỗng
                setFileHeaders([]);
                console.warn("Preview: File rỗng.");
            } else {
                const headers = data[0].map(String); // Dòng đầu là header
                // Lấy nội dung dưới dạng object array (mặc định của sheet_to_json)
                const content = XLSX.utils.sheet_to_json(worksheet, { defval: "" });
                setFileHeaders(headers);
                setSelectedFileContent(content);
            }

        } catch (error: any) {
            console.error("Lỗi khi xem trước file:", error);
            toast({
                title: "Không thể xem trước file",
                description: error.message || "Không thể đọc hoặc tải dữ liệu.",
                variant: "warning",
            });
            setSelectedFileContent([]);
            setFileHeaders([]);
            setFilePreviewError("File có thể đã bị xoá, không thể truy cập hoặc không đúng định dạng. Xin hãy kiểm tra lại.");
        } finally {
            setFileContentLoading(false); // Luôn dừng loading
        }
    };

    // Status Badge function
    const getStatusBadge = (status: Request['status']) => {
        switch (status) {
            case "pending": return (<Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">Đang đợi</Badge>);
            case "approved": return (<Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">Đã duyệt</Badge>);
            case "rejected": return (<Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">Từ chối</Badge>);
            // Trạng thái cancelled có thể vẫn tồn tại nếu backend có logic soft delete khác
            case "cancelled": return (<Badge variant="outline" className="bg-gray-50 text-gray-700 border-gray-200">Đã huỷ</Badge>);
            default: return <Badge variant="outline">Không xác định</Badge>;
        }
    };

    // --- Render Logic ---
    if (isSessionLoading || loading) {
        return (
            <div className="container mx-auto py-10 flex flex-col items-center justify-center min-h-[50vh]">
                <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                <p>Đang tải dữ liệu...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mx-auto py-10 flex flex-col items-center justify-center min-h-[50vh]">
                <AlertCircle className="h-12 w-12 text-destructive mb-4" />
                <h2 className="text-xl font-semibold mb-2 text-destructive">Lỗi tải dữ liệu</h2>
                <p className="text-muted-foreground mb-6">{error}</p>
                <Button onClick={() => window.location.reload()}>Thử lại</Button>
            </div>
        );
    }

    // ---- Main Render ----
    return (
        <div className="container mx-auto py-10">
            {/* Header */}
            <div className="flex justify-between items-center mb-8">
                <h1 className="text-3xl font-bold">Yêu cầu của tôi</h1>
                <Button asChild>
                    <Link href="/dashboardofficer/request/new">Yêu cầu mới</Link>
                </Button>
            </div>

            {/* Empty State */}
            {requests.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center border rounded-lg bg-card text-card-foreground shadow-sm">
                    <FileText className="h-12 w-12 text-muted-foreground mb-4" />
                    <h2 className="text-xl font-semibold mb-2">Không có yêu cầu nào</h2>
                    <p className="text-muted-foreground mb-6">Bạn chưa tạo yêu cầu nào.</p>
                    <Button asChild>
                        <Link href="/dashboardofficer/request/new">Tạo yêu cầu đầu tiên</Link>
                    </Button>
                </div>
            ) : (
                // ---- Request List ----
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                    {requests.map((request) => (
                        <Card key={request.id} className={cn("border-l-4 flex flex-col", `border-l-${getStatusTailwindClasses(request.status).border.split('-')[1]}-500`)}>
                            {/* Card Header */}
                            <CardHeader className="pb-3">
                                <div className="flex justify-between items-start gap-2">
                                    <CardTitle className="text-lg font-semibold leading-tight">{request.title}</CardTitle>
                                    {getStatusBadge(request.status)}
                                </div>
                                <p className="text-xs text-muted-foreground pt-1">
                                    Mã: <span className="font-mono">{request.id}</span> | Ngày: {formatRequestDate(request.createdAt)}
                                </p>
                            </CardHeader>
                            {/* Card Content */}
                            <CardContent className="pt-2 flex-grow">
                                <div className="space-y-3">
                                    {/* Description */}
                                    {request.description && (
                                        <div className="space-y-1">
                                            <h4 className="text-sm font-medium text-muted-foreground">Nội dung:</h4>
                                            <p className="text-sm text-foreground line-clamp-3">{request.description}</p>
                                        </div>
                                    )}
                                    {/* File Attachments */}
                                    {request.fileDetails && request.fileDetails.length > 0 && (
                                        <div className="space-y-1 pt-2 border-t">
                                            <h4 className="text-sm font-medium text-muted-foreground">File đính kèm:</h4>
                                            <div className="flex flex-wrap gap-2 mt-1">
                                                {request.fileDetails.map((fileInfo) => (
                                                    <Button
                                                        key={fileInfo.id}
                                                        variant="outline"
                                                        size="sm"
                                                        className="flex items-center gap-1.5 text-xs h-8"
                                                        onClick={() => handleFilePreview(request, fileInfo.id)}
                                                        title={`Xem trước ${fileInfo.name}`}
                                                    >
                                                        <Eye className="h-3.5 w-3.5" />
                                                        <span className="truncate max-w-[100px]">
                                                            {fileInfo.name || `File ID: ${fileInfo.id}`}
                                                        </span>
                                                    </Button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {/* Fallback nếu lỗi fetch file details */}
                                    {!request.fileDetails && request.fileIds && request.fileIds.length > 0 && (
                                        <div className="space-y-1 pt-2 border-t">
                                             <h4 className="text-sm font-medium text-muted-foreground">File IDs đính kèm:</h4>
                                             <div className="flex flex-wrap gap-1 mt-1">
                                                 {request.fileIds.map((fileId, index) => (
                                                     <Badge key={index} variant="secondary" className="text-xs font-mono">{fileId}</Badge>
                                                 ))}
                                             </div>
                                             <p className="text-xs text-destructive italic">Lỗi tải chi tiết file.</p>
                                         </div>
                                    )}
                                </div>
                            </CardContent>
                            {/* --- CARD FOOTER (ĐÃ SỬA NÚT VÀ DIALOG CHO CHỨC NĂNG XOÁ) --- */}
                            {/* Chỉ hiện nút xóa khi đang chờ duyệt */}
                            {request.status === "pending" && (
                                <CardFooter className="pt-3 border-t mt-auto">
                                    <AlertDialog>
                                        <AlertDialogTrigger asChild>
                                            <Button variant="destructive" size="sm" disabled={deletingId === request.id} className="bg-red-600 hover:bg-red-700 text-white">
                                                {deletingId === request.id ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Trash2 className="mr-1.5 h-3.5 w-3.5" />} {/* Icon Xóa */}
                                                Xoá yêu cầu
                                            </Button>
                                        </AlertDialogTrigger>
                                        <AlertDialogContent>
                                            <AlertDialogHeader>
                                                <AlertDialogTitle>Xác nhận Xoá Yêu Cầu?</AlertDialogTitle>
                                                <AlertDialogDescription>
                                                    Bạn có chắc muốn <span className="font-semibold text-red-600">xoá vĩnh viễn</span> yêu cầu &quot;{request.title}&quot;? Hành động này sẽ không thể hoàn tác.
                                                </AlertDialogDescription>
                                            </AlertDialogHeader>
                                            <AlertDialogFooter>
                                                <AlertDialogCancel>Không</AlertDialogCancel>
                                                {/* Gọi hàm handleDeleteRequest */}
                                                <AlertDialogAction onClick={() => handleDeleteRequest(request.id)} disabled={!!deletingId} className="bg-destructive hover:bg-destructive/90">
                                                    Đồng ý xoá
                                                </AlertDialogAction>
                                            </AlertDialogFooter>
                                        </AlertDialogContent>
                                    </AlertDialog>
                                </CardFooter>
                            )}
                        </Card>
                    ))}
                </div>
            )}

            {/* --- FILE PREVIEW DIALOG --- */}
            <Dialog open={fileDialogOpen} onOpenChange={setFileDialogOpen}>
                <DialogContent className="max-w-none w-[95vw] h-[90vh] p-4 flex flex-col">
                    {/* Dialog Header */}
                    <DialogHeader className="flex-shrink-0 pb-2 border-b">
                        <DialogTitle>Xem trước file: {previewingFileInfo?.name || 'Đang tải...'}</DialogTitle>
                        {previewingFileInfo && (
                            <DialogDescription className="text-xs text-muted-foreground">
                                ID: {previewingFileInfo.id} | Loại: {previewingFileInfo.type} | Kích thước: {(Number(previewingFileInfo.size) / 1024).toFixed(1)} KB
                            </DialogDescription>
                        )}
                    </DialogHeader>
                    {/* Dialog Body - Content Area */}
                    <div className="flex-grow overflow-auto mt-4 border rounded-md relative bg-white">
                        {/* Loading indicator */}
                        {fileContentLoading && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 z-20">
                                <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                                <p>Đang tải dữ liệu file...</p>
                            </div>
                        )}
                        {/* Error Message */}
                        {!fileContentLoading && filePreviewError && (
                            <div className="flex flex-col items-center justify-center h-full text-destructive p-6">
                                <AlertCircle className="h-10 w-10 mx-auto mb-3" />
                                <p className="text-center font-medium">Không thể xem trước file</p>
                                <p className="text-center text-sm mt-1">{filePreviewError}</p>
                            </div>
                        )}
                        {/* Table Preview */}
                        {!fileContentLoading && !filePreviewError && selectedFileContent.length > 0 && fileHeaders.length > 0 ? (
                            <table className="min-w-full text-xs border-collapse">
                                <thead className="bg-slate-100 sticky top-0 z-10 shadow-sm">
                                    <tr>
                                        {fileHeaders.map((key) => (
                                            <th key={key} className="px-3 py-2 border border-slate-200 font-semibold text-slate-700 text-left whitespace-nowrap">
                                                {key}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {selectedFileContent.map((row, idx) => (
                                        <tr key={idx} className="hover:bg-slate-50">
                                            {fileHeaders.map((header) => (
                                                <td key={`${idx}-${header}`} className="px-3 py-1.5 border-x border-slate-200 whitespace-nowrap text-slate-600">
                                                    {String(row[header] ?? '')}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        ) : (
                             // Empty state inside dialog
                             !fileContentLoading && !filePreviewError && (
                                <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-6">
                                    <FileText className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                                    <p className="text-center">Không có dữ liệu để hiển thị hoặc file rỗng.</p>
                                </div>
                            )
                        )}
                    </div>
                    {/* Dialog Footer */}
                    <DialogFooter className="flex-shrink-0 border-t pt-4 mt-4 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:justify-end sm:space-x-2">
                       <DialogClose asChild>
                           <Button variant="outline" className="w-full sm:w-auto">Đóng</Button>
                       </DialogClose>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
            {/* --- END FILE PREVIEW DIALOG --- */}
        </div>
    );
}