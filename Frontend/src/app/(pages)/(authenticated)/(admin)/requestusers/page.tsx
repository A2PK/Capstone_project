// src/app/(pages)/(authenticated)/(admin)/dashboardadmin/requests/page.tsx (Adjust path as needed)
"use client"
import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation"; // Import if needed for navigation
import { Request } from "@/types/requestofficial"; // Your Request type
import { FileInfo } from "@/types/file"; // Your FileInfo type
import { User } from "@/types/user"; // Your User type
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertCircle, FileText, Loader2, Eye, X, Database, Check, Upload, UserCircle } from "lucide-react";
import { findRequestsByUserParticipant, updateRequest } from "@/lib/request"; // Your request API functions
import { getMultipleFilesByFileIds, importFileToDataBase } from "@/lib/file"; // Your file API functions
import { getUserById } from "@/lib/user"; // Your user API function
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter,
    DialogClose,
} from "@/components/ui/dialog";
import * as XLSX from "xlsx"; // For reading Excel/CSV
import { useToast } from "@/hooks/use-toast";
import { format } from 'date-fns'; // For formatting dates
import axios from "axios";

// Interface combining Request with optional sender details
// Ensure this correctly extends the 'Request' type you provided
interface RequestWithSender extends Request {
    senderDetails?: User | null;
}

export default function AdminRequestsPage() {
    const { toast } = useToast();
    const [requests, setRequests] = useState<RequestWithSender[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // File Preview State
    const [selectedFileContent, setSelectedFileContent] = useState<any[]>([]);
    const [fileHeaders, setFileHeaders] = useState<string[]>([]);
    const [fileContentLoading, setFileContentLoading] = useState(false);
    const [previewingFileInfo, setPreviewingFileInfo] = useState<FileInfo | null>(null);
    const [fileDialogOpen, setFileDialogOpen] = useState(false);

    // Action State
    const [activeRequestId, setActiveRequestId] = useState<string | null>(null); // For approve/reject
    const [actionLoading, setActionLoading] = useState<Record<string, 'approve' | 'reject' | 'import' | null>>({}); // Track loading per request/action
    const [alertOpen, setAlertOpen] = useState(false); // For approve/reject confirmation
    // State to hold the action type for the confirmation dialog
    const [alertAction, setAlertAction] = useState<'approve' | 'reject' | null>(null);
    const [addToDatabaseAlertOpen, setAddToDatabaseAlertOpen] = useState(false);
    const [importSuccess, setImportSuccess] = useState(false); // Track import success for the button state

    // Session State
    const [token, setToken] = useState<string | null>(null);
    const [currentAdmin, setCurrentAdmin] = useState<User | null>(null);

    // --- Fetch Session Info ---
    useEffect(() => {
        const fetchSession = async () => {
            try {
                const response = await fetch('/api/auth/session');
                if (!response.ok) throw new Error('Failed to fetch session');
                const data: { isLoggedIn: boolean; user: User | null; token: string | null } = await response.json();
                // Ensure user exists and is an admin
                if (data.isLoggedIn && data.user && data.token && data.user.role === 'admin') {
                    setCurrentAdmin(data.user);
                    setToken(data.token);
                    console.log("Admin session loaded:", data.user.id);
                } else {
                    setError("Không thể lấy thông tin phiên đăng nhập Admin hoặc bạn không có quyền truy cập.");
                    setLoading(false); // Stop loading if session fails
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : "Lỗi tải thông tin phiên.");
                setLoading(false);
                console.error("Session fetch error:", err);
            }
        };
        fetchSession();
    }, []);

    // --- Fetch Requests for the Admin ---
    useEffect(() => {
        // Only run if token and admin user info are available
        if (token && currentAdmin) {
            const fetchRequests = async () => {
                setLoading(true);
                setError(null);
                try {
                    // Fetch requests where the current admin is the receiver
                    // Using options for sorting by creation date descending
                    const fetchedRequests = await findRequestsByUserParticipant(token, currentAdmin.id, { sortBy: 'created_at', sortDesc: true });

                    // Filter for requests where the current user is the receiver
                    // This might be redundant if findRequestsByUserParticipant already handles this,
                    // but it ensures correctness based on the function name. Adjust if needed.
                    const receivedRequests = fetchedRequests.filter(req => req.receiverId === currentAdmin.id);

                    // Fetch sender details for each request concurrently
                    const requestsWithSenders = await Promise.all(
                        receivedRequests.map(async (req): Promise<RequestWithSender> => {
                            try {
                                // Assuming getUserById doesn't need token based on your snippet
                                // If it does, pass the token here.
                                const sender = await getUserById(req.senderId);
                                // FIXED (Error 4): Ensure the returned object matches RequestWithSender
                                // The spread operator `...req` correctly includes all properties from `req` (which is of type Request)
                                return { ...req, senderDetails: sender };
                            } catch (userError) {
                                console.error(`Failed to fetch sender ${req.senderId} for request ${req.id}:`, userError);
                                // Return the original request with null sender details on error
                                return { ...req, senderDetails: null };
                            }
                        })
                    );

                    setRequests(requestsWithSenders);
                    console.log("Fetched requests for admin:", requestsWithSenders);
                } catch (err) {
                    console.error("Failed to fetch requests:", err);
                    setError(err instanceof Error ? err.message : "Không thể tải danh sách yêu cầu.");
                    setRequests([]); // Clear requests on error
                } finally {
                    setLoading(false);
                }
            };
            fetchRequests();
        } else if (!token && !loading && !currentAdmin) { // Handle case where session loaded but no token/admin
             // Avoid setting error if still loading session initially
             if (!loading) {
                 setError("Thông tin phiên không hợp lệ để tải yêu cầu.");
                 setLoading(false);
             }
        }
    }, [loading, token, currentAdmin]); // Rerun when token/admin is available


    // --- Helper Functions ---
    const getStatusBadge = (status: Request["status"]) => {
        // (Keep the existing badge logic)
        switch (status) {
            case "pending": return <Badge variant="outline" className="bg-yellow-100 text-yellow-800 border-yellow-300">Đang đợi</Badge>;
            case "approved": return <Badge variant="outline" className="bg-green-100 text-green-800 border-green-300">Đã duyệt</Badge>;
            case "rejected": return <Badge variant="outline" className="bg-red-100 text-red-800 border-red-300">Bị từ chối</Badge>;
            case "cancelled": return <Badge variant="outline" className="bg-gray-100 text-gray-800 border-gray-300">Đã hủy</Badge>;
            default: return <Badge variant="secondary">Không xác định</Badge>;
        }
    };

    const formatDate = (dateString: string | undefined) => {
        if (!dateString) return 'N/A';
        try {
            // Use date-fns to format the date string
            return format(new Date(dateString), 'dd/MM/yyyy HH:mm');
        } catch {
            // Fallback if the date string is invalid
            return dateString;
        }
    };

    // --- Action Handlers ---

    const handleUpdateRequestStatus = async () => {
        if (!activeRequestId || !alertAction || !token) return;

        // FIXED (Error 2): Map alertAction ('approve'/'reject') to the required status type ('approved'/'rejected')
        const newStatus = alertAction === 'approve' ? 'approved' : 'rejected';

        // Store original status for potential rollback (optional)
        const originalStatus = requests.find(r => r.id === activeRequestId)?.status;

        setActionLoading(prev => ({ ...prev, [activeRequestId]: alertAction }));
        setAlertOpen(false); // Close confirmation dialog immediately

        try {
            // Call updateRequest with the mapped status
            const updatedRequest = await updateRequest(token, activeRequestId, { status: newStatus });

            // Update the request in the local state with the response from the API
            setRequests(prev =>
                prev.map((req) => (req.id === activeRequestId ? { ...req, status: updatedRequest.status, updatedAt: updatedRequest.updatedAt } : req))
            );

            // FIXED (Error 1 & 3): Use alertAction for comparison
            toast({
                title: alertAction === "approve" ? "Yêu cầu đã được duyệt" : "Yêu cầu đã bị từ chối",
                variant: "success",
            });

        } catch (err: any) {
            console.error(`Failed to ${alertAction} request ${activeRequestId}:`, err);
            toast({
                title: "Lỗi cập nhật",
                description: err.message || `Không thể ${alertAction === 'approve' ? 'duyệt' : 'từ chối'} yêu cầu.`,
                variant: "destructive",
            });
            // Optional: Revert state if API call fails
            // if (originalStatus) {
            //     setRequests(prev =>
            //         prev.map((req) => (req.id === activeRequestId ? { ...req, status: originalStatus } : req))
            //     );
            // }
        } finally {
            // Reset loading state and active request/action
            setActionLoading(prev => ({ ...prev, [activeRequestId]: null }));
            setActiveRequestId(null);
            setAlertAction(null);
        }
    };

    // Opens the confirmation dialog for approving or rejecting
    const openConfirmationAlert = (id: string, action: 'approve' | 'reject') => {
        setActiveRequestId(id);
        setAlertAction(action); // Set the action type ('approve' or 'reject')
        setAlertOpen(true);
    };

    // Handles opening the file preview dialog and fetching file data
    const handleFilePreview = async (request: RequestWithSender, fileId: string) => {
        // Ensure token and fileId are present
        if (!token || !fileId) return;

        // Reset state for the new preview
        setPreviewingFileInfo(null);
        setSelectedFileContent([]);
        setFileHeaders([]);
        setFileContentLoading(true);
        setFileDialogOpen(true);
        setImportSuccess(false); // Reset import success state for the new file
        setActiveRequestId(request.id); // Set the active request ID for context

        try {
            // 1. Get FileInfo (including the URL) using the fileId
            const fileInfos = await getMultipleFilesByFileIds(token, { ids: [fileId] });
            if (!fileInfos || fileInfos.length === 0) {
                throw new Error(`Không tìm thấy thông tin cho file ID: ${fileId}`);
            }
            const fileInfo = fileInfos[0];
            setPreviewingFileInfo(fileInfo); // Store the full FileInfo

            // 2. Fetch the actual file content from the URL
            // IMPORTANT: Ensure the fileInfo.url is accessible (CORS headers might be needed on the server)
            const response = await axios.get(fileInfo.url, {
              responseType: 'arraybuffer' // <--- Crucial for getting ArrayBuffer
          });
            
          const arrayBuffer: ArrayBuffer = response.data;

            // 3. Parse the file content (Excel/CSV) using xlsx library
            const workbook = XLSX.read(arrayBuffer, { type: 'array' });
            const firstSheetName = workbook.SheetNames[0];
            if (!firstSheetName) throw new Error("File không có sheet nào.");
            const worksheet = workbook.Sheets[firstSheetName];
            // Use header: 1 to get an array of arrays, making header extraction easier
            const data: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" });

            if (!data || data.length < 1) { // Check if data array is valid and has at least one row (header)
                 setSelectedFileContent([]);
                 setFileHeaders([]);
                 console.warn("File is empty or header row could not be read.");
                 // Optionally show a message to the user in the dialog
            } else {
                 const headers = data[0].map(String); // Extract first row as headers
                 // Use sheet_to_json normally to get array of objects for the content display
                 const content = XLSX.utils.sheet_to_json(worksheet, { defval: "" });
                 setFileHeaders(headers);
                 setSelectedFileContent(content);
            }

        } catch (error: any) {
            console.error("Lỗi khi xem trước file:", error);
            toast({
                title: "Lỗi xem trước file",
                description: error.message || "Không thể đọc hoặc tải dữ liệu từ file.",
                variant: "destructive",
            });
            setSelectedFileContent([]); // Clear content on error
            setFileHeaders([]);
            // Optionally close the dialog on critical errors: setFileDialogOpen(false);
        } finally {
            setFileContentLoading(false);
        }
    };

    // Handles the action of importing the previewed file data to the database
    const handleAddToDatabase = async () => {
        // Ensure necessary info is available
        if (!previewingFileInfo || !previewingFileInfo.url || !token || !activeRequestId) {
            toast({ title: "Lỗi", description: "Thiếu thông tin file hoặc yêu cầu để thực hiện.", variant: "destructive" });
            return;
        }

        // Set loading state specifically for the import action
        setActionLoading(prev => ({ ...prev, [activeRequestId]: 'import' }));
        setAddToDatabaseAlertOpen(false); // Close confirmation dialog

        try {
            // Call the import API function with the file URL
            // Assuming the API expects the URL directly in the body based on your function definition
            const status = await importFileToDataBase(token, previewingFileInfo.url);

            // Check for successful HTTP status code (e.g., 2xx)
            if (status >= 200 && status < 300) {
                setImportSuccess(true); // Indicate success for UI feedback
                toast({
                    title: "Thêm dữ liệu thành công",
                    description: `Dữ liệu từ file "${previewingFileInfo.name}" đã được thêm vào hệ thống.`,
                    variant: "success",
                });
                // Optionally: Close the preview dialog after successful import
                // setFileDialogOpen(false);
            } else {
                 // Handle non-2xx status codes as errors
                 throw new Error(`API trả về trạng thái không thành công: ${status}`);
            }

        } catch (err: any) {
            console.error("Failed to import file to database:", err);
            setImportSuccess(false); // Ensure success state is false on error
            toast({
                title: "Lỗi thêm dữ liệu",
                description: err.message || "Không thể thêm dữ liệu từ file vào hệ thống.",
                variant: "destructive",
            });
        } finally {
            // Reset loading state for the import action
            setActionLoading(prev => ({ ...prev, [activeRequestId]: null }));
        }
    };

    // --- JSX Rendering ---
    return (
        <div className="container mx-auto py-6 px-4 md:px-6">
            <div className="flex items-center justify-between mb-6">
                <h1 className="text-2xl md:text-3xl font-bold">Quản lý yêu cầu</h1>
                {/* Placeholder for potential filters or search bar */}
            </div>

            {/* Loading State */}
            {loading && (
                <div className="flex flex-col items-center justify-center min-h-[50vh] text-muted-foreground">
                    <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                    <p>Đang tải danh sách yêu cầu...</p>
                </div>
            )}

            {/* Error State */}
            {!loading && error && (
                 <Card className="bg-destructive/10 border-destructive">
                     <CardHeader>
                         <CardTitle className="flex items-center gap-2 text-destructive">
                             <AlertCircle className="h-5 w-5"/> Lỗi Tải Dữ Liệu
                         </CardTitle>
                     </CardHeader>
                     <CardContent>
                         <p>{error}</p>
                         {/* Provide a way to retry */}
                         <Button variant="secondary" size="sm" className="mt-4" onClick={() => window.location.reload()}>Thử lại</Button>
                     </CardContent>
                 </Card>
            )}

            {/* Empty State */}
            {!loading && !error && requests.length === 0 && (
                <div className="text-center text-muted-foreground min-h-[40vh] flex flex-col justify-center items-center">
                    <FileText className="h-12 w-12 mb-4 text-gray-400" />
                    <p>Không có yêu cầu nào được gửi đến bạn.</p>
                </div>
            )}

            {/* Request List */}
            {!loading && !error && requests.length > 0 && (
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                    {/* FIXED (Error 5): Access properties based on 'Request' type */}
                    {requests.map((request) => (
                        <Card key={request.id} className={`border-l-4 ${request.status === 'pending' ? 'border-l-yellow-500' : request.status === 'approved' ? 'border-l-green-500' : 'border-l-red-500'}`}>
                            <CardHeader>
                                <div className="flex justify-between items-start gap-2">
                                    <div className="flex-1 min-w-0">
                                        {/* Use request.title */}
                                        <CardTitle className="text-lg truncate" title={request.title}>{request.title}</CardTitle>
                                        <CardDescription className="text-xs mt-1">Mã: {request.id}</CardDescription>
                                        {/* Display Sender Info */}
                                        <div className="text-xs text-muted-foreground mt-2 flex items-center gap-1.5" title={`Gửi bởi: ${request.senderDetails?.username || request.senderId}`}>
                                             <UserCircle className="h-3.5 w-3.5 flex-shrink-0"/>
                                             <span className="truncate">
                                                 {request.senderDetails ? `${request.senderDetails.firstName || ''} ${request.senderDetails.lastName || ''}`.trim() || request.senderDetails.username : `ID: ${request.senderId}`}
                                             </span>
                                        </div>
                                    </div>
                                    <div className="flex-shrink-0">
                                         {getStatusBadge(request.status)}
                                    </div>
                                </div>
                            </CardHeader>

                            <CardContent className="space-y-3 text-sm">
                                {/* Use request.createdAt and request.updatedAt */}
                                <p><strong className="font-medium">Ngày gửi:</strong> {formatDate(request.createdAt)}</p>
                                {request.updatedAt !== request.createdAt && (
                                    <p><strong className="font-medium">Cập nhật:</strong> {formatDate(request.updatedAt)}</p>
                                )}
                                {/* Use request.description */}
                                {request.description && (
                                    <p className="text-muted-foreground bg-gray-50 p-2 rounded border border-dashed">
                                        <strong className="font-medium text-foreground block mb-1">Nội dung:</strong>
                                        {request.description}
                                    </p>
                                )}

                                {/* Use request.fileIds */}
                                {request.fileIds && request.fileIds.length > 0 && (
                                    <div>
                                        <h4 className="font-medium mb-2">File đính kèm ({request.fileIds.length}):</h4>
                                        <div className="flex flex-wrap gap-2">
                                            {request.fileIds.map((fileId) => (
                                                <Button
                                                    key={fileId}
                                                    variant="outline"
                                                    size="sm"
                                                    className="flex items-center gap-1.5 text-xs h-8"
                                                    onClick={() => handleFilePreview(request, fileId)}
                                                    // Disable preview if importing this request's file
                                                    disabled={actionLoading[request.id] === 'import'}
                                                >
                                                    <Eye className="h-3.5 w-3.5" />
                                                    Xem File
                                                    {/* Displaying File ID is okay if name isn't readily available */}
                                                </Button>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </CardContent>

                            {/* Action Buttons for Pending Requests */}
                            {request.status === "pending" && (
                                <CardFooter className="flex gap-3 pt-4 border-t">
                                    <Button
                                        variant="default"
                                        size="sm"
                                        className="bg-green-600 hover:bg-green-700 text-white flex-1"
                                        // Pass 'approve' action type
                                        onClick={() => openConfirmationAlert(request.id, 'approve')}
                                        // Disable if any action is loading for this request
                                        disabled={!!actionLoading[request.id]}
                                    >
                                        {actionLoading[request.id] === 'approve' ? <Loader2 className="w-4 h-4 mr-1.5 animate-spin" /> : <Check className="w-4 h-4 mr-1.5"/>}
                                        Duyệt
                                    </Button>
                                    <Button
                                        variant="destructive"
                                        size="sm"
                                        className="flex-1"
                                        // Pass 'reject' action type
                                        onClick={() => openConfirmationAlert(request.id, 'reject')}
                                        disabled={!!actionLoading[request.id]}
                                    >
                                         {actionLoading[request.id] === 'reject' ? <Loader2 className="w-4 h-4 mr-1.5 animate-spin" /> : <X className="w-4 h-4 mr-1.5"/>}
                                        Từ chối
                                    </Button>
                                </CardFooter>
                            )}
                        </Card>
                    ))}
                </div>
            )}

            {/* --- Dialogs --- */}

            {/* File Preview Dialog */}
            <Dialog open={fileDialogOpen} onOpenChange={setFileDialogOpen}>
                {/* Content remains largely the same, ensure it uses fileHeaders and selectedFileContent */}
                <DialogContent className="max-w-none w-[95vw] h-[90vh] p-4 flex flex-col">
                    <DialogHeader className="flex-shrink-0 pb-2 border-b">
                        <DialogTitle>Xem trước file: {previewingFileInfo?.name || 'Đang tải...'}</DialogTitle>
                        {previewingFileInfo && (
                             <DialogDescription className="text-xs text-muted-foreground">
                                 ID: {previewingFileInfo.id} | Loại: {previewingFileInfo.type} | Kích thước: {(Number(previewingFileInfo.size) / 1024).toFixed(1)} KB
                             </DialogDescription>
                        )}
                    </DialogHeader>

                    <div className="flex-grow overflow-auto mt-4 border rounded-md relative bg-white">
                        {fileContentLoading && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 z-20">
                                <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                                <p>Đang tải dữ liệu file...</p>
                            </div>
                        )}
                        {!fileContentLoading && selectedFileContent.length > 0 && fileHeaders.length > 0 ? (
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
                                            {fileHeaders.map((header) => ( // Iterate through headers to maintain order
                                                <td key={`${idx}-${header}`} className="px-3 py-1.5 border-x border-slate-200 whitespace-nowrap text-slate-600">
                                                    {/* Ensure value is stringified */}
                                                    {String(row[header] ?? '')}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        ) : (
                             !fileContentLoading && (
                                 <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-6">
                                     <AlertCircle className="h-8 w-8 mx-auto mb-2 text-orange-500" />
                                     <p className="text-center">Không có dữ liệu để hiển thị hoặc không thể đọc file.</p>
                                 </div>
                             )
                         )}
                    </div>

                    <DialogFooter className="flex-shrink-0 border-t pt-4 mt-4 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:justify-end sm:space-x-2">
                        <Button
                            variant="default"
                            className="w-full sm:w-auto"
                            onClick={() => setAddToDatabaseAlertOpen(true)}
                            // Disable if loading, importing, no content, or already imported
                            disabled={fileContentLoading || !!actionLoading[activeRequestId ?? ''] || selectedFileContent.length === 0 || importSuccess}
                        >
                            {actionLoading[activeRequestId ?? ''] === 'import' ? (
                                <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Đang xử lý...</>
                            ) : importSuccess ? (
                                <><Check className="h-4 w-4 mr-2 text-green-500" /> Đã thêm</>
                            ) : (
                                <><Database className="h-4 w-4 mr-2" /> Thêm vào kho dữ liệu</>
                            )}
                        </Button>
                        <DialogClose asChild>
                            <Button variant="outline" className="w-full sm:w-auto">Đóng</Button>
                        </DialogClose>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Add to Database Alert Dialog */}
            <AlertDialog open={addToDatabaseAlertOpen} onOpenChange={setAddToDatabaseAlertOpen}>
                 <AlertDialogContent>
                     <AlertDialogHeader>
                         <AlertDialogTitle>Xác nhận thêm vào kho dữ liệu</AlertDialogTitle>
                         <AlertDialogDescription>
                             Bạn có chắc chắn muốn thêm dữ liệu từ file &quot;{previewingFileInfo?.name}&quot; vào kho dữ liệu không?
                             <p className="mt-2 text-amber-600 font-medium">Hành động này sẽ xử lý và lưu trữ dữ liệu.</p>
                         </AlertDialogDescription>
                     </AlertDialogHeader>
                     <AlertDialogFooter>
                         <AlertDialogCancel>Hủy</AlertDialogCancel>
                         <AlertDialogAction onClick={handleAddToDatabase} className="bg-primary hover:bg-primary/90">
                             Xác nhận thêm
                         </AlertDialogAction>
                     </AlertDialogFooter>
                 </AlertDialogContent>
             </AlertDialog>

            {/* Approve/Reject Confirmation Alert Dialog */}
            <AlertDialog open={alertOpen} onOpenChange={setAlertOpen}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>
                            {/* Use alertAction state to determine title */}
                            {alertAction === 'approve' ? "Xác nhận duyệt yêu cầu" : "Xác nhận từ chối yêu cầu"}
                        </AlertDialogTitle>
                        <AlertDialogDescription>
                             {/* Use alertAction state to determine description */}
                            Bạn có chắc chắn muốn {alertAction === 'approve' ? "duyệt" : "từ chối"} yêu cầu này không?
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel onClick={() => { setActiveRequestId(null); setAlertAction(null); }}>Hủy</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleUpdateRequestStatus}
                            // Style action button based on alertAction
                            className={alertAction === 'approve' ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'}
                        >
                            {/* Use alertAction state to determine button text */}
                            {alertAction === 'approve' ? "Duyệt" : "Từ chối"}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
}
