// src/app/(pages)/(authenticated)/(officer)/dashboardofficer/request/page.tsx (Adjust path as needed)
"use client"

import { Request } from "@/types/requestofficial"; // Use correct type
import { User } from "@/types/user"; // Import User type
import { useState, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, FileText, Loader2 } from "lucide-react";
import {
    AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast"; // Import toast
import { findRequestsByUserParticipant, deleteRequest } from "@/lib/request"; // Import your API functions
import { format, parseISO, isValid } from 'date-fns'; // Import date-fns
import { cn } from "@/lib/utils";
const getStatusTailwindClasses = (status: Request['status']): { border: string; } => {
  // Simplified version for border color used in Card className
  switch (status) {
      case "pending": return { border: "border-yellow-500" };
      case "approved": return { border: "border-green-500" };
      case "rejected": return { border: "border-red-500" };
      case "cancelled": return { border: "border-gray-500" };
      default: return { border: "border-gray-400" };
  }
};
// Helper function to format date
function formatRequestDate(dateString: string): string {
    try {
        const date = parseISO(dateString);
        return isValid(date) ? format(date, 'dd/MM/yyyy') : 'Ngày không hợp lệ'; // Vietnamese format
    } catch {
        return 'Ngày không hợp lệ';
    }
}

export default function RequestsPage() {
    const [requests, setRequests] = useState<Request[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [cancellingId, setCancellingId] = useState<string | null>(null); // Use string for ID

    // Auth state
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [isSessionLoading, setIsSessionLoading] = useState(true);

    const { toast } = useToast();

    // Fetch Session Info
    useEffect(() => {
        const fetchSession = async () => {
            setIsSessionLoading(true);
            try {
                const response = await fetch('/api/auth/session');
                if (!response.ok) throw new Error('Failed to fetch session');
                const data: { isLoggedIn: boolean; user: User | null; token: string | null } = await response.json();
                if (data.isLoggedIn && data.user && data.token) {
                    setUser(data.user);
                    setToken(data.token);
                } else {
                    setError("Vui lòng đăng nhập để xem yêu cầu."); // Set error if not logged in
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : "Lỗi tải thông tin phiên.");
                console.error(err);
            } finally {
                setIsSessionLoading(false);
            }
        };
        fetchSession();
    }, []);

    // Fetch Requests when user and token are available
    useEffect(() => {
        if (user && token) {
            const loadRequests = async () => {
                setLoading(true);
                setError(null); // Clear previous errors
                try {
                    // Assuming officer sees requests they sent (using senderId = user.id)
                    // If backend filters by sender from token, you might call getAllRequests
                    const fetchedRequests = await findRequestsByUserParticipant(token, user.id, { sortBy: 'created_at', sortDesc: true });
                    setRequests(fetchedRequests || []);
                } catch (err) {
                    setError(err instanceof Error ? err.message : "Không thể tải danh sách yêu cầu.");
                    console.error(err);
                    setRequests([]); // Clear requests on error
                } finally {
                    setLoading(false);
                }
            };
            loadRequests();
        }
    }, [user, token]); // Rerun if user or token changes

    // Handle Cancel Request API Call
    const handleCancelRequest = async (id: string) => {
        if (!token) {
            toast({ variant: "destructive", title: "Lỗi", description: "Mất phiên đăng nhập. Vui lòng tải lại trang." });
            return;
        }
        setCancellingId(id);
        try {
            // Call API to delete (cancel) the request
            // Assuming deleteRequest updates status or soft deletes if hardDelete=false
            const status = await deleteRequest(token, id); // Use soft delete if applicable

            if (status >= 200 && status < 300) {
                 // Update UI optimistically or refetch
                 setRequests(prev => prev.map((req) => (req.id === id ? { ...req, status: "cancelled" } : req)));
                 toast({ variant: "success", title: "Thành công", description: "Đã huỷ yêu cầu." });
            } else {
                 throw new Error(`Lỗi huỷ yêu cầu: ${status}`);
            }

        } catch (error) {
            console.error("Huỷ yêu cầu thất bại.", error);
            toast({ variant: "destructive", title: "Lỗi", description: error instanceof Error ? error.message : "Không thể huỷ yêu cầu." });
        } finally {
            setCancellingId(null);
        }
    };

    // Status Badge function (looks okay, ensure status values match interface)
    const getStatusBadge = (status: Request['status']) => {
        switch (status) {
            case "pending": return (<Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">Đang đợi</Badge>);
            case "approved": return (<Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">Đã duyệt</Badge>);
            case "rejected": return (<Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">Từ chối</Badge>);
            case "cancelled": return (<Badge variant="outline" className="bg-gray-50 text-gray-700 border-gray-200">Đã huỷ</Badge>);
            default: return <Badge variant="outline">Không xác định</Badge>;
        }
    };

    if (isSessionLoading || (loading && requests.length === 0 && !error)) { // Show loader while checking session or loading initial requests
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

    return (
        <div className="container mx-auto py-10"> {/* Increased padding */}
            <div className="flex justify-between items-center mb-8"> {/* Increased margin */}
                <h1 className="text-3xl font-bold">Yêu cầu của tôi</h1>
                <Button asChild>
                    <Link href="/dashboardofficer/request/new">Yêu cầu mới</Link>
                </Button>
            </div>

            {requests.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center border rounded-lg bg-card text-card-foreground shadow-sm">
                    <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
                    <h2 className="text-xl font-semibold mb-2">Không có yêu cầu nào</h2>
                    <p className="text-muted-foreground mb-6">Bạn chưa tạo yêu cầu nào.</p>
                    <Button asChild>
                        <Link href="/dashboardofficer/request/new">Tạo yêu cầu đầu tiên</Link>
                    </Button>
                </div>
            ) : (
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3"> {/* Use grid for responsiveness */}
                    {requests.map((request) => (
                        <Card key={request.id} className={cn("border-l-4 flex flex-col", `border-l-${getStatusTailwindClasses(request.status).border.split('-')[1]}-500`)}> {/* Dynamic border color */}
                            <CardHeader className="pb-3">
                                <div className="flex justify-between items-start gap-2">
                                    <CardTitle className="text-lg font-semibold leading-tight">{request.title}</CardTitle>
                                    {getStatusBadge(request.status)}
                                </div>
                                <p className="text-xs text-muted-foreground pt-1">
                                    Mã: <span className="font-mono">{request.id}</span> | Ngày: {formatRequestDate(request.date)}
                                </p>
                            </CardHeader>
                            <CardContent className="pt-2 flex-grow">
                                <div className="space-y-3">
                                    {/* Use request.description if available */}
                                    {request.description && (
                                        <div className="space-y-1">
                                            <h4 className="text-sm font-medium text-muted-foreground">Nội dung:</h4>
                                            <p className="text-sm text-foreground line-clamp-3">{request.description}</p>
                                        </div>
                                    )}
                                    {/* Display File IDs */}
                                    {request.fileIds && request.fileIds.length > 0 && (
                                        <div className="space-y-1 pt-2 border-t">
                                            <h4 className="text-sm font-medium text-muted-foreground">File IDs đính kèm:</h4>
                                            <div className="flex flex-wrap gap-1 mt-1">
                                                {request.fileIds.map((fileId, index) => (
                                                     <Badge key={index} variant="secondary" className="text-xs font-mono">{fileId}</Badge>
                                                    // You might want to fetch file names based on IDs here later
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                            {/* Show Cancel button only for pending requests */}
                            {request.status === "pending" && (
                                <CardFooter className="pt-3 border-t mt-auto">
                                    <AlertDialog>
                                        <AlertDialogTrigger asChild>
                                            <Button variant="destructive" size="sm" disabled={cancellingId === request.id}>
                                                {cancellingId === request.id ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                                Huỷ yêu cầu
                                            </Button>
                                        </AlertDialogTrigger>
                                        <AlertDialogContent>
                                            <AlertDialogHeader> <AlertDialogTitle>Huỷ yêu cầu?</AlertDialogTitle> <AlertDialogDescription> Bạn có chắc muốn huỷ yêu cầu &quot;{request.title}&quot;? Hành động này không thể hoàn tác. </AlertDialogDescription> </AlertDialogHeader>
                                            <AlertDialogFooter> <AlertDialogCancel>Không</AlertDialogCancel> <AlertDialogAction onClick={() => handleCancelRequest(request.id)} disabled={!!cancellingId}>Đồng ý huỷ</AlertDialogAction> </AlertDialogFooter>
                                        </AlertDialogContent>
                                    </AlertDialog>
                                </CardFooter>
                            )}
                        </Card>
                    ))}
                </div>
            )}
        </div>
    );
}