// src/app/(pages)/(authenticated)/(officer)/dashboardofficer/request/new/page.tsx (Adjust path as needed)
"use client"
import * as XLSX from "xlsx"; // Library for reading Excel/CSV files
import React, { useState, useRef, useEffect, ChangeEvent, FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { FileUp, Loader2, X, FileCheck2, AlertCircle } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { User } from "@/types/user"; // Assuming User type is defined
import { FileInfo, postFileParam } from "@/types/file"; // Assuming FileInfo/postFileParam types are defined
import { CreateRequestData } from "@/lib/request"; // Assuming type defined in request lib
import { getAllUsers } from "@/lib/user"; // Assuming getAllUsers is here
import { createRequest } from "@/lib/request"; // Import your API functions
import { uploadFile } from "@/lib/file"; // Import the UPDATED uploadFile function

// --- Constants ---
const ALLOWED_EXTENSIONS = ["csv", "xlsx"];
const MAX_FILE_SIZE_MB = 5;
const MAX_FILES = 5; // Limit number of files
// Define the exact expected header row for validation
const EXPECTED_HEADER: string[] = ["Điểm Quan Trắc", "Tỉnh", "Huyện", "Tọa độ", "Ngày quan trắc", "Nhiệt độ", "pH", "DO", "Độ dẫn", "Độ kiềm", "N-NO2", "N-NH4", "P-PO4", "H2S", "TSS", "COD", "Aeromonas tổng số", "Edwardsiella ictaluri", "Aeromonas hydrophila", "Coliform", "WQI (tính theo Aeromonas)", "Chất lượng nước", "Chỉ tiêu vượt ngưỡng", "Khuyến cáo"];


// --- Interface for storing file state ---
// Removed base64Data property
interface FileToUpload {
    file: File; // The actual File object
    isUploading: boolean;
    uploadError: string | null;
    uploadedId: string | null; // Store ID after successful upload
}

export default function NewRequestPage() {
    const router = useRouter();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const { toast } = useToast();

    // --- Form State ---
    const [title, setTitle] = useState("");
    const [description, setDescription] = useState("");
    const [filesToUpload, setFilesToUpload] = useState<FileToUpload[]>([]); // State for files being processed
    const [receiverId, setReceiverId] = useState<string>(""); // State for selected Admin ID

    // --- API/User State ---
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null); // General form error
    const [showErrorDialog, setShowErrorDialog] = useState(false);
    const [adminUsers, setAdminUsers] = useState<User[]>([]);
    const [isLoadingAdmins, setIsLoadingAdmins] = useState(true);
    const [token, setToken] = useState<string | null>(null);
    const [currentUser, setCurrentUser] = useState<User | null>(null); // Store logged-in user

    // --- Fetch Session Info (Token and current User) ---
    useEffect(() => {
        const fetchSession = async () => {
            try {
                const response = await fetch('/api/auth/session');
                if (!response.ok) throw new Error('Failed to fetch session');
                const data: { isLoggedIn: boolean; user: User | null; token: string | null } = await response.json();
                if (data.isLoggedIn && data.user && data.token) {
                    setCurrentUser(data.user);
                    setToken(data.token);
                } else {
                    setError("Không thể lấy thông tin phiên đăng nhập. Vui lòng đăng nhập lại.");
                    setShowErrorDialog(true);
                    // Consider redirecting: router.push('/auth/login');
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : "Lỗi tải thông tin phiên.");
                setShowErrorDialog(true);
                console.error("Session fetch error:", err);
            }
        };
        fetchSession();
    }, [router]); // Added router dependency if used for redirect

    // --- Fetch Admin Users when token is available ---
    useEffect(() => {
        if (token) {
            const fetchAdmins = async () => {
                setIsLoadingAdmins(true);
                try {
                    // Assume getAllUsers now requires token
                    const userData = await getAllUsers(token, { limit: 500 }); // Fetch a reasonable limit
                    const admins = userData.users.filter(user => user.role === 'admin');
                    setAdminUsers(admins);
                } catch (err) {
                    console.error("Failed to fetch admin users:", err);
                    setError("Không thể tải danh sách quản trị viên.");
                    setShowErrorDialog(true);
                } finally {
                    setIsLoadingAdmins(false);
                }
            };
            fetchAdmins();
        }
    }, [token]);

    // --- File Handling ---

    /**
     * Reads the header row of an Excel or CSV file and compares it to the expected header.
     * @param file The File object to validate.
     * @returns Promise<boolean> True if the header matches, false otherwise.
     */
    const validateFileHeader = (file: File): Promise<boolean> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                try {
                    const data = event.target?.result;
                    if (!data) {
                        return reject(new Error("Không thể đọc nội dung tệp."));
                    }
                    // Parse the file data using xlsx library
                    const workbook = XLSX.read(data, { type: 'array' }); // Use 'array' for ArrayBuffer
                    const firstSheetName = workbook.SheetNames[0];
                    if (!firstSheetName) {
                        return reject(new Error("Tệp không có trang tính (sheet) nào."));
                    }
                    const worksheet = workbook.Sheets[firstSheetName];
                    // Extract the first row (header) using sheet_to_json with header: 1
                    // This returns an array of arrays, where the first inner array is the header.
                    const jsonData: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" }); // Use defval to handle empty cells

                    if (!jsonData || jsonData.length === 0 || !jsonData[0]) {
                         return reject(new Error("Không thể đọc hàng tiêu đề từ tệp hoặc tệp trống."));
                    }

                    const headerRow = jsonData[0].map(cell => String(cell).trim()); // Get the first row and trim whitespace

                    console.log("Extracted Header:", headerRow);
                    console.log("Expected Header:", EXPECTED_HEADER);

                    // Compare extracted header with the expected header
                    const headersMatch = headerRow.length === EXPECTED_HEADER.length &&
                                         headerRow.every((value, index) => value === EXPECTED_HEADER[index]);

                    resolve(headersMatch);

                } catch (parseError) {
                    console.error("Error parsing file header:", parseError);
                    reject(new Error("Đã xảy ra lỗi khi đọc định dạng tệp. Đảm bảo tệp không bị lỗi."));
                }
            };

            reader.onerror = (error) => {
                console.error("FileReader error:", error);
                reject(new Error("Không thể đọc tệp."));
            };

            // Read the file as an ArrayBuffer, suitable for XLSX.read
            reader.readAsArrayBuffer(file);
        });
    };


    const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
        const selectedFiles = e.target.files;
        if (!selectedFiles) return;

        const newFiles: File[] = Array.from(selectedFiles);
        const filesToAdd: FileToUpload[] = [];
        let validationErrorOccurred = false; // Flag to stop adding files if one fails validation

        if (filesToUpload.length + newFiles.length > MAX_FILES) {
            setError(`Chỉ được phép tải lên tối đa ${MAX_FILES} tệp.`);
            setShowErrorDialog(true);
            // Clear the input value
            if (fileInputRef.current) fileInputRef.current.value = "";
            return;
        }

        // Process each selected file
        for (const file of newFiles) {
            // 1. Basic Validation (Extension, Size)
            const extension = file.name.split('.').pop()?.toLowerCase();
            if (!extension || !ALLOWED_EXTENSIONS.includes(extension)) {
                setError(`Tệp "${file.name}" có định dạng không được hỗ trợ. Chỉ chấp nhận: ${ALLOWED_EXTENSIONS.join(", ")}`);
                setShowErrorDialog(true);
                validationErrorOccurred = true;
                break; // Stop processing further files in this batch
            }
            if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
                setError(`Tệp "${file.name}" (${(file.size / 1024 / 1024).toFixed(1)}MB) vượt quá dung lượng tối đa ${MAX_FILE_SIZE_MB}MB.`);
                setShowErrorDialog(true);
                validationErrorOccurred = true;
                break; // Stop processing further files
            }

            // 2. Header Validation (Asynchronous)
            try {
                // const headerIsValid = await validateFileHeader(file);
                // if (!headerIsValid) {
                //     setError(`Tệp "${file.name}" có cấu trúc cột không đúng theo file mẫu. Vui lòng kiểm tra lại.`);
                //     setShowErrorDialog(true);
                //     validationErrorOccurred = true;
                //     break; // Stop processing further files
                // }
                // If validation passes, prepare the file object for the state
                 filesToAdd.push({
                    file, // Store the actual File object
                    isUploading: false,
                    uploadError: null,
                    uploadedId: null,
                 });

            } catch (headerValidationError: any) {
                 // Catch errors from validateFileHeader (e.g., file reading errors)
                 setError(`Lỗi khi kiểm tra tệp "${file.name}": ${headerValidationError.message}`);
                 setShowErrorDialog(true);
                 validationErrorOccurred = true;
                 break; // Stop processing further files
            }
        } // End of loop through files

        // Only add files to state if no validation errors occurred during the loop
        if (!validationErrorOccurred) {
            setFilesToUpload(prev => [...prev, ...filesToAdd]);
        }

        // Clear the input value regardless of validation outcome
        // so the same file can be selected again if needed after fixing errors.
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const removeFile = (index: number) => {
        setFilesToUpload(prevFiles => prevFiles.filter((_, i) => i !== index));
    };

    // --- Form Submission ---
    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        // Basic form validation
        if (!token || !currentUser) {
            setError("Mất phiên đăng nhập. Vui lòng tải lại trang hoặc đăng nhập lại.");
            setShowErrorDialog(true);
            return;
        }
        if (!title.trim()) { setError("Tiêu đề không được để trống."); setShowErrorDialog(true); return; }
        if (!description.trim()) { setError("Nội dung không được để trống."); setShowErrorDialog(true); return; }
        if (!receiverId) { setError("Vui lòng chọn người nhận (Admin)."); setShowErrorDialog(true); return; }

        setIsSubmitting(true);
        setError(null);
        const uploadedFileIds: string[] = [];
        let uploadFailed = false;

        // 1. Upload Files Sequentially
        if (filesToUpload.length > 0) {
            // Update state to show uploading status for all pending files
            setFilesToUpload(prev => prev.map(f =>
                f.uploadedId ? f : { ...f, isUploading: true, uploadError: null } // Only mark non-uploaded files as uploading
            ));

            // Use a loop that respects the current state
            for (let i = 0; i < filesToUpload.length; i++) {
                 // Get the potentially updated file item from state inside the loop
                 // NOTE: It's generally safer to read from the *state* inside the loop if state updates can happen
                 // For this sequential upload, reading the initial array `filesToUpload[i]` might be okay,
                 // but using state ensures consistency if the logic becomes more complex.
                 // Let's get the item from the current state for robustness:
                 const currentFileItem = (filesToUploadState => filesToUploadState[i])(filesToUpload);


                 // Skip if already uploaded or already failed (based on potentially updated state)
                 if (currentFileItem.uploadedId || currentFileItem.uploadError) {
                     continue;
                 }

                try {
                    console.log(`Uploading file: ${currentFileItem.file.name}`);

                    // Pass the File object directly
                    const fileParam: postFileParam = {
                        file: currentFileItem.file, // Pass the actual File object
                        name: currentFileItem.file.name, // Use original filename or a custom one if needed
                        type: currentFileItem.file.type || 'text/csv', // Provide MIME type or a default
                    };

                    // Call the updated uploadFile function
                    const uploadedFileInfo = await uploadFile(token, fileParam);
                    uploadedFileIds.push(uploadedFileInfo.id);

                    // Update state for this specific file upon success
                    // Use functional update to ensure we're working with the latest state
                     setFilesToUpload(prev => prev.map((f, idx) =>
                        idx === i ? { ...f, isUploading: false, uploadedId: uploadedFileInfo.id } : f
                    ));
                    console.log(`Uploaded ${currentFileItem.file.name}, ID: ${uploadedFileInfo.id}`);

                } catch (uploadError: any) {
                    console.error(`Failed to upload ${currentFileItem.file.name}:`, uploadError);
                    uploadFailed = true;
                    // Update state for this specific file upon failure
                    // Use functional update
                     setFilesToUpload(prev => prev.map((f, idx) =>
                        idx === i ? { ...f, isUploading: false, uploadError: uploadError.message || "Lỗi không xác định" } : f
                    ));
                    // Stop the whole submission process if one file fails
                    break;
                }
            }
        }

        // 2. Create Request if all uploads succeeded (or no files to upload)
        if (!uploadFailed) {
            try {
                const requestData: CreateRequestData = {
                    title: title.trim(),
                    description: description.trim(),
                    receiverId: receiverId,
                    fileIds: uploadedFileIds, // Use the collected IDs
                    status: 'pending', // Or your desired initial status
                    senderId: currentUser.id,
                    date: new Date().toISOString(), // Add current date/time
                };

                console.log("Creating request with data:", requestData);
                const createdRequest = await createRequest(token, requestData);
                console.log("Request created successfully:", createdRequest);

                toast({
                    title: "Thành công!",
                    description: "Yêu cầu của bạn đã được gửi đi.",
                    variant: "success",
                });
                router.push("/dashboardofficer/request"); // Navigate on success

            } catch (requestError: any) {
                console.error("Lỗi tạo yêu cầu:", requestError);
                setError(requestError.message || "Gửi yêu cầu không thành công. Vui lòng thử lại.");
                setShowErrorDialog(true);
                // Consider how to handle already uploaded files if request creation fails (e.g., inform user, attempt cleanup)
            }
        } else {
            // If upload failed, set general error message
            setError("Một hoặc nhiều tệp tải lên thất bại. Yêu cầu chưa được gửi. Vui lòng kiểm tra lại các tệp bị lỗi.");
            setShowErrorDialog(true);
        }

        // Ensure submitting state is reset regardless of outcome
        setIsSubmitting(false);
    };


    // --- JSX Rendering ---
    return (
        <div className="container mx-auto py-10 max-w-4xl">
            <Card className="mx-auto shadow-lg">
                <form onSubmit={handleSubmit}>
                    <CardHeader>
                        <CardTitle className="text-2xl">Tạo Yêu Cầu Mới</CardTitle>
                        <CardDescription>Điền thông tin, đính kèm tệp (nếu có) và gửi yêu cầu đến quản trị viên.</CardDescription>
                    </CardHeader>

                    <CardContent className="space-y-6">
                        {/* Tiêu đề yêu cầu */}
                        <div className="space-y-2">
                            <Label htmlFor="title">Tiêu Đề <span className="text-red-500">*</span></Label>
                            <Input id="title" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Ví dụ: Gửi dữ liệu quan trắc tháng 5" required disabled={isSubmitting} />
                        </div>

                         {/* Admin Recipient Selection */}
                         <div className="space-y-2">
                             <Label htmlFor="receiverId">Gửi đến Admin <span className="text-red-500">*</span></Label>
                             <Select
                                 value={receiverId}
                                 onValueChange={(value) => setReceiverId(value)}
                                 required
                                 disabled={isSubmitting || isLoadingAdmins}
                             >
                                 <SelectTrigger id="receiverId" className="w-full" disabled={isLoadingAdmins}>
                                     <SelectValue placeholder={isLoadingAdmins ? "Đang tải danh sách..." : "Chọn quản trị viên"} />
                                 </SelectTrigger>
                                 <SelectContent>
                                     {!isLoadingAdmins && adminUsers.length === 0 && <p className="p-4 text-sm text-muted-foreground">Không tìm thấy quản trị viên.</p>}
                                     {adminUsers.map((admin) => (
                                         <SelectItem key={admin.id} value={admin.id}>
                                             {`${admin.firstName || ''} ${admin.lastName || ''}`.trim() || admin.username} ({admin.email})
                                         </SelectItem>
                                     ))}
                                 </SelectContent>
                             </Select>
                         </div>

                        {/* Nội dung yêu cầu */}
                        <div className="space-y-2">
                            <Label htmlFor="description">Nội Dung <span className="text-red-500">*</span></Label>
                            <Textarea id="description" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Nhập nội dung chi tiết của yêu cầu..." className="min-h-[120px]" required disabled={isSubmitting} />
                        </div>

                        {/* Tệp đính kèm */}
                        <div className="space-y-2">
                            <Label htmlFor="files">Tệp đính kèm (Tối đa {MAX_FILES} tệp, tối đa {MAX_FILE_SIZE_MB}MB/tệp, chỉ .csv, .xlsx)</Label>
                            <div className="flex items-center gap-3 flex-wrap"> {/* Use flex-wrap */}
                                <Input
                                    id="files"
                                    ref={fileInputRef}
                                    type="file"
                                    onChange={handleFileChange}
                                    accept=".csv,.xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv" // More specific MIME types
                                    className="hidden" // Hide the default input
                                    multiple // Allow multiple files selection
                                    disabled={isSubmitting || filesToUpload.length >= MAX_FILES}
                                />
                                {/* Custom Button to trigger file input */}
                                <Button
                                    type="button"
                                    variant="outline"
                                    onClick={() => fileInputRef.current?.click()} // Trigger the hidden input
                                    className="flex items-center gap-2"
                                    disabled={isSubmitting || filesToUpload.length >= MAX_FILES}
                                >
                                    <FileUp className="h-4 w-4" />
                                    Chọn Tệp
                                </Button>
                                {/* Link to download template file */}
                                <Button variant="link" asChild className="text-blue-600 underline p-0 h-auto">
                                     <a href="/file_test.xlsx" download="file_mau_quan_trac.xlsx"> Tải file mẫu (.xlsx) </a>
                                </Button>
                            </div>
                             <p className="text-xs text-muted-foreground mt-1">Đã chọn: {filesToUpload.length}/{MAX_FILES} tệp. Yêu cầu tệp phải đúng cấu trúc cột theo file mẫu.</p>

                            {/* Display Selected/Uploading/Uploaded Files */}
                            {filesToUpload.length > 0 && (
                                <div className="mt-4 space-y-3 border-t pt-4">
                                     <h4 className="text-sm font-medium text-muted-foreground">Tệp đã chọn:</h4>
                                    {filesToUpload.map((fileItem, index) => (
                                        <div key={index} className={`flex items-center justify-between p-2 rounded text-sm ${fileItem.uploadError ? 'bg-red-100' : 'bg-muted'}`}>
                                            <div className="flex items-center gap-2 overflow-hidden min-w-0"> {/* Ensure child truncation works */}
                                                 {/* Status Icon - FIXED: Removed invalid 'title' prop */}
                                                 {fileItem.isUploading ? <Loader2 className="h-4 w-4 animate-spin flex-shrink-0 text-blue-600" />
                                                     : fileItem.uploadError ? <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0" /> /* Removed title */
                                                     : fileItem.uploadedId ? <FileCheck2 className="h-4 w-4 text-green-600 flex-shrink-0" /> /* Removed title */
                                                     : <FileUp className="h-4 w-4 flex-shrink-0 text-gray-500" />} {/* Ready state - Removed title */}

                                                 {/* File Name (Truncated) - Apply title here for tooltip if desired */}
                                                 <span className="truncate font-medium" title={fileItem.file.name}>{fileItem.file.name}</span>
                                                 {/* File Size */}
                                                 <span className="text-muted-foreground text-xs flex-shrink-0">({(fileItem.file.size / 1024).toFixed(1)} KB)</span>
                                             </div>

                                             <div className="flex items-center gap-2 flex-shrink-0">
                                                 {/* Error Message (if applicable) - Apply title here for tooltip if desired */}
                                                 {fileItem.uploadError && <span className="text-xs text-red-600 font-medium" title={fileItem.uploadError}>{fileItem.uploadError}</span>}

                                                 {/* Remove Button */}
                                                 {/* Show remove button only if not currently uploading AND hasn't successfully uploaded yet */}
                                                 {!fileItem.isUploading && !fileItem.uploadedId && (
                                                     <Button
                                                         type="button"
                                                         variant="ghost"
                                                         size="icon" // Make it a small icon button
                                                         onClick={() => removeFile(index)}
                                                         className="h-6 w-6 p-0 text-muted-foreground hover:text-red-600"
                                                         disabled={isSubmitting} // Disable remove during final submission
                                                         title="Xóa tệp" // Title on the button itself is fine
                                                     >
                                                         <X className="h-4 w-4" />
                                                         <span className="sr-only">Xóa tệp</span>
                                                     </Button>
                                                 )}
                                             </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </CardContent>

                    <CardFooter className="flex justify-between border-t pt-6">
                        <Button type="button" variant="outline" onClick={() => router.back()} disabled={isSubmitting}>
                            Hủy
                        </Button>
                        <Button type="submit" disabled={isSubmitting || filesToUpload.some(f => f.isUploading)}> {/* Disable if any file is still uploading */}
                            {isSubmitting ? (
                                <> <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Đang gửi... </>
                            ) : (
                                "Gửi Yêu Cầu"
                            )}
                        </Button>
                    </CardFooter>
                </form>
            </Card>

             {/* Dialog Báo Lỗi Chung */}
             <AlertDialog open={showErrorDialog} onOpenChange={setShowErrorDialog}>
                 <AlertDialogContent>
                     <AlertDialogHeader>
                         <AlertDialogTitle className="flex items-center gap-2">
                             <AlertCircle className="h-5 w-5 text-red-500"/> Lỗi
                         </AlertDialogTitle>
                         <AlertDialogDescription className="pt-2">
                             {error || "Đã xảy ra lỗi không xác định."}
                         </AlertDialogDescription>
                     </AlertDialogHeader>
                     <AlertDialogFooter>
                         <AlertDialogAction onClick={() => setShowErrorDialog(false)}>OK</AlertDialogAction>
                     </AlertDialogFooter>
                 </AlertDialogContent>
             </AlertDialog>
        </div>
    );
}
