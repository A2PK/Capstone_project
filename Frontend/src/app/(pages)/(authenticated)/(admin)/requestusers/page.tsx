"use client"
import { useState, useEffect,useRef  } from "react"
import { Request } from "@/types/request"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { AlertCircle, FileText, Loader2, Eye, X, Database, Check,Upload  } from "lucide-react"
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
} from "@/components/ui/alert-dialog"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog"
import * as XLSX from "xlsx"
import { useToast } from "@/hooks/use-toast";
const MOCK_REQUESTS: Request[] = [
  {
    id: "req-001",
    subject: "Gửi data mới tháng 5",
    message: "Tôi gửi data nhé",
    status: "pending",
    date: "2025-03-15",
    files: ["dataset_DACN.xlsx"],
  },
  {
    id: "req-002",
    subject: "Gửi data mới tháng 4",
    message: "Tôi gửi data nhé",
    status: "approved",
    date: "2025-03-10",
    files: ["dataset_DACN.xlsx"],
  },
]

export default function AdminRequestsPage() {
  const uploadInputRef = useRef<HTMLInputElement | null>(null)
  const { toast } = useToast();
  const [requests, setRequests] = useState<Request[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedFileContent, setSelectedFileContent] = useState<any[]>([])
  const [fileContentLoading, setFileContentLoading] = useState(false)
  const [previewingFile, setPreviewingFile] = useState<string | null>(null)
  const [fileDialogOpen, setFileDialogOpen] = useState(false)
  const [activeRequestId, setActiveRequestId] = useState<string | null>(null)
  const [isApproving, setIsApproving] = useState(false)
  const [isRejecting, setIsRejecting] = useState(false)
  const [isAddingToDatabase, setIsAddingToDatabase] = useState(false)
  const [alertOpen, setAlertOpen] = useState(false)
  const [addToDatabaseAlertOpen, setAddToDatabaseAlertOpen] = useState(false)
  const [addSuccess, setAddSuccess] = useState(false)

  useEffect(() => {
    setTimeout(() => {
      setRequests(MOCK_REQUESTS)
      setLoading(false)
    }, 1000)
  }, [])

  const getStatusBadge = (status: Request["status"]) => {
    switch (status) {
      case "pending":
        return (
          <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">
            Đang đợi
          </Badge>
        )
      case "approved":
        return (
          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
            Đã duyệt
          </Badge>
        )
      case "rejected":
        return (
          <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
            Bị từ chối
          </Badge>
        )
      default:
        return <Badge variant="outline">Không xác định</Badge>
    }
  }

  const updateRequestStatus = (status: "approved" | "rejected") => {
    if (activeRequestId) {
      setRequests(prev =>
        prev.map((req) => (req.id === activeRequestId ? { ...req, status } : req))
      )
      setActiveRequestId(null)
      setAlertOpen(false)
      toast({
        title: status === "approved" ? "Yêu cầu đã được duyệt" : "Yêu cầu đã bị từ chối",
        description: status === "approved" ? "Yêu cầu đã được duyệt thành công" : "Yêu cầu đã bị từ chối",
        variant: status === "approved" ? "success" : "destructive",
      })
    }
  }

  const openApproveAlert = (id: string) => {
    setActiveRequestId(id)
    setIsApproving(true)
    setIsRejecting(false)
    setAlertOpen(true)
  }

  const openRejectAlert = (id: string) => {
    setActiveRequestId(id)
    setIsApproving(false)
    setIsRejecting(true)
    setAlertOpen(true)
  }

  const handleAddToDatabase = () => {
    setIsAddingToDatabase(true)

    setTimeout(() => {
      // Hiển thị success indicator tạm thời
      setAddSuccess(true)

      // Sau 1 khoảng thời gian ngắn, đóng popup và reset state
      setTimeout(() => {
        // Reset state, nhưng GIỮ dialog preview mở
        setIsAddingToDatabase(false)
        setAddSuccess(true) // Show success indicator
        
        setAddToDatabaseAlertOpen(false); // đóng confirm alert
        toast({
          title: "Thêm dữ liệu thành công",
          description: "Dữ liệu đã được thêm vào hệ thống",
          variant: "success",
        })
      }, 400)
    }, 100)
  }

  const handleFilePreview = async (fileName: string) => {
    setPreviewingFile(fileName)
    setFileDialogOpen(true)
    setFileContentLoading(true)
    setSelectedFileContent([])
    setAddSuccess(false)
    try {
      const response = await fetch('/file_data.xlsx')
      const arrayBuffer = await response.arrayBuffer()
      const workbook = XLSX.read(arrayBuffer, { type: 'array' })

      const firstSheetName = workbook.SheetNames[0]
      const worksheet = workbook.Sheets[firstSheetName]

      // Update to get all columns
      const data = XLSX.utils.sheet_to_json(worksheet, { defval: "" })
      setSelectedFileContent(data)
    } catch (error) {
      console.error("Lỗi khi đọc file Excel:", error)
      toast({
        title: "Lỗi đọc file",
        description: "Không thể đọc dữ liệu từ file Excel",
        variant: "destructive",
      });

      const sampleData = [
        { "Name": "John Doe", "Age": 28, "City": "New York" },
        { "Name": "Jane Smith", "Age": 34, "City": "Los Angeles" },
        { "Name": "David Johnson", "Age": 45, "City": "Chicago" },
        { "Name": "Sarah Williams", "Age": 32, "City": "Boston" },
        { "Name": "Michael Brown", "Age": 41, "City": "Seattle" }
      ]
      setSelectedFileContent(sampleData)
    } finally {
      setFileContentLoading(false)
    }

  }

  const handleTemplateUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const arrayBuffer = e.target?.result
        if (arrayBuffer) {
          const workbook = XLSX.read(arrayBuffer, { type: "array" })
          const firstSheet = workbook.SheetNames[0]
          const worksheet = workbook.Sheets[firstSheet]
          const data = XLSX.utils.sheet_to_json(worksheet, { defval: "" })
  
          toast({
            title: "Tải file mẫu thành công",
            description: `Đã tải lên ${file.name} với ${data.length} dòng`,
            variant: "success"
          })
  
          console.log("Dữ liệu mẫu:", data)
        }
      }
      reader.readAsArrayBuffer(file)
    }
  }
  return (
    <div className="container mx-auto py-1">
      <div className="flex items-center justify-between mb-6">
  <h1 className="text-3xl font-bold">Quản lý yêu cầu người dùng</h1>
  
  <div className="flex items-center gap-2">
    <input
      ref={uploadInputRef}
      type="file"
      accept=".xlsx,.csv"
      className="hidden"
      onChange={handleTemplateUpload}
    />
    <Button
      variant="outline"
      onClick={() => uploadInputRef.current?.click()}
    >
      <Upload className="w-4 h-4 mr-2" />
      Upload file mẫu
    </Button>
  </div>
</div>

      {loading ? (
        <div className="flex flex-col items-center justify-center min-h-[50vh]">
          <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
          <p>Đang tải các yêu cầu...</p>
        </div>
      ) : (
        <div className="grid gap-6">
          {requests.map((request) => (
            <Card key={request.id} className="border-l-4 border-l-primary">
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle>{request.subject}</CardTitle>
                    <p className="text-sm mt-1 text-muted-foreground">Mã: {request.id}</p>
                  </div>
                  {getStatusBadge(request.status)}
                </div>
              </CardHeader>

              <CardContent className="space-y-4">
                <p><strong>Ngày:</strong> {request.date}</p>
                <p><strong>Mô tả:</strong> {request.message}</p>

                {request.files.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">File đính kèm:</h4>
                    <div className="flex flex-wrap gap-3">
                      {request.files.map((file, index) => (
                        <div key={index} className="flex items-center gap-2 bg-muted p-2 rounded-md">
                          <FileText className="h-4 w-4 text-primary" />
                          <span>{file}</span>
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => handleFilePreview(file)}
                          >
                            <Eye className="w-4 h-4 mr-1" /> Xem
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {request.status === "pending" && (
                  <div className="flex gap-4 pt-4 border-t">
                    <Button
                      variant="secondary"
                      onClick={() => openApproveAlert(request.id)}
                      disabled={isApproving}
                    >
                      {isApproving ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : null} Duyệt
                    </Button>
                    <Button variant="destructive" onClick={() => openRejectAlert(request.id)}>Từ chối</Button>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* File Preview Dialog */}
      <Dialog open={fileDialogOpen} onOpenChange={(open) => {
        // Logic kiểm tra state giữ nguyên
         if (!isAddingToDatabase && !addToDatabaseAlertOpen) {
           setFileDialogOpen(open);
         }
      }}>
        {/* --- Thay đổi DialogContent --- */}
        <DialogContent className="w-screen h-screen max-w-full max-h-full p-4 flex flex-col"> {/* Tăng padding và dùng flex column */}
          <DialogHeader className="flex-shrink-0 pb-2 border-b"> {/* Thêm border dưới header */}
            <DialogTitle>Xem trước file: {previewingFile}</DialogTitle>
             {/* Có thể thêm DialogDescription nếu muốn */}
          </DialogHeader>

          {/* --- Container cho bảng với overflow --- */}
          <div className="flex-grow overflow-auto mt-4 relative border rounded-md"> {/* Thêm border ở đây */}
            {fileContentLoading ? (
              <div className="flex flex-col items-center justify-center h-full"> {/* Căn giữa loading */}
                <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                <p>Đang tải dữ liệu file...</p>
              </div>
            ) : selectedFileContent.length > 0 ? (
              // --- Bảng hiển thị dữ liệu ---
              // `table-fixed` có thể hữu ích nếu bạn muốn set width cột cụ thể,
              // nhưng `whitespace-nowrap` thường đủ cho việc giãn cột tự động.
              // `border-collapse` cho đường kẻ đẹp hơn.
              <table className="min-w-full text-sm border-collapse">
                <thead className="bg-slate-100 sticky top-0 z-10"> {/* Đổi màu nền thead, thêm z-index */}
                  <tr>
                    {Object.keys(selectedFileContent[0]).map((key) => (
                      <th
                        key={key}
                        // --- THÊM whitespace-nowrap VÀ STYLING CHO TH ---
                        className="px-4 py-2 border border-slate-300 font-medium text-slate-700 text-left whitespace-nowrap bg-slate-100"
                      >
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {selectedFileContent.map((row, idx) => (
                    // --- Thêm hiệu ứng hover và zebra striping cho hàng ---
                    <tr key={idx} className="hover:bg-slate-50 even:bg-white odd:bg-slate-50/50">
                      {Object.keys(row).map((key) => (
                        <td
                          key={key}
                           // --- THÊM whitespace-nowrap VÀO TD ---
                          className="px-4 py-2 border border-slate-200 whitespace-nowrap text-slate-600"
                        >
                          {/* Đảm bảo giá trị là string để hiển thị */}
                          {String(row[key])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
               // --- Phần báo lỗi giữ nguyên ---
              <div className="flex flex-col items-center justify-center h-full text-red-600"> {/* Căn giữa lỗi */}
                <AlertCircle className="h-8 w-8 mx-auto mb-2" />
                <p>Không thể đọc hoặc không có dữ liệu từ file.</p>
              </div>
            )}
             {/* Success indicator có thể đặt lại ở đây nếu cần */}
          </div>

          {/* --- DialogFooter giữ nguyên cấu trúc nhưng điều chỉnh styling --- */}
          <DialogFooter className="flex-shrink-0 border-t border-slate-200 pt-4 mt-4 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:justify-end sm:space-x-2">
            <Button
              variant="default" // Có thể dùng màu primary của bạn
              className="w-full sm:w-auto" // Bỏ màu cứng, dùng variant
              onClick={() => setAddToDatabaseAlertOpen(true)}
              disabled={fileContentLoading || isAddingToDatabase || selectedFileContent.length === 0 || addSuccess}
            >
              {isAddingToDatabase ? (
                <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Đang xử lý...</>
              ) : addSuccess ? ( // Thêm trạng thái sau khi thành công
                 <><Check className="h-4 w-4 mr-2" /> Đã thêm</>
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
            <AlertDialogTitle>
              Xác nhận thêm vào kho dữ liệu
            </AlertDialogTitle>
            <AlertDialogDescription>
              Bạn có chắc chắn muốn thêm dữ liệu từ file "{previewingFile}" vào kho dữ liệu không?
              <p className="mt-2 text-amber-600">Hành động này không thể hoàn tác.</p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setAddToDatabaseAlertOpen(false)}>
  Hủy
</AlertDialogCancel>
            <AlertDialogAction onClick={handleAddToDatabase}>
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
              {isApproving ? "Xác nhận duyệt" : "Xác nhận từ chối"}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {isApproving
                ? "Bạn có chắc chắn muốn duyệt yêu cầu này không?"
                : "Bạn có chắc chắn muốn từ chối yêu cầu này không?"
              }
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => {
              setActiveRequestId(null);
              setAlertOpen(false);
            }}>
              Hủy
            </AlertDialogCancel>
            <AlertDialogAction onClick={() => updateRequestStatus(isApproving ? "approved" : "rejected")}>
              {isApproving ? "Duyệt" : "Từ chối"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}