"use client"
import { useState, useEffect } from "react"
import { Request } from "@/types/request"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertCircle, FileText, Loader2, Eye, X, Database, Check } from "lucide-react"
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

  return (
    <div className="container mx-auto py-1">
      <h1 className="text-3xl font-bold mb-6">Quản lý yêu cầu người dùng</h1>

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
  if (!isAddingToDatabase && !addToDatabaseAlertOpen) {
    setFileDialogOpen(open);
  }
}}>
        <DialogContent className="w-screen h-screen max-w-full max-h-full p-1 gap-0">
          <DialogHeader className="flex-shrink-0">
            <DialogTitle>{previewingFile}</DialogTitle>
          </DialogHeader>

          <div className="flex-grow overflow-auto mt-4 relative">
            {fileContentLoading ? (
              <div className="flex flex-col items-center justify-center py-8">
                <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                <p>Đang tải dữ liệu file...</p>
              </div>
            ) : selectedFileContent.length > 0 ? (
              <div className="border rounded-md h-full">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-100 sticky top-0">
                    <tr>
                      {Object.keys(selectedFileContent[0]).map((key) => (
                        <th key={key} className="px-4 py-2 border">{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selectedFileContent.map((row, idx) => (
                      <tr key={idx}>
                        {Object.keys(row).map((key) => (
                          <td key={key} className="px-4 py-2 border">{row[key]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-red-500">
                <AlertCircle className="h-8 w-8 mx-auto mb-2" />
                <p>Không thể đọc dữ liệu từ file.</p>
              </div>
            )}

            {/* Success indicator */}
            {/* {addSuccess && (
              <div className="absolute top-4 right-4 bg-green-100 text-green-800 px-4 py-2 rounded-md flex items-center shadow-md animate-in fade-in duration-300">
                <Check className="h-5 w-5 mr-2" />
                <span>Đã thêm dữ liệu thành công!</span>
              </div>
            )} */}
          </div>

          <DialogFooter className="flex-shrink-0 border-t pt-4 mt-4 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
            <Button
              variant="default"
              className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700"
              onClick={() => setAddToDatabaseAlertOpen(true)}
              disabled={fileContentLoading || isAddingToDatabase || selectedFileContent.length === 0 || addSuccess}
            >
              {isAddingToDatabase ? (
                <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Đang xử lý...</>
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