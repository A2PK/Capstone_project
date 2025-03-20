"use client"
import { Request } from "@/types/request"
import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, FileText, Loader2 } from "lucide-react"
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

// Mock data for demonstration
const MOCK_REQUESTS : Request[] = [
  {
    id: "req-001",
    subject: "Gửi data mới tháng 5",
    message: "Tôi gửi data nhé",
    status: "pending",
    date: "2025-03-15",
    files: ["thang5.xlsx"],
  },
  {
    id: "req-002",
    subject: "Gửi data mới tháng 4",
    message: "Tôi gửi data nhé",
    status: "approved",
    date: "2025-03-10",
    files: ["thg4.csv"],
  },
  {
    id: "req-003",
    subject: "Gửi data mới tháng 3 và tháng 2",
    message: "Tôi gửi data nhé",
    status: "pending",
    date: "2025-03-18",
    files: ["thg3.xlsx", "thg2.csv"],
  },
]

export default function RequestsPage() {
  const [requests, setRequests] = useState<Request[]>([])
  const [loading, setLoading] = useState(true)
  const [cancellingId, setCancellingId] = useState(null)

  useEffect(() => {
    // In a real app, this would fetch from the server
    // const fetchRequests = async () => {
    //   try {
    //     const data = await getRequests()
    //     setRequests(data)
    //   } catch (error) {
    //     console.error("Failed to fetch requests:", error)
    //   } finally {
    //     setLoading(false)
    //   }
    // }
    // fetchRequests()

    // Using mock data for demonstration
    setTimeout(() => {
      setRequests(MOCK_REQUESTS)
      setLoading(false)
    }, 1000)
  }, [])

  const handleCancelRequest = async (id: any) => {
    setCancellingId(id)
    try {
      // In a real app, this would call the server action
      // await cancelRequest(id)

      // For demonstration, we'll just update the local state
      setTimeout(() => {
        setRequests(requests.map((req) => (req.id === id ? { ...req, status: "cancelled" } : req)))
        setCancellingId(null)
      }, 1000)
    } catch (error) {
      console.error("Huỷ yêu cầu thất bại.", error)
      setCancellingId(null)
    }
  }

  const getStatusBadge = (status: any) => {
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
            Đã được duyệt
          </Badge>
        )
      case "rejected":
        return (
          <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
            Bị từ chối
          </Badge>
        )
      case "cancelled":
        return (
          <Badge variant="outline" className="bg-gray-50 text-gray-700 border-gray-200">
            Đã huỷ
          </Badge>
        )
      default:
        return <Badge variant="outline">Không xác định</Badge>
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto py-10 flex flex-col items-center justify-center min-h-[50vh]">
        <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
        <p>Đang tải các yêu cầu của bạn...</p>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-1">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Yêu cầu của tôi</h1>
        <Button asChild>
          <Link href="/dashboardofficer/request/new">Yêu cầu mới</Link>
        </Button>
      </div>

      {requests.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
          <h2 className="text-xl font-semibold mb-2">Không có yêu cầu được tìm thấy.</h2>
          <p className="text-muted-foreground mb-6">Bạn chưa tạo yêu cầu nào.</p>
          <Button asChild>
            <Link href="/request/new">Tạo yêu cầu đầu tiên của bạn</Link>
          </Button>
        </div>
      ) : (
        <div className="grid gap-6">
          {requests.map((request) => (
            <Card key={request.id} className="border-l-4 border-l-primary">
              <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle className="text-xl">{request.subject}</CardTitle>
                    <p className="text-sm text-muted-foreground mt-1">
                      Mã yêu cầu: <span className="font-mono">{request.id}</span>
                    </p>
                  </div>
                  {getStatusBadge(request.status)}
                </div>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <h4 className="text-sm font-semibold text-muted-foreground">Ngày yêu cầu:</h4>
                      <p>
                        {new Date(request.date).toLocaleDateString("en-US", {
                          year: "numeric",
                          month: "long",
                          day: "numeric",
                        })}
                      </p>
                    </div>
                    <div className="space-y-1">
                      <h4 className="text-sm font-semibold text-muted-foreground">Tình trạng:</h4>
                      <p className="capitalize"> {request.status}</p>
                    </div>
                  </div>

                  <div className="space-y-1 pt-2 border-t">
                    <h4 className="text-sm font-semibold text-muted-foreground">Mô tả yêu cầu:</h4>
                    <p className="text-base">{request.message}</p>
                  </div>

                  {request.files && request.files.length > 0 && (
                    <div className="space-y-1 pt-2 border-t">
                      <h4 className="text-sm font-semibold text-muted-foreground">File đính kèm:</h4>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {request.files.map((file, index) => (
                          <div key={index} className="flex items-center gap-2 text-sm bg-muted px-3 py-2 rounded-md">
                            <FileText className="h-4 w-4 text-primary" />
                            <span>{file}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {request.status === "pending" && (
                    <div className="pt-4 border-t mt-4">
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="destructive" size="sm">
                            Huỷ yêu cầu
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Huỷ yêu cầu</AlertDialogTitle>
                            <AlertDialogDescription>
                              Bạn có thật sự muốn huỷ yêu cầu này. Hành động huỷ sẽ không được hoàn tác.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Không huỷ</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => handleCancelRequest(request.id)}
                              disabled={cancellingId === request.id}
                            >
                              {cancellingId === request.id ? (
                                <>
                                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                  Đang huỷ...
                                </>
                              ) : (
                                "Đồng ý huỷ"
                              )}
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>

  )
}

