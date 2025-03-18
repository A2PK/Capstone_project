"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
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
import { ArrowLeft, RefreshCw } from "lucide-react"

export default function ModelManagement() {
    const [isRetraining, setIsRetraining] = useState(false)
    const [activeTab, setActiveTab] = useState("overview")

    const handleRetrain = () => {
        setIsRetraining(true)
        // Simulate retraining process
        setTimeout(() => {
            setIsRetraining(false)
        }, 30000)
    }

    return (
        <div className="flex min-h-screen flex-col">

            <main className="flex-1">
                <div className="container py-6">
                    <div className="mb-6 flex items-center justify-between">
                        <h1 className="text-3xl font-bold">Quản lý Model AI</h1>
                        <div className="flex gap-2">
                            <AlertDialog>
                                <AlertDialogTrigger asChild>
                                    <Button>
                                        <RefreshCw className="mr-2 h-4 w-4" />
                                        Huấn luyện lại mô hình
                                    </Button>
                                </AlertDialogTrigger>
                                <AlertDialogContent>
                                    <AlertDialogHeader>
                                        <AlertDialogTitle>Bạn có chắc không?</AlertDialogTitle>
                                        <AlertDialogDescription>
                                            Việc huấn luyện lại mô hình sẽ sử dụng dữ liệu mới nhất và có thể mất vài giờ để hoàn thành. Mô hình hiện tại sẽ tiếp tục hoạt động cho đến khi quá trình huấn luyện kết thúc.
                                        </AlertDialogDescription>
                                    </AlertDialogHeader>
                                    <AlertDialogFooter>
                                        <AlertDialogCancel>Huỷ bỏ</AlertDialogCancel>
                                        <AlertDialogAction onClick={handleRetrain} disabled={isRetraining}>
                                            {isRetraining ? (
                                                <>
                                                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                                    Đang huấn luyện lại mô hình...
                                                </>
                                            ) : (
                                                "Bắt đầu huấn luyện"
                                            )}
                                        </AlertDialogAction>
                                    </AlertDialogFooter>
                                </AlertDialogContent>
                            </AlertDialog>
                        </div>
                    </div>

                    <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
                        <TabsList className="mb-4">
                            <TabsTrigger value="overview">Tổng quan</TabsTrigger>
                            <TabsTrigger value="performance">Hiệu quả hoạt động</TabsTrigger>
                        </TabsList>

                        <TabsContent value="overview">
                            <div className="grid gap-6 md:grid-cols-2">
                                <Card>
                                    <CardHeader>
                                        <CardTitle>Thông tin Model</CardTitle>
                                        <CardDescription>Thông tin và trạng thái Model</CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="space-y-4">
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Tên Model:</div>
                                                <div>Itransformer</div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Phiên bản:</div>
                                                <div>v1.0</div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Lần huấn luyện gần nhất:</div>
                                                <div>15, Tháng 3, 2025</div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Trạng thái:</div>
                                                <div>
                                                    <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-800">
                                                        Hoạt động
                                                    </span>
                                                </div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Dữ liệu huấn luyện:</div>
                                                <div>3K mẫu</div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Độ chính xác:</div>
                                                <div>90.2%</div>
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>

                                <Card>
                                    <CardHeader>
                                        <CardTitle>Thống kê sử dụng</CardTitle>
                                        <CardDescription>Số liệu sử dụng mô hình trong 30 ngày qua</CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="space-y-4">
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Tổng số yêu cầu:</div>
                                                <div>24</div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Thời gian phản hồi trung bình:</div>
                                                <div>4s</div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Tỷ lệ lỗi:</div>
                                                <div>3%</div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Khách hàng đang hoạt động:</div>
                                                <div>2</div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="text-sm font-medium">Thời điểm sử dụng cao nhất:</div>
                                                <div>17 tháng 3, 2025 (21 yêu cầu)</div>
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>

                            </div>
                        </TabsContent>

                        <TabsContent value="performance">
                            <Card>
                                <CardHeader>
                                    <CardTitle>Chỉ số hiệu suất mô hình</CardTitle>
                                    <CardDescription>Phân tích chi tiết hiệu suất của mô hình</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-6">
                                        <div className="grid gap-6 md:grid-cols-3">
                                            <div className="rounded-lg border p-4">
                                                <div className="text-sm font-medium text-muted-foreground">Độ chính xác</div>
                                                <div className="mt-2 text-2xl font-bold">90.2%</div>
                                            </div>
                                            <div className="rounded-lg border p-4">
                                                <div className="text-sm font-medium text-muted-foreground">Độ chính xác phân loại</div>
                                                <div className="mt-2 text-2xl font-bold">90.8%</div>
                                            </div>
                                            <div className="rounded-lg border p-4">
                                                <div className="text-sm font-medium text-muted-foreground">Khả năng nhận diện</div>
                                                <div className="mt-2 text-2xl font-bold">90.5%</div>
                                            </div>
                                        </div>

                                        <div className="rounded-lg border p-4">
                                            <h3 className="mb-4 text-lg font-medium">Hiệu suất theo thời gian</h3>
                                            <div className="h-64 w-full bg-muted flex items-center justify-center">
                                                <p className="text-muted-foreground">Biểu đồ hiệu suất sẽ hiển thị tại đây</p>
                                            </div>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                        </TabsContent>
                    </Tabs>
                </div>
            </main>
        </div>
    )
}

