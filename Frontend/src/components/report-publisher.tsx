"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, AlertTriangle, BookOpen, CheckCircle, FileText, Info, Upload } from "lucide-react"
import { cn } from "@/lib/utils"

type BadgeType = "important" | "danger" | "urgent"

interface ReportData {
  title: string
  description: string
  badgeType: BadgeType | null
  backCoverImage: string | null
}

export default function ReportPublisher() {
  const [reportData, setReportData] = useState<ReportData>({
    title: "",
    description: "",
    badgeType: null,
    backCoverImage: null,
  })
  const [isPublished, setIsPublished] = useState(false)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      const reader = new FileReader()

      reader.onload = (event) => {
        if (event.target) {
          setReportData({
            ...reportData,
            backCoverImage: event.target.result as string,
          })
        }
      }

      reader.readAsDataURL(file)
    }
  }

  const handlePublish = () => {
    // In a real application, you would send the data to your backend here
    setIsPublished(true)

    // Reset after 3 seconds
    setTimeout(() => {
      setIsPublished(false)
    }, 3000)
  }

  const getBadgeColor = (type: BadgeType | null) => {
    switch (type) {
      case "important":
        return "bg-blue-600 hover:bg-blue-700"
      case "danger":
        return "bg-red-600 hover:bg-red-700"
      case "urgent":
        return "bg-amber-600 hover:bg-amber-700"
      default:
        return "bg-gray-600 hover:bg-gray-700"
    }
  }

  const getBadgeIcon = (type: BadgeType | null) => {
    switch (type) {
      case "important":
        return <Info className="h-4 w-4 mr-1" />
      case "danger":
        return <AlertCircle className="h-4 w-4 mr-1" />
      case "urgent":
        return <AlertTriangle className="h-4 w-4 mr-1" />
      default:
        return null
    }
  }

  return (
    <div className="relative">
      {/* Decorative elements */}
      <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 w-32 h-1 bg-gradient-to-r from-amber-300 via-amber-500 to-amber-300"></div>

      <div className="grid md:grid-cols-2 gap-8 mt-1">
        <div className="space-y-6">
          <Card className="border-0 shadow-lg bg-gradient-to-b from-white to-gray-50 dark:from-gray-900 dark:to-gray-950 overflow-hidden">
            <div className="bg-gradient-to-r from-amber-50 via-amber-100 to-amber-50 dark:from-gray-800 dark:via-gray-700 dark:to-gray-800 p-6 border-b border-amber-200 dark:border-gray-700">
              <div className="flex items-center space-x-3">
                <FileText className="h-6 w-6 text-amber-700 dark:text-amber-400" />
                <h2 className="text-xl font-serif font-semibold text-gray-800 dark:text-gray-100">
                Tạo báo cáo
                </h2>
              </div>
            </div>

            <CardContent className="p-6">
              <form className="space-y-6">
                <div className="space-y-2">
                  <Label
                    htmlFor="title"
                    className="text-sm font-medium uppercase tracking-wide text-gray-700 dark:text-gray-300"
                  >
                    Tiêu đề báo cáo
                  </Label>
                  <Input
                    id="title"
                    placeholder="Nhập tiêu đề cho báo cáo"
                    value={reportData.title}
                    onChange={(e) => setReportData({ ...reportData, title: e.target.value })}
                    className="border-gray-300 dark:border-gray-600 focus:ring-amber-500 focus:border-amber-500 dark:focus:ring-amber-400 dark:focus:border-amber-400"
                  />
                </div>

                <div className="space-y-2">
                  <Label
                    htmlFor="description"
                    className="text-sm font-medium uppercase tracking-wide text-gray-700 dark:text-gray-300"
                  >
                    Mô tả
                  </Label>
                  <Textarea
                    id="description"
                    placeholder="Nhập mô tả cho báo cáo"
                    className="min-h-[180px] border-gray-300 dark:border-gray-600 focus:ring-amber-500 focus:border-amber-500 dark:focus:ring-amber-400 dark:focus:border-amber-400 whitespace-pre-wrap"
                    value={reportData.description}
                    onChange={(e) => setReportData({ ...reportData, description: e.target.value })}
                  />
                </div>

                <div className="space-y-2">
                  <Label
                    htmlFor="badge-type"
                    className="text-sm font-medium uppercase tracking-wide text-gray-700 dark:text-gray-300"
                  >
                    Phân loại
                  </Label>
                  <Select onValueChange={(value: BadgeType) => setReportData({ ...reportData, badgeType: value })}>
                    <SelectTrigger
                      id="badge-type"
                      className="border-gray-300 dark:border-gray-600 focus:ring-amber-500 focus:border-amber-500 dark:focus:ring-amber-400 dark:focus:border-amber-400"
                    >
                      <SelectValue placeholder="Dạng báo cáo" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="important">Bình thường</SelectItem>
                      <SelectItem value="danger">Nguy hiểm</SelectItem>
                      <SelectItem value="urgent">Rủi ro</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label
                    htmlFor="back-cover"
                    className="text-sm font-medium uppercase tracking-wide text-gray-700 dark:text-gray-300"
                  >
                    Ảnh bìa
                  </Label>
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-md p-6 text-center hover:border-amber-500 dark:hover:border-amber-400 transition-colors duration-200">
                    <Upload className="h-8 w-8 mx-auto text-gray-400 dark:text-gray-500 mb-2" />
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Tải lên một ảnh cho bìa</p>
                    <Input
                      id="back-cover"
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      className="mt-2 border-amber-500 text-amber-700 hover:bg-amber-50 dark:border-amber-400 dark:text-amber-300 dark:hover:bg-gray-800"
                      onClick={() => document.getElementById("back-cover")?.click()}
                    >
                      Chọn tệp
                    </Button>
                    {reportData.backCoverImage && (
                      <p className="text-sm text-green-600 dark:text-green-400 mt-2 flex items-center justify-center">
                        <CheckCircle className="h-4 w-4 mr-1" /> Ảnh đã được tải lên
                      </p>
                    )}
                  </div>
                </div>

                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <Button
                    type="button"
                    className="w-full bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-700 hover:to-amber-800 text-white"
                    onClick={handlePublish}
                    disabled={!reportData.title || !reportData.description}
                  >
                    {isPublished ? "Published Successfully" : "Publish Expert Report"}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="border-0 shadow-lg overflow-hidden bg-white dark:bg-gray-900">
            <div className="bg-gradient-to-r from-gray-800 via-gray-900 to-gray-800 p-6 border-b border-gray-700">
              <div className="flex items-center space-x-3">
                <BookOpen className="h-6 w-6 text-amber-400" />
                <h2 className="text-xl font-serif font-semibold text-white">Xem trước</h2>
              </div>
            </div>

            <CardContent className="p-0">
              <div className="p-8 bg-white dark:bg-gray-900 min-h-[200px]">
                {/* Decorative header */}
                <div className="border-b-2 border-amber-200 dark:border-amber-800 pb-4 mb-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-2xl font-serif font-bold text-gray-900 dark:text-gray-100">
                      {reportData.title || "Tiêu đề báo cáo"}
                    </h3>

                    {reportData.badgeType && (
                      <Badge className={cn("flex items-center font-semibold", getBadgeColor(reportData.badgeType))}>
                        {getBadgeIcon(reportData.badgeType)}
                        {reportData.badgeType.charAt(0).toUpperCase() + reportData.badgeType.slice(1)}
                      </Badge>
                    )}
                  </div>

                  <div className="flex items-center mt-2 text-sm text-gray-500 dark:text-gray-400">
                    <span>
                    Ngày {new Date().toLocaleDateString("vi-VN", { year: "numeric", month: "long", day: "numeric" }).replace(",", "")}

                    </span>
                    <span className="mx-2">•</span>
                    <span>
                    Mã tham chiếu: EXP-
                      {Math.floor(Math.random() * 10000)
                        .toString()
                        .padStart(4, "0")}
                    </span>
                  </div>
                </div>

                {/* Report content */}
                <div className="prose prose-gray dark:prose-invert max-w-none">
                  <p className="text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap break-words">
                    {reportData.description ||
                      "Nội dung mô tả báo cáo sẽ hiển thị ở đây."}
                  </p>
                </div>

                {/* Decorative footer */}
                <div className="mt-8 pt-1 border-t border-gray-200 dark:border-gray-700 text-sm text-gray-500 dark:text-gray-400 flex justify-between items-center">
                  <span>Confidential</span>
                  <span>Trang 1/1</span>
                </div>
              </div>

              {/* Back cover preview */}
              {reportData.backCoverImage && (
                <div className="border-t border-gray-200 dark:border-gray-700">
                  <div className="p-6 bg-gray-50 dark:bg-gray-800">
                    <h4 className="text-sm font-medium uppercase tracking-wide text-gray-700 dark:text-gray-300 mb-4">
                      Bìa
                    </h4>
                    <div className="aspect-video relative overflow-hidden rounded-md shadow-md">
                      <img
                        src={reportData.backCoverImage || "/placeholder.svg"}
                        alt="Back cover preview"
                        className="object-cover w-full h-full"
                      />
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {isPublished && (
            <div className="bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 text-green-800 dark:text-green-300 p-4 rounded-md shadow-sm">
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 mr-2 text-green-600 dark:text-green-400" />
                <p className="font-medium">Báo cáo đã được tải lên thành công!</p>
              </div>
              <p className="text-sm mt-1 text-green-700 dark:text-green-400">
              Báo cáo đã được tải lên và được công khai.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom decorative element */}
      <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 w-32 h-1 bg-gradient-to-r from-amber-300 via-amber-500 to-amber-300"></div>
    </div>
  )
}

