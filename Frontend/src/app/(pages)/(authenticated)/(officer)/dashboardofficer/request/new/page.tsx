"use client"
import * as XLSX from "xlsx"
import { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { FileUp, Loader2, X } from "lucide-react"
import {
  AlertDialog,
  AlertDialogTrigger,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogCancel,
  AlertDialogAction,
} from "@/components/ui/alert-dialog"

// Constants
const ALLOWED_EXTENSIONS = ["csv", "xlsx"]
const MAX_FILE_SIZE_MB = 5

export default function NewRequestPage() {
    const expectedHeader = ["Điểm Quan Trắc", "Tỉnh", "Huyện", "Tọa độ", "Ngày quan trắc", "Nhiệt độ", "pH",
                           "DO", "Độ dẫn","Độ kiềm","N-NO2","N-NH4","P-PO4","H2S","TSS","COD","Aeromonas tổng số","Edwardsiella ictaluri",
                           "Aeromonas hydrophila","Coliform","WQI (tính theo Aeromonas)","Chất lượng nước","Chỉ tiêu vượt ngưỡng",
                           "Khuyến cáo"
    ]
  const router = useRouter()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [subject, setSubject] = useState("")
  const [message, setMessage] = useState("")
  const [files, setFiles] = useState<File[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState("")
  const [showErrorDialog, setShowErrorDialog] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files
    if (selectedFiles) {
      const file = selectedFiles[0]
      const reader = new FileReader()
  
      reader.onload = (event) => {
        const data = new Uint8Array(event.target?.result as ArrayBuffer)
        const workbook = XLSX.read(data, { type: "array" })
        const sheetName = workbook.SheetNames[0]
        const worksheet = workbook.Sheets[sheetName]
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 })
  
        const uploadedHeader = jsonData[0] as string[]
        const isValid = JSON.stringify(uploadedHeader) === JSON.stringify(expectedHeader)
  
        if (!isValid) {
          alert("Tệp không đúng định dạng! Vui lòng tải và làm theo file mẫu.")
          return
        }
  
        setFiles(Array.from(selectedFiles))
      }
  
      reader.readAsArrayBuffer(file)
    }
  }

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index))
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!subject.trim()) {
      setError("Tiêu đề bắt buộc nhập.")
      setShowErrorDialog(true)
      return
    }

    if (!message.trim()) {
      setError("Tin nhắn bắt buộc nhập")
      setShowErrorDialog(true)
      return
    }

    setIsSubmitting(true)
    setError("")
    try {
      // Simulated API call
      await new Promise((resolve) => setTimeout(resolve, 2000))

      router.push("/dashboardofficer/request")
    } catch (err) {
      console.error("Lỗi nộp yêu cầu:", err)
      setError("Gửi yêu cầu không thành công. Vui lòng thử lại.")
      setShowErrorDialog(true)
      setIsSubmitting(false)
    }
  }

  return (
    <div className="container mx-auto py-1">
  <Card className="mx-auto">
    <form onSubmit={handleSubmit}>
      <CardHeader>
        <CardTitle>Yêu Cầu Mới</CardTitle>
        <CardDescription>Gửi yêu cầu mới đến ban quản trị</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Tiêu đề yêu cầu */}
        <div className="space-y-2">
          <Label htmlFor="subject">Tiêu Đề</Label>
          <Input
            id="subject"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="Mô tả ngắn gọn về yêu cầu của bạn"
            required
          />
        </div>

        {/* Nội dung yêu cầu */}
        <div className="space-y-2">
          <Label htmlFor="message">Nội Dung</Label>
          <Textarea
            id="message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Chi tiết yêu cầu của bạn"
            className="min-h-[120px]"
            required
          />
        </div>

        {/* Tệp đính kèm */}
        <div className="space-y-2">
        <Label htmlFor="files">Tệp đính kèm (Chỉ chấp nhận CSV hoặc XLSX)</Label>
  <div className="flex items-center gap-2">
    <Input
      id="files"
      ref={fileInputRef}
      type="file"
      onChange={handleFileChange}
      accept=".csv,.xlsx"
      className="hidden"
    />
    <Button
      type="button"
      variant="outline"
      onClick={() => fileInputRef.current?.click()}
      className="flex items-center gap-2"
    >
      <FileUp className="h-4 w-4" />
      Chọn Tệp
    </Button>
    <Button variant="link" asChild className="text-blue-600 underline p-0">
  <a href="/file_mau.xlsx" download>
    Tải file mẫu
  </a>
</Button>
    <span className="text-sm text-muted-foreground">
      {files.length} {files.length === 1 ? "tệp" : "tệp"} đã chọn
    </span>
  </div>

          {files.length > 0 && (
            <div className="mt-4 space-y-2">
              {files.map((file, index) => (
                <div key={index} className="flex items-center justify-between bg-muted p-2 rounded">
                  <div className="flex items-center gap-2 text-sm">
                    <FileUp className="h-4 w-4" />
                    <span>{file.name}</span>
                    <span className="text-muted-foreground">({(file.size / 1024).toFixed(1)} KB)</span>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(index)}
                    className="h-6 w-6 p-0"
                  >
                    <X className="h-4 w-4" />
                    <span className="sr-only">Xóa tệp</span>
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>

      <CardFooter className="flex justify-between">
        <Button type="button" variant="outline" onClick={() => router.push("/dashboardofficer/request")} disabled={isSubmitting}>
          Hủy
        </Button>
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Đang gửi...
            </>
          ) : (
            "Gửi Yêu Cầu"
          )}
        </Button>
      </CardFooter>
    </form>
  </Card>

  {/* Dialog Báo Lỗi */}
  <AlertDialog open={showErrorDialog} onOpenChange={setShowErrorDialog}>
    <AlertDialogContent>
      <AlertDialogHeader>
        <AlertDialogTitle>Lỗi</AlertDialogTitle>
        <AlertDialogDescription>{error}</AlertDialogDescription>
      </AlertDialogHeader>
      <AlertDialogFooter>
        <AlertDialogAction onClick={() => setShowErrorDialog(false)}>OK</AlertDialogAction>
      </AlertDialogFooter>
    </AlertDialogContent>
  </AlertDialog>
</div>

  )
}
