// src/components/article-form.tsx
"use client";

import { useState, useEffect, ChangeEvent, FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2 } from "lucide-react";
import { Article, CreateArticleData } from "@/types/article";
import { FileInfo } from "@/types/file";
import { uploadFile } from "@/lib/file";
import { createArticle, updateArticle } from "@/lib/article";
import { useToast } from "../hooks/use-toast"; // Bạn cần cài đặt: npm install react-hot-toast

interface ArticleFormProps {
  token: string;
  existingArticle?: Article | null; // Dùng khi chỉnh sửa
  onSuccess: (article: Article) => void; // Callback khi thành công
  onCancel: () => void; // Callback khi hủy
}

const DEFAULT_BADGE = 'common';

const ArticleForm: React.FC<ArticleFormProps> = ({ token, existingArticle, onSuccess, onCancel }) => {
  const {toast} = useToast()
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [badge, setBadge] = useState<string>(DEFAULT_BADGE);
  const [pictureFile, setPictureFile] = useState<File | null>(null);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [existingPictureUrl, setExistingPictureUrl] = useState<string | null>(null);
  const [existingFileId, setExistingFileId] = useState<string | null>(null); // Chỉ lưu ID file PDF chính

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isEditing = !!existingArticle;

  useEffect(() => {
    if (isEditing && existingArticle) {
      setTitle(existingArticle.title);
      setContent(existingArticle.content);
      setBadge(existingArticle.badge || DEFAULT_BADGE);
      setExistingPictureUrl(existingArticle.pictureUrl);
      // Giả sử file PDF chính là file đầu tiên trong mảng fileIds
      setExistingFileId(existingArticle.fileIds?.[0] || null);
    } else {
      // Reset form khi tạo mới hoặc không có article
      setTitle('');
      setContent('');
      setBadge(DEFAULT_BADGE);
      setPictureFile(null);
      setPdfFile(null);
      setExistingPictureUrl(null);
      setExistingFileId(null);
    }
  }, [existingArticle, isEditing]);

  const handlePictureChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPictureFile(e.target.files[0]);
    }
  };

  const handlePdfChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPdfFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    let uploadedPictureUrl = existingPictureUrl; // Giữ URL cũ nếu không thay đổi
    let uploadedFileId = existingFileId; // Giữ ID file cũ nếu không thay đổi

    try {
      // 1. Upload ảnh nếu có file mới
      if (pictureFile) {
        const pictureFormData = {
          type: 'article_image', // Hoặc loại phù hợp
          name: pictureFile.name,
          file: pictureFile,
        };
        console.log("Uploading picture...");
        const uploadedImageInfo: FileInfo = await uploadFile(token, pictureFormData);
        uploadedPictureUrl = uploadedImageInfo.url;
        console.log("Picture uploaded:", uploadedPictureUrl);
      }

      // 2. Upload PDF nếu có file mới
      if (pdfFile) {
        const pdfFormData = {
          type: 'article_pdf', // Hoặc loại phù hợp
          name: pdfFile.name,
          file: pdfFile,
        };
        console.log("Uploading PDF...");
        const uploadedFileInfo: FileInfo = await uploadFile(token, pdfFormData);
        uploadedFileId = uploadedFileInfo.id;
        console.log("PDF uploaded:", uploadedFileId);
      }

      // 3. Chuẩn bị dữ liệu cho article
      const articleData = {
        authorId: '', // TODO: Lấy authorId từ user đang đăng nhập nếu cần
        title,
        content,
        badge,
        pictureUrl: uploadedPictureUrl || '', // Đảm bảo không phải null/undefined
        fileIds: uploadedFileId ? [uploadedFileId] : [], // Chỉ lưu ID file PDF mới hoặc cũ
      };

      // 4. Gọi API tạo hoặc cập nhật
      let resultArticle: Article;
      if (isEditing && existingArticle) {
        console.log("Updating article:", existingArticle.id, articleData);
        resultArticle = await updateArticle(token, existingArticle.id, articleData);
        toast({
            title: "Thành công",
            description: "Cập nhật bản tin thành công.",
            variant: "success", // hoặc 'success'
        });
    
      } else {
        console.log("Creating article:", articleData);
        resultArticle = await createArticle(token, articleData as CreateArticleData); // Cần ép kiểu vì thiếu authorId
        toast({
            title: "Thành công",
            description: "Đăng bản tin thành công!",
            variant: "success", // hoặc 'success'
        });
      }

      onSuccess(resultArticle); // Gọi callback thành công

    } catch (err: any) {
      console.error("Error submitting article:", err);
      setError(err.message || "Đã xảy ra lỗi. Vui lòng thử lại.");
      toast({
        title: "Lỗi",
        description: err.message || "Đã xảy ra lỗi khi gửi bản tin.",
        variant: "destructive", // hoặc 'success'
    });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {error && <p className="text-red-600 text-center">{error}</p>}

      <div>
        <Label htmlFor="title" className="text-lg font-medium">Tiêu đề</Label>
        <Input
          id="title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          required
          className="mt-1"
          placeholder="Nhập tiêu đề bản tin"
        />
      </div>

      <div>
        <Label htmlFor="content" className="text-lg font-medium">Nội dung tóm tắt</Label>
        <Textarea
          id="content"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          required
          className="mt-1"
          rows={4}
          placeholder="Nhập mô tả ngắn gọn cho bản tin"
        />
      </div>

      <div>
        <Label htmlFor="badge" className="text-lg font-medium">Mức độ</Label>
        <Select value={badge} onValueChange={setBadge}>
          <SelectTrigger className="w-full mt-1">
            <SelectValue placeholder="Chọn mức độ cảnh báo" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="common">Thông thường</SelectItem>
            <SelectItem value="good">Tốt</SelectItem>
            <SelectItem value="danger">Khẩn cấp</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
         <div>
            <Label htmlFor="picture" className="text-lg font-medium">Ảnh đại diện</Label>
             {existingPictureUrl && !pictureFile && (
                <div className="mt-2 text-sm">
                    Ảnh hiện tại: <a href={existingPictureUrl} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline truncate block">{existingPictureUrl}</a>
                </div>
             )}
            <Input
            id="picture"
            type="file"
            accept="image/*"
            onChange={handlePictureChange}
            className="mt-1 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-orange-50 file:text-orange-700 hover:file:bg-orange-100"
            />
            {pictureFile && <p className="mt-1 text-sm text-gray-600">Đã chọn: {pictureFile.name}</p>}
             <p className="mt-1 text-xs text-gray-500">Để trống nếu muốn giữ ảnh cũ (khi sửa) hoặc dùng ảnh mặc định.</p>
        </div>

        <div>
            <Label htmlFor="pdf" className="text-lg font-medium">File PDF nội dung</Label>
             {existingFileId && !pdfFile && (
                <div className="mt-2 text-sm">
                    File PDF hiện tại ID: <span className="text-blue-600 truncate block">{existingFileId}</span>
                     {/* Có thể thêm link xem file nếu API trả về URL */}
                </div>
             )}
            <Input
            id="pdf"
            type="file"
            accept="application/pdf"
            onChange={handlePdfChange}
            required={!isEditing || !existingFileId} // Bắt buộc khi tạo mới, hoặc khi sửa mà chưa có file
            className="mt-1 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            {pdfFile && <p className="mt-1 text-sm text-gray-600">Đã chọn: {pdfFile.name}</p>}
             <p className="mt-1 text-xs text-gray-500">{isEditing ? 'Chọn file mới để thay thế file PDF cũ.' : 'Chọn file PDF chứa nội dung chi tiết.'}</p>
        </div>
      </div>


      <div className="flex justify-end space-x-3 pt-4">
        <Button type="button" variant="outline" onClick={onCancel} disabled={isLoading}>
          Hủy
        </Button>
        <Button type="submit" disabled={isLoading} className="bg-orange-500 hover:bg-orange-600 text-white">
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              {isEditing ? 'Đang cập nhật...' : 'Đang đăng...'}
            </>
          ) : (
            isEditing ? 'Lưu thay đổi' : 'Đăng bản tin'
          )}
        </Button>
      </div>
    </form>
  );
};

export default ArticleForm;