export interface NotificationItem {
    id: string;
    type: 'alert' | 'info' | 'update' | 'warning'; // Loại thông báo
    text: string;         // Nội dung thông báo
    timestamp: Date;      // Thời gian nhận
    read: boolean;        // Trạng thái đã đọc
    link?: string;        // Link tùy chọn khi click vào
  }
  
  // Định nghĩa kiểu dữ liệu cho tin nhắn/thư
 export interface MessageItem {
    id: string;
    sender: string;       // Người gửi
    avatar?: string;      // Avatar người gửi (URL)
    subject: string;      // Chủ đề
    snippet: string;      // Đoạn trích ngắn
    timestamp: Date;      // Thời gian nhận
    read: boolean;        // Trạng thái đã đọc
  }