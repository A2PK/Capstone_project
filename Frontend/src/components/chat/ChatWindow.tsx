// src/components/chat/ChatWindow.tsx
"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SendHorizonal, Loader2, AlertCircle } from "lucide-react";
import { User } from '@/types/user';
import { Message } from '@/types/message';
import { getAllMessagesBetweenUsers, sendDirectMessage, markAsReadMessages } from '@/lib/message';
import { QueryOptions } from '@/types/station2';
import { toast } from 'react-hot-toast';
import { format, isToday, isYesterday } from 'date-fns'; // Thêm isToday, isYesterday
import { vi } from 'date-fns/locale'; // Vietnamese locale

// Định nghĩa kiểu props
interface ChatWindowProps {
  currentUser: User;
  chatPartner: User | null; // Có thể là null nếu chưa chọn ai
  token: string | null; // Có thể là null nếu chưa đăng nhập xong
}

const POLLING_INTERVAL = 4000; // Thời gian poll tin nhắn mới (ms)
const MESSAGE_FETCH_LIMIT = 40; // Số lượng tin nhắn fetch mỗi lần

const ChatWindow: React.FC<ChatWindowProps> = ({ currentUser, chatPartner, token }) => {
  // State lưu danh sách tin nhắn
  const [messages, setMessages] = useState<Message[]>([]);
  // State lưu nội dung tin nhắn đang soạn
  const [newMessage, setNewMessage] = useState('');
  // State loading khi tải tin nhắn
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  // State loading khi đang gửi tin nhắn
  const [isSending, setIsSending] = useState(false);
  // State lưu lỗi (nếu có)
  const [error, setError] = useState<string | null>(null);
  // Ref tới ScrollArea để cuộn
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  // Ref lưu trữ interval ID của polling
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // --- Hàm cuộn xuống dưới cùng ---
  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
     // Dùng setTimeout để đảm bảo DOM đã cập nhật trước khi cuộn
     setTimeout(() => {
        if (scrollAreaRef.current) {
            // Tìm viewport bên trong ScrollArea của Shadcn
            const scrollViewport = scrollAreaRef.current.querySelector('div[data-radix-scroll-area-viewport]');
            if (scrollViewport) {
                // Cuộn xuống dưới cùng
                scrollViewport.scrollTo({ top: scrollViewport.scrollHeight, behavior });
            }
        }
     }, 50); // Delay nhỏ
  };

  // --- Hàm fetch tin nhắn ---
  const fetchMessages = useCallback(async (isPolling = false) => {
    // Chỉ fetch khi có đủ thông tin
    if (!chatPartner || !token || !currentUser) return;
    // Chỉ hiển thị loading khi fetch lần đầu (không phải polling)
    if (!isPolling) {
      setIsLoadingMessages(true);
      setError(null); // Xóa lỗi cũ
    }

    try {
      // Options để lấy tin nhắn mới nhất trước
      const options: QueryOptions = { limit: MESSAGE_FETCH_LIMIT, sortBy: 'created_at', sortDesc: true };
      const fetchedMessagesRaw = await getAllMessagesBetweenUsers(token, currentUser.id, chatPartner.id, options);
      // API trả về mới nhất trước, cần đảo ngược lại để hiển thị đúng thứ tự thời gian
      const fetchedMessages = fetchedMessagesRaw.reverse();

      // Xác định các tin nhắn chưa đọc (do người khác gửi) TỪ DỮ LIỆU VỪA FETCH
      const unreadMessageIds = fetchedMessages
        .filter(msg => !msg.read && msg.receiverId === currentUser.id)
        .map(msg => msg.id);

      let finalMessages = fetchedMessages; // Mảng tin nhắn cuối cùng để set state

      // Gọi API đánh dấu đã đọc NẾU CÓ tin nhắn chưa đọc
      if (unreadMessageIds.length > 0) {
        console.log(`Marking ${unreadMessageIds.length} messages as read:`, unreadMessageIds);
        try {
          // !!! Cảnh báo: API endpoint `/api/v1/messages` cho việc mark read có thể không đúng
          await markAsReadMessages(token, { messageIds: unreadMessageIds });
          // Cập nhật trạng thái read trong mảng finalMessages SAU KHI API thành công
          finalMessages = fetchedMessages.map(msg =>
            unreadMessageIds.includes(msg.id) ? { ...msg, read: true } : msg
          );
        } catch (readError) {
          console.error("Failed to mark messages as read (API endpoint might be incorrect):", readError);
          toast.error("Lỗi đánh dấu tin nhắn đã đọc.");
          // Nếu lỗi, vẫn dùng fetchedMessages (chưa cập nhật read status)
        }
      }

      // --- Cập nhật state chỉ MỘT LẦN với dữ liệu cuối cùng ---
      // Chỉ cập nhật nếu thực sự có thay đổi (tránh re-render không cần thiết khi polling)
      setMessages(currentMessages => {
          if (JSON.stringify(finalMessages) !== JSON.stringify(currentMessages)) {
              // Nếu là lần fetch đầu tiên, cuộn xuống dưới
              if (!isPolling) {
                  scrollToBottom('auto'); // Cuộn tức thì lần đầu
              } else if (finalMessages.length > currentMessages.length) {
                  // Nếu polling có tin nhắn mới, cuộn mượt
                  scrollToBottom('smooth');
              }
              return finalMessages; // Trả về mảng mới
          }
          return currentMessages; // Không có thay đổi, trả về mảng cũ
      });

    } catch (err: any) {
      console.error("Error fetching messages:", err);
      if (!isPolling) { // Chỉ hiển thị lỗi khi fetch lần đầu
        setError("Không thể tải tin nhắn. Vui lòng thử lại.");
        toast.error("Lỗi tải tin nhắn.");
      }
    } finally {
      if (!isPolling) {
        setIsLoadingMessages(false); // Kết thúc loading lần đầu
      }
    }
  }, [chatPartner, token, currentUser]); // Dependencies cho useCallback

  // --- useEffect quản lý fetch và polling ---
  useEffect(() => {
    setMessages([]); // Xóa tin nhắn cũ khi đổi người chat
    setError(null);   // Xóa lỗi cũ
    setIsLoadingMessages(true); // Bắt đầu loading khi chọn người mới

    // Dọn dẹp interval cũ trước khi thiết lập cái mới
    if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
    }

    if (chatPartner && token) {
      fetchMessages(); // Fetch lần đầu khi chọn partner

      // Bắt đầu polling
      pollingIntervalRef.current = setInterval(() => {
        fetchMessages(true); // Fetch dạng polling
      }, POLLING_INTERVAL);
      console.log(`Polling started for chat with ${chatPartner.id}`);

    } else {
        setIsLoadingMessages(false); // Dừng loading nếu không có partner/token
    }

    // Hàm cleanup: Dọn dẹp interval khi component unmount hoặc dependencies thay đổi
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
        console.log(`Polling stopped for chat with ${chatPartner?.id}`);
      }
    };
    // Phụ thuộc vào chatPartner và token (fetchMessages đã được useCallback)
  }, [chatPartner, token, fetchMessages]); // fetchMessages thay đổi khi chatPartner/token/currentUser thay đổi

  // --- Hàm gửi tin nhắn ---
  const handleSendMessage = useCallback(async () => {
    // Điều kiện kiểm tra trước khi gửi
    if (!newMessage.trim() || !chatPartner || !token || isSending) return;

    setIsSending(true); // Bắt đầu trạng thái đang gửi
    const messageToSend = newMessage.trim();
    setNewMessage(''); // Xóa input ngay lập tức

    // --- Optimistic Update (Thêm tin nhắn tạm thời vào UI) ---
    const optimisticId = `temp-${Date.now()}`;
    const optimisticMessage: Message = {
      id: optimisticId,
      senderId: currentUser.id,
      receiverId: chatPartner.id,
      message: messageToSend,
      read: false, // Tin nhắn mới chưa được đọc
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      deletedAt: null,
    };
    setMessages(prev => [...prev, optimisticMessage]);
    scrollToBottom('smooth'); // Cuộn xuống khi gửi tin nhắn mới

    try {
      // Dữ liệu gửi đi
      const data = { message: messageToSend, receiverId: chatPartner.id };
      // Gọi API
      const sentMessage = await sendDirectMessage(token, data);

      // --- Cập nhật UI sau khi API thành công ---
      // Thay thế tin nhắn tạm thời bằng tin nhắn thật từ server
      setMessages(prev => prev.map(msg => msg.id === optimisticId ? sentMessage : msg));
      // Không cần fetch lại ngay, polling sẽ xử lý hoặc có thể fetch nếu muốn đảm bảo 100% đồng bộ

    } catch (err: any) {
      console.error("Error sending message:", err);
      toast.error("Gửi tin nhắn thất bại: " + err.message);
      // --- Xóa tin nhắn tạm thời nếu gửi lỗi ---
      setMessages(prev => prev.filter(msg => msg.id !== optimisticId));
      // Khôi phục nội dung input nếu gửi lỗi
      setNewMessage(messageToSend);
    } finally {
      setIsSending(false); // Kết thúc trạng thái đang gửi
    }
  }, [newMessage, chatPartner, token, isSending, currentUser]); // Dependencies

  // --- Xử lý thay đổi input và nhấn Enter ---
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewMessage(e.target.value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Gửi khi nhấn Enter (và không nhấn Shift)
    if (e.key === 'Enter' && !e.shiftKey && !isSending) {
      e.preventDefault(); // Ngăn không xuống dòng trong input
      handleSendMessage();
    }
  };

  // --- Hàm tiện ích lấy chữ cái đầu ---
  const getInitials = (firstName?: string, lastName?: string): string => {
    const first = firstName ? firstName[0] : '';
    const last = lastName ? lastName[0] : '';
     if (!first && !last) return 'U';
    return `${first}${last}`.toUpperCase();
  };

  // --- Hàm định dạng thời gian tin nhắn ---
   const formatMessageTime = (isoDate: string): string => {
       const date = new Date(isoDate);
       if (isToday(date)) {
           return format(date, 'HH:mm'); // Hôm nay: chỉ giờ:phút
       }
       if (isYesterday(date)) {
           return `Hôm qua, ${format(date, 'HH:mm')}`; // Hôm qua: "Hôm qua, giờ:phút"
       }
       return format(date, 'HH:mm, dd/MM/yyyy'); // Cũ hơn: giờ:phút, ngày/tháng/năm
   };

  // --- Render ---
  // Nếu chưa chọn người chat
  if (!chatPartner) {
    return (
      <div className="flex-1 flex items-center justify-center bg-white h-full p-4 text-center">
        <p className="text-gray-500">Chọn một người trong danh sách để bắt đầu trò chuyện.</p>
      </div>
    );
  }

  // Render cửa sổ chat chính
  return (
    <div className="flex-1 flex flex-col bg-white h-[90%] border-l">
      {/* Header của cửa sổ chat */}
      <div className="flex items-center p-3 border-b shadow-sm">
        <Avatar className="h-9 w-9 mr-3 flex-shrink-0">
           <AvatarImage src={chatPartner.profilePic || undefined} alt={`${chatPartner.firstName} ${chatPartner.lastName}`} />
           <AvatarFallback className="text-xs">{getInitials(chatPartner.firstName, chatPartner.lastName)}</AvatarFallback>
        </Avatar>
        <div className="flex-1 min-w-0">
          <p className="font-semibold truncate text-sm">{chatPartner.firstName} {chatPartner.lastName}</p>
           {chatPartner.role === 'admin' && <p className="text-xs text-orange-600">Quản trị viên</p>}
          {/* TODO: Thêm trạng thái online/offline nếu có */}
        </div>
        {/* TODO: Thêm nút gọi điện, video call,... nếu cần */}
      </div>

      {/* Khu vực hiển thị tin nhắn */}
      <ScrollArea className="flex-grow p-4 space-y-3 bg-gray-50/50" ref={scrollAreaRef}>
         {/* Hiển thị loading */}
         {isLoadingMessages && messages.length === 0 && ( // Chỉ hiện loading nếu chưa có tin nhắn nào
             <div className="flex justify-center items-center h-40">
                <Loader2 className="h-6 w-6 animate-spin text-orange-500" />
             </div>
         )}
         {/* Hiển thị lỗi */}
         {error && !isLoadingMessages && messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-40 text-red-600 text-center">
                  <AlertCircle className="h-8 w-8 mb-2" />
                  <p className="text-sm">{error}</p>
              </div>
          )}
         {/* Hiển thị danh sách tin nhắn */}
         {!isLoadingMessages && !error && messages.length === 0 && (
              <div className="flex justify-center items-center h-40">
                   <p className="text-sm text-gray-500 italic">Chưa có tin nhắn nào.</p>
               </div>
          )}
         {!error && messages.map((msg) => (
           <div
             key={msg.id.startsWith('temp-') ? msg.id : msg.id} // Sử dụng ID, kể cả ID tạm thời
             className={`flex items-end text-sm mb-1 ${msg.senderId === currentUser.id ? 'justify-end' : 'justify-start'}`}
           >
              {/* Avatar cho tin nhắn nhận (tùy chọn) */}
             {/* {msg.senderId !== currentUser.id && (
                  <Avatar className="h-6 w-6 mr-2 flex-shrink-0 self-end mb-1">
                      <AvatarImage src={chatPartner.profilePic || undefined} />
                      <AvatarFallback className="text-xs">{getInitials(chatPartner.firstName, chatPartner.lastName)}</AvatarFallback>
                  </Avatar>
             )} */}
             {/* Bubble tin nhắn */}
             <div
                title={formatMessageTime(msg.createdAt)} // Hiển thị thời gian đầy đủ khi hover
                className={`max-w-[75%] sm:max-w-[65%] px-3 py-2 rounded-lg shadow-sm relative ${
                 msg.senderId === currentUser.id
                   ? 'bg-orange-500 text-white rounded-br-none'
                   : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'
                } ${msg.id.startsWith('temp-') ? 'opacity-70' : ''}`} // Làm mờ tin nhắn tạm thời
             >
               <p className="break-words whitespace-pre-wrap">{msg.message}</p>
               {/* Thời gian (hiển thị gọn hơn, chi tiết khi hover) */}
                <p className={`text-xs mt-1 opacity-80 ${msg.senderId === currentUser.id ? 'text-orange-100' : 'text-gray-500'} text-right`}>
                  {format(new Date(msg.createdAt), 'HH:mm')}
                   {/* Chỉ báo đã gửi/đã xem (ví dụ) */}
                   {/* {msg.senderId === currentUser.id && !msg.id.startsWith('temp-') && (msg.read ? ' ✓✓' : ' ✓')} */}
               </p>
             </div>
           </div>
         ))}
      </ScrollArea>

      {/* Khu vực nhập tin nhắn */}
      <div className="p-3 border-t bg-gray-50 flex items-center space-x-2">
        {/* TODO: Thêm nút đính kèm file, emoji,... */}
        <Input
          placeholder="Nhập tin nhắn..."
          className="flex-1 bg-white h-10 text-sm" // Kích thước chuẩn
          value={newMessage}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          disabled={isSending} // Vô hiệu hóa khi đang gửi
          aria-label="Soạn tin nhắn"
        />
        <Button
           onClick={handleSendMessage}
           disabled={!newMessage.trim() || isSending} // Vô hiệu hóa khi input trống hoặc đang gửi
           className="bg-orange-500 hover:bg-orange-600 text-white rounded-md h-10 px-4"
           aria-label="Gửi tin nhắn"
           size="icon" // Kích thước nhỏ gọn
        >
          {isSending ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendHorizonal className="h-4 w-4" />}
        </Button>
      </div>
    </div>
  );
};

export default ChatWindow;