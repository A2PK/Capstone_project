// src/components/chat/ChatSidebar.tsx
"use client";

import React from 'react';
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { User } from "@/types/user";
import { Search, Loader2 } from "lucide-react";

// Định nghĩa kiểu props cho component
interface ChatSidebarProps {
  contacts: User[]; // Danh sách người dùng hiển thị
  currentUser: User; // Người dùng hiện tại
  onSelectUser: (user: User) => void; // Hàm gọi khi chọn user
  isLoading: boolean; // Trạng thái loading
  selectedUserId?: string | null; // ID của người đang được chọn (để highlight)
  onSearch: (query: string) => void; // Hàm gọi khi thay đổi nội dung tìm kiếm
  searchTerm: string; // Nội dung tìm kiếm hiện tại
}

const ChatSidebar: React.FC<ChatSidebarProps> = ({
  contacts,
  currentUser,
  onSelectUser,
  isLoading,
  selectedUserId,
  onSearch,
  searchTerm,
}) => {

  // Hàm tiện ích lấy chữ cái đầu của tên
  const getInitials = (firstName?: string, lastName?: string): string => {
    const first = firstName ? firstName[0] : '';
    const last = lastName ? lastName[0] : '';
    // Nếu không có tên nào thì trả về icon user hoặc ký tự đặc biệt
    if (!first && !last) return 'U';
    return `${first}${last}`.toUpperCase();
  };

  return (
    <div className="w-full max-w-xs border-r bg-gray-50 flex flex-col h-full flex-shrink-0">
      {/* Thanh tìm kiếm */}
      <div className="p-3 border-b">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
          <Input
            type="search"
            placeholder={currentUser.role === 'officer' ? "Tìm Admin hoặc email Officer..." : "Tìm kiếm người dùng..."}
            className="pl-8 w-full bg-white text-sm" // Kích thước text nhỏ hơn
            value={searchTerm}
            onChange={(e) => onSearch(e.target.value)}
            aria-label="Tìm kiếm người dùng"
          />
        </div>
      </div>

      {/* Danh sách cuộc trò chuyện/liên hệ */}
      <ScrollArea className="flex-grow">
        {/* Hiển thị loader */}
        {isLoading && (
          <div className="flex justify-center items-center h-full p-4">
            <Loader2 className="h-6 w-6 animate-spin text-orange-500" />
          </div>
        )}
        {/* Hiển thị thông báo khi không có kết quả hoặc danh sách trống */}
        {!isLoading && contacts.length === 0 && searchTerm && (
           <p className="text-center text-sm text-gray-500 p-4 italic">Không tìm thấy người dùng.</p>
        )}
         {!isLoading && contacts.length === 0 && !searchTerm && (
             <p className="text-center text-sm text-gray-500 p-4 italic">
                 {currentUser.role === 'officer' ? 'Chưa có quản trị viên hoặc liên hệ nào.' : 'Chưa có người dùng nào khác.'}
             </p>
        )}
        {/* Hiển thị danh sách người dùng */}
        {!isLoading && contacts.length > 0 && (
          <div className="py-1"> {/* Giảm padding */}
            {contacts.map((user) => (
              <button
                key={user.id}
                onClick={() => onSelectUser(user)}
                className={`flex items-center w-full px-3 py-2.5 text-left hover:bg-gray-100 transition-colors duration-150 focus:outline-none focus:bg-gray-100 ${
                  selectedUserId === user.id ? "bg-orange-100 hover:bg-orange-100" : "" // Highlight người được chọn
                }`}
                aria-current={selectedUserId === user.id ? 'page' : undefined}
              >
                {/* Avatar */}
                <Avatar className="h-9 w-9 mr-2.5 flex-shrink-0"> {/* Kích thước nhỏ hơn */}
                  <AvatarImage src={user.profilePic || undefined} alt={`${user.firstName} ${user.lastName}`} />
                  <AvatarFallback className="text-xs">{getInitials(user.firstName, user.lastName)}</AvatarFallback>
                </Avatar>
                {/* Thông tin user */}
                <div className="flex-1 overflow-hidden min-w-0">
                  <p className="font-medium text-sm truncate leading-tight">
                    {user.firstName} {user.lastName}
                    {/* Hiển thị (Admin) nếu là admin */}
                    {user.role === 'admin' && <span className="text-xs text-orange-600 ml-1 font-normal">(Admin)</span>}
                  </p>
                  {/* Email hoặc thông tin khác */}
                  <p className="text-xs text-gray-500 truncate leading-tight">{user.email}</p>
                  {/* TODO: Hiển thị tin nhắn cuối cùng và trạng thái chưa đọc */}
                </div>
                {/* TODO: Hiển thị thời gian và chỉ báo chưa đọc */}
              </button>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
};

export default ChatSidebar;