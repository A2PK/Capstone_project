// src/components/chat/ChatSidebar.tsx
"use client";

import React from 'react';
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { User } from "@/types/user";
import { Search, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge"; // *** Đảm bảo đã import Badge ***

// Định nghĩa kiểu props cho component
interface ChatSidebarProps {
    contacts: User[]; // Danh sách người dùng hiển thị (đã sắp xếp)
    currentUser: User; // Người dùng hiện tại
    onSelectUser: (user: User) => void; // Hàm gọi khi chọn user
    isLoading: boolean; // Trạng thái loading (cho user list hoặc search)
    selectedUserId?: string | null; // ID của người đang được chọn (để highlight)
    onSearch: (query: string) => void; // Hàm gọi khi thay đổi nội dung tìm kiếm
    searchTerm: string; // Nội dung tìm kiếm hiện tại
    unreadCounts: Map<string, number>; // *** THÊM PROP NHẬN UNREAD COUNTS ***
}

const ChatSidebar: React.FC<ChatSidebarProps> = ({
    contacts,
    currentUser,
    onSelectUser,
    isLoading,
    selectedUserId,
    onSearch,
    searchTerm,
    unreadCounts, // *** NHẬN PROP UNREAD COUNTS ***
}) => {

    // Hàm tiện ích lấy chữ cái đầu của tên
    const getInitials = (firstName?: string, lastName?: string): string => {
        const first = firstName ? firstName[0] : '';
        const last = lastName ? lastName[0] : '';
        // Nếu không có tên nào thì trả về '?' hoặc ký tự mặc định
        if (!first && !last) return '?';
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
                        className="pl-8 w-full bg-white text-sm rounded-md h-9" // Input nhỏ hơn chút
                        value={searchTerm}
                        onChange={(e) => onSearch(e.target.value)}
                        aria-label="Tìm kiếm người dùng"
                    />
                </div>
            </div>

            {/* Danh sách cuộc trò chuyện/liên hệ */}
            <ScrollArea className="flex-grow">
                {/* Hiển thị loader */}
                {isLoading && contacts.length === 0 && !searchTerm && ( // Chỉ hiện loader chính nếu chưa có contact nào và ko tìm kiếm
                    <div className="flex justify-center items-center h-full p-4">
                        <Loader2 className="h-6 w-6 animate-spin text-orange-500" />
                        <span className="ml-2 text-sm text-gray-500">Đang tải...</span>
                    </div>
                )}
                {/* Hiển thị thông báo khi không có kết quả tìm kiếm */}
                {isLoading && searchTerm && ( // Loader nhỏ khi đang search
                     <div className="flex justify-center items-center p-4">
                        <Loader2 className="h-5 w-5 animate-spin text-orange-500" />
                     </div>
                )}
                {!isLoading && contacts.length === 0 && searchTerm && (
                    <p className="text-center text-sm text-gray-500 p-4 italic">Không tìm thấy người dùng.</p>
                )}
                {/* Hiển thị thông báo khi danh sách trống (không tìm kiếm) */}
                {!isLoading && contacts.length === 0 && !searchTerm && (
                    <p className="text-center text-sm text-gray-500 p-4 italic">
                        {currentUser.role === 'officer' ? 'Chưa có quản trị viên hoặc liên hệ nào.' : 'Chưa có người dùng nào khác.'}
                    </p>
                )}
                {/* Hiển thị danh sách người dùng */}
                {!isLoading && contacts.length > 0 && (
                    <div className="py-1">
                        {contacts.map((user) => {
                            // *** LẤY SỐ LƯỢNG TIN NHẮN CHƯA ĐỌC CHO USER NÀY ***
                            const unreadCount = unreadCounts.get(user.id) ?? 0;
                            const hasUnread = unreadCount > 0;
                            const isSelected = selectedUserId === user.id;

                            return (
                                <button
                                    key={user.id}
                                    onClick={() => onSelectUser(user)}
                                    // Cập nhật className để xử lý highlight và unread
                                    className={`flex items-center w-full px-3 py-2.5 text-left transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-orange-400 focus-visible:ring-offset-1 ${
                                        isSelected
                                            ? "bg-orange-100 hover:bg-orange-100" // Selected state
                                            : hasUnread
                                            ? "bg-orange-50/60 hover:bg-gray-100" // Unread state
                                            : "hover:bg-gray-100" // Default state
                                    }`}
                                    aria-current={isSelected ? 'page' : undefined}
                                >
                                    {/* Avatar */}
                                    <Avatar className="h-9 w-9 mr-2.5 flex-shrink-0">
                                        <AvatarImage src={user.profilePic || undefined} alt={`${user.firstName ?? ''} ${user.lastName ?? ''}`} />
                                        <AvatarFallback className="text-xs">{getInitials(user.firstName, user.lastName)}</AvatarFallback>
                                    </Avatar>
                                    {/* Thông tin user */}
                                    <div className="flex-1 overflow-hidden min-w-0">
                                        <div className="flex justify-between items-center gap-2"> {/* Wrapper để badge không đẩy text xuống và có khoảng cách */}
                                            <p className={`text-sm truncate leading-tight ${
                                                hasUnread ? 'font-semibold text-gray-800' : 'font-medium text-gray-700' // In đậm tên nếu có tin chưa đọc
                                            }`}>
                                                {user.firstName ?? ''} {user.lastName ?? ''}
                                                {((user.role === 'admin')||(user.role === 'manager')) && <span className="text-xs text-orange-600 ml-1 font-normal">{user.role === 'admin' ? "(Quản trị viên)" : "(Tổng quản nhiệm)"}</span>}
                                            </p>
                                            {/* *** HIỂN THỊ BADGE SỐ LƯỢNG CHƯA ĐỌC *** */}
                                            {hasUnread && (
                                                <Badge
                                                   variant="destructive" // Màu đỏ
                                                   // Style badge nhỏ gọn
                                                   className="h-5 min-w-[1.25rem] px-1.5 text-xs flex-shrink-0 flex items-center justify-center rounded-full"
                                                >
                                                   {/* Giới hạn hiển thị số lớn */}
                                                   {unreadCount > 99 ? '99+' : unreadCount}
                                                </Badge>
                                            )}
                                        </div>
                                        <p className="text-xs text-gray-500 truncate leading-tight">{user.email}</p>
                                        {/* TODO: Hiển thị tin nhắn cuối cùng (nếu có) */}
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                )}
            </ScrollArea>
        </div>
    );
};

export default ChatSidebar;