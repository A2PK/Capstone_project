// src/components/chat/ChatLayout.tsx
"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import ChatSidebar from './ChatSideBar';
import ChatWindow from './ChatWindow';
import { User } from '@/types/user'; // Import kiểu User
import { useAuth } from '@/hooks/useAuth'; // Hook xác thực
import PageLoader from '@/components/pageloader'; // Component loader
// Đảm bảo import đúng các hàm API user, bao gồm cả searchUser
import { getAllUsers, searchUser, getUserById, searchParamsUser } from '@/lib/user';
import { getUsers } from '@/types/user'; // Kiểu trả về của API user (và searchUser)
import { QueryOptions } from '@/types/station2'; // Kiểu QueryOptions (đảm bảo đúng đường dẫn)
import { toast } from 'react-hot-toast'; // Thư viện thông báo
// Đảm bảo import đúng hàm API message
import { getUnseenMessagesForCurrentUser } from '@/lib/message';
import { Message } from '@/types/message'; // Import kiểu Message
import axios from 'axios'; // Import axios để kiểm tra lỗi

// Định nghĩa interface searchParamsUser (nếu chưa import được)
// Interface này phải khớp với định nghĩa trong file user.ts


const UNSEEN_POLLING_INTERVAL = 5000; // Thời gian poll tin nhắn chưa đọc (ms)

const ChatLayout = () => {
    // State từ hook xác thực
    const { user: currentUser, token, isLoading: isAuthLoading, isLoggedIn } = useAuth();

    // State quản lý người dùng đang được chọn để chat
    const [selectedChatPartner, setSelectedChatPartner] = useState<User | null>(null);
    // State lưu trữ *tất cả* người dùng fetch được từ API (để lọc client-side)
    const [allUsers, setAllUsers] = useState<User[]>([]);
    // State trạng thái loading khi fetch danh sách người dùng ban đầu
    const [isLoadingUsers, setIsLoadingUsers] = useState(true);
    // State lưu nội dung ô tìm kiếm
    const [searchTerm, setSearchTerm] = useState('');
    // State lưu kết quả tìm kiếm từ API hoặc lọc client-side
    const [searchResults, setSearchResults] = useState<User[]>([]);
    // State trạng thái loading khi đang thực hiện tìm kiếm
    const [isSearching, setIsSearching] = useState(false);
    // State (Set) lưu ID của các partner (không phải admin) đã được officer chọn trước đó
    const [persistedPartnerIds, setPersistedPartnerIds] = useState<Set<string>>(new Set());
    // State (Map) lưu thông tin chi tiết của các partner được "ghim"
    const [persistedPartnerDetails, setPersistedPartnerDetails] = useState<Map<string, User>>(new Map());

    // State và Logic cho tin nhắn chưa đọc
    const [unseenMessages, setUnseenMessages] = useState<Message[]>([]);
    const unseenPollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    // Hàm fetch tin nhắn chưa đọc
    const fetchUnseen = useCallback(async () => {
        if (!token || !isLoggedIn) return;
        try {
            const unseen = await getUnseenMessagesForCurrentUser(token);
            setUnseenMessages(currentUnseen => {
                 if (JSON.stringify(unseen) !== JSON.stringify(currentUnseen)) {
                     console.log("Fetched new unseen messages count:", unseen.length);
                     return unseen;
                 }
                 return currentUnseen;
            });
        } catch (error) {
            console.error("Polling unseen messages failed:", error);
        }
    }, [token, isLoggedIn]);

    // useEffect để quản lý polling tin nhắn chưa đọc
    useEffect(() => {
        if (unseenPollingIntervalRef.current) {
            clearInterval(unseenPollingIntervalRef.current);
            unseenPollingIntervalRef.current = null;
        }
        if (isLoggedIn && token) {
            fetchUnseen();
            unseenPollingIntervalRef.current = setInterval(fetchUnseen, UNSEEN_POLLING_INTERVAL);
            console.log("Unseen message polling started.");
        } else {
             setUnseenMessages([]);
        }
        return () => {
            if (unseenPollingIntervalRef.current) {
                clearInterval(unseenPollingIntervalRef.current);
                unseenPollingIntervalRef.current = null;
                console.log("Unseen message polling stopped.");
            }
        };
    }, [isLoggedIn, token, fetchUnseen]);

    // Tính toán số lượng tin nhắn chưa đọc theo senderId
    const unreadCountsBySender = useMemo(() => {
        const counts = new Map<string, number>();
        if (!currentUser) return counts;
        unseenMessages.forEach(msg => {
             if (msg.receiverId === currentUser.id && !msg.read && msg.senderId) {
                 counts.set(msg.senderId, (counts.get(msg.senderId) || 0) + 1);
             }
        });
        return counts;
    }, [unseenMessages, currentUser]);

    // --- Fetch danh sách người dùng ban đầu ---
    const fetchInitialUsers = useCallback(async () => {
        if (!token || !currentUser) return;
        setIsLoadingUsers(true);
        try {
            const options: QueryOptions = { limit: 1000 };
            const response: getUsers = await getAllUsers(token, options);
            const users = response.users || [];
            setAllUsers(users);
            console.log("Fetched initial users:", users.length);
        } catch (error) {
            console.error("Error fetching initial users:", error);
            toast.error("Lỗi tải danh sách người dùng.");
        } finally {
            setIsLoadingUsers(false);
        }
    }, [token, currentUser]);

    // useEffect chạy fetchInitialUsers
    useEffect(() => {
        if (isLoggedIn && token && currentUser) {
            fetchInitialUsers();
        } else if (!isAuthLoading && !isLoggedIn) {
            setIsLoadingUsers(false);
            setAllUsers([]);
        }
    }, [isLoggedIn, token, currentUser, isAuthLoading, fetchInitialUsers]);

    // --- Fetch thông tin chi tiết cho các persisted partners ---
    const fetchPersistedPartnerDetails = useCallback(async () => {
        if (!token) return;
        const idsToFetch = Array.from(persistedPartnerIds).filter(id =>
            !persistedPartnerDetails.has(id) && !allUsers.some(u => u.id === id)
        );
        if (idsToFetch.length === 0) return;
        console.log("Fetching details for persisted partner IDs:", idsToFetch);
        try {
            const promises = idsToFetch.map(id => getUserById(id /*, token */)); // Truyền token nếu getUserById cần
            const results = await Promise.allSettled(promises);
            setPersistedPartnerDetails(prevMap => {
                const newMap = new Map(prevMap);
                results.forEach((result, index) => {
                    if (result.status === 'fulfilled' && result.value) {
                        newMap.set(idsToFetch[index], result.value);
                    } else if (result.status === 'rejected') {
                        console.error(`Failed to fetch persisted partner ${idsToFetch[index]}:`, result.reason);
                    }
                });
                return newMap;
            });
        } catch (error) {
            console.error("Error fetching details for persisted partners:", error);
            toast.error("Lỗi tải thông tin người dùng đã lưu.");
        }
    }, [persistedPartnerIds, allUsers, persistedPartnerDetails, token]);

    // useEffect chạy fetchPersistedPartnerDetails
    useEffect(() => {
        if (token) {
           fetchPersistedPartnerDetails();
        }
    }, [fetchPersistedPartnerDetails, token]);

    // --- Xử lý tìm kiếm người dùng ---
    const handleSearch = useCallback(async (query: string) => {
        setSearchTerm(query);
        if (!query.trim()) {
            setSearchResults([]);
            setIsSearching(false);
            return;
        }
        setIsSearching(true);
        setSearchResults([]);

        try {
            let foundUsers: User[] = [];
            const lowerCaseQuery = query.toLowerCase();
            if (!currentUser) { setIsSearching(false); return; }

            if (currentUser.role === 'admin') {
                // Admin: Lọc client-side
                foundUsers = allUsers.filter(u =>
                    u.id !== currentUser.id &&
                    (u.username.toLowerCase().includes(lowerCaseQuery) ||
                     u.email.toLowerCase().includes(lowerCaseQuery) ||
                     `${u.firstName ?? ''} ${u.lastName ?? ''}`.toLowerCase().includes(lowerCaseQuery)
                    )
                );
            } else if (currentUser.role === 'officer') {
                // Officer:
                // 1. Tìm Admin (client-side)
                const foundAdmins = allUsers.filter(u =>
                    u.role === 'admin' &&
                    u.id !== currentUser!.id &&
                    (`${u.firstName ?? ''} ${u.lastName ?? ''}`.toLowerCase().includes(lowerCaseQuery) ||
                     u.email.toLowerCase().includes(lowerCaseQuery))
                );

                // 2. Tìm Officer khác bằng email (dùng API searchUser)
                let foundOfficersByEmail: User[] = [];
                const isEmailQuery = query.includes('@') && query.includes('.');

                if (isEmailQuery) {
                    // TẠO PAYLOAD CHO THAM SỐ THỨ BA (`param`) CỦA searchUser
                    const searchPayload: searchParamsUser = {
                        options: {
                            filters: { email: query },
                            limit: 10,
                            offset: 0,
                            sortBy: 'email', // Hoặc giá trị mặc định khác
                            sortDesc: false,
                            includeDeleted: false
                        }
                    };
                    console.log("Searching officers by email with payload (param):", searchPayload);

                    // TẠO PLACEHOLDER CHO THAM SỐ THỨ NHẤT (bị bỏ qua bởi hàm searchUser bạn cung cấp)
                    const placeholderFirstArg = { options: { filters: { email: query }, limit: 10 } };

                    try {
                        console.log("Calling searchUser(placeholderFirstArg, token, searchPayload)");
                        // GỌI searchUser VỚI ĐÚNG 3 THAM SỐ
                        // Lưu ý: token (tham số thứ 2) hiện đang bị bỏ qua bởi hàm searchUser bạn cung cấp
                        const searchResult: getUsers = await searchUser(placeholderFirstArg, token ?? '', searchPayload);

                        foundOfficersByEmail = (searchResult?.users || []).filter(u =>
                            u.role === 'officer' && u.id !== currentUser!.id
                        );
                        console.log("Found officers by email:", foundOfficersByEmail.length);

                    } catch (searchError: unknown) {
                        console.error("Error searching officers by email:", searchError);
                        // Xử lý lỗi chi tiết
                        if (axios.isAxiosError(searchError)) {
                           console.error("Axios error details (searchUser):", { status: searchError.response?.status, data: searchError.response?.data });
                           const status = searchError.response?.status;
                           if (status === 401) { toast.error("Lỗi quyền truy cập khi tìm kiếm."); }
                           else if (status === 400) { const message = searchError.response?.data?.message || "Email tìm kiếm không hợp lệ."; toast.error(`Lỗi tìm kiếm: ${message}`); }
                           else { toast.error("Không thể tìm kiếm Officer qua email lúc này. Lỗi máy chủ hoặc mạng."); }
                        } else if (searchError instanceof Error) {
                           console.error("Non-Axios Error (searchUser):", searchError.message);
                           toast.error(`Đã xảy ra lỗi không mong muốn: ${searchError.message}`);
                       } else {
                           console.error("Caught non-error value (searchUser):", searchError);
                           toast.error("Đã xảy ra lỗi không mong muốn khi tìm kiếm Officer.");
                       }
                        foundOfficersByEmail = []; // Đảm bảo là mảng rỗng khi lỗi
                    }
                }

                // Kết hợp kết quả và loại bỏ trùng lặp
                const combinedResults = [...foundAdmins, ...foundOfficersByEmail];
                const uniqueUserIds = new Set<string>();
                foundUsers = combinedResults.filter(user => {
                    if (!uniqueUserIds.has(user.id)) {
                        uniqueUserIds.add(user.id);
                        return true;
                    }
                    return false;
                });
            }
            setSearchResults(foundUsers);
        } catch (error: unknown) {
             console.error("Error during user search (outer try-catch):", error);
             if (error instanceof Error) {
                 toast.error(`Lỗi trong quá trình tìm kiếm người dùng: ${error.message}`);
             } else {
                 toast.error("Lỗi không xác định trong quá trình tìm kiếm người dùng.");
             }
        } finally {
            setIsSearching(false);
        }
    }, [currentUser, allUsers, token]); // Dependencies

    // --- Xử lý khi chọn một người dùng từ sidebar ---
    const handleSelectUser = useCallback((user: User) => {
        setSelectedChatPartner(user);
        // Logic "ghim" user
        if (currentUser && currentUser.role === 'officer' && user.role !== 'admin') {
            setPersistedPartnerIds(prev => {
                if (prev.has(user.id)) return prev;
                const newSet = new Set(prev);
                newSet.add(user.id);
                console.log("Persisted partner ID added:", user.id);
                return newSet;
            });
            if (!persistedPartnerDetails.has(user.id)) {
                setPersistedPartnerDetails(prevMap => new Map(prevMap).set(user.id, user));
            }
        }
        // Xóa tìm kiếm
        setSearchTerm('');
        setSearchResults([]);
        setIsSearching(false);
        // fetchUnseen(); // Có thể gọi để cập nhật badge ngay, nhưng polling thường là đủ
    }, [currentUser, persistedPartnerDetails]);

    // --- Tính toán danh sách contacts cuối cùng để hiển thị ---
    const displayedContacts = useMemo(() => {
        if (!currentUser) return [];

        if (searchTerm.trim()) {
            return searchResults; // Trả về kết quả tìm kiếm nếu đang tìm
        }

        // Logic hiển thị mặc định (Admin + Officer đã ghim / Tất cả user khác)
        let baseContacts: User[] = [];
        if (currentUser.role === 'admin') {
            baseContacts = allUsers.filter(u => u.id !== currentUser.id);
        } else {
            baseContacts = allUsers.filter(u => u.role === 'admin' && u.id !== currentUser.id);
        }

        const persistedPartners: User[] = Array.from(persistedPartnerIds)
            .map(id => persistedPartnerDetails.get(id) || allUsers.find(u => u.id === id))
            .filter((user): user is User => !!user && user.id !== currentUser.id && user.role !== 'admin');

        const combined = [...baseContacts, ...persistedPartners];
        const uniqueMap = new Map<string, User>();
        combined.forEach(user => {
            if (!uniqueMap.has(user.id)) {
                uniqueMap.set(user.id, user);
            }
        });

        // Sắp xếp: Ưu tiên chưa đọc -> tên
        const sortedContacts = Array.from(uniqueMap.values()).sort((a, b) => {
            const unreadA = unreadCountsBySender.get(a.id) ?? 0;
            const unreadB = unreadCountsBySender.get(b.id) ?? 0;
            if (unreadA > 0 && unreadB === 0) return -1;
            if (unreadA === 0 && unreadB > 0) return 1;
            const nameA = `${a.firstName ?? ''} ${a.lastName ?? ''}`;
            const nameB = `${b.firstName ?? ''} ${b.lastName ?? ''}`;
            return nameA.localeCompare(nameB);
        });

        return sortedContacts;

    }, [currentUser, searchTerm, searchResults, allUsers, persistedPartnerIds, persistedPartnerDetails, unreadCountsBySender]);

    // --- Render ---
    if (isAuthLoading || (isLoadingUsers && !currentUser && isLoggedIn)) {
        return <PageLoader message={isAuthLoading ? "Đang tải thông tin xác thực..." : "Đang tải danh sách người dùng..."} />;
    }
    if (!isLoggedIn) {
        return ( <div className="flex items-center justify-center h-full text-center p-4"> Vui lòng đăng nhập để sử dụng tính năng chat. </div> );
    }
    if (!currentUser) {
        return <PageLoader message="Đang khởi tạo dữ liệu người dùng..." />;
    }

    return (
        <div className="flex h-full border-t">
            <ChatSidebar
                contacts={displayedContacts}
                currentUser={currentUser}
                onSelectUser={handleSelectUser}
                isLoading={isLoadingUsers || isSearching}
                selectedUserId={selectedChatPartner?.id}
                onSearch={handleSearch}
                searchTerm={searchTerm}
                unreadCounts={unreadCountsBySender} // Truyền unread counts
            />
            <ChatWindow
                key={selectedChatPartner?.id || currentUser.id}
                currentUser={currentUser}
                chatPartner={selectedChatPartner}
                token={token ?? null}
            />
        </div>
    );
};

export default ChatLayout;