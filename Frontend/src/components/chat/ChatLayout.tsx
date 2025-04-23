// src/components/chat/ChatLayout.tsx
"use client";

import { useState, useEffect, useCallback, useMemo } from 'react';
import ChatSidebar from './ChatSideBar';
import ChatWindow from './ChatWindow';
import { User } from '@/types/user'; // Import kiểu User
import { useAuth } from '@/hooks/useAuth'; // Hook xác thực
import PageLoader from '@/components/pageloader'; // Component loader
import { getAllUsers, searchUser, getUserById } from '@/lib/user'; // API user
import { getUsers } from '@/types/user'; // Kiểu trả về của API user
import { QueryOptions } from '@/types/station2'; // Kiểu QueryOptions (đảm bảo đúng đường dẫn)
import { toast } from 'react-hot-toast'; // Thư viện thông báo

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

  // --- Fetch danh sách người dùng ban đầu ---
  const fetchInitialUsers = useCallback(async () => {
     if (!token || !currentUser) return; // Chỉ fetch khi có token và currentUser
     setIsLoadingUsers(true);
     try {
        // Lấy nhiều user, có thể cần pagination nếu quá lớn
        const options: QueryOptions = { limit: 1000 };
        const response: getUsers = await getAllUsers(token, options); // API yêu cầu token
        const users = response.users || [];
        setAllUsers(users); // Lưu tất cả vào state `allUsers`
        console.log("Fetched initial users:", users.length);
     } catch (error) {
       console.error("Error fetching initial users:", error);
       toast.error("Lỗi tải danh sách người dùng.");
     } finally {
       setIsLoadingUsers(false);
     }
  }, [token, currentUser]);

  // Chạy fetchInitialUsers khi component mount và các dependency thay đổi
  useEffect(() => {
    if (isLoggedIn && token && currentUser) {
      fetchInitialUsers();
    } else if (!isAuthLoading && !isLoggedIn) {
        // Xử lý khi người dùng chưa đăng nhập (nếu cần)
        setIsLoadingUsers(false);
    }
  }, [isLoggedIn, token, currentUser, isAuthLoading, fetchInitialUsers]);

  // --- Fetch thông tin chi tiết cho các persisted partners ---
  const fetchPersistedPartnerDetails = useCallback(async () => {
    // Lấy danh sách ID cần fetch (chưa có trong details map và allUsers)
    const idsToFetch = Array.from(persistedPartnerIds).filter(id =>
        !persistedPartnerDetails.has(id) && !allUsers.some(u => u.id === id)
    );

    if (idsToFetch.length === 0) return; // Không có gì để fetch

    console.log("Fetching details for persisted partner IDs:", idsToFetch);
    try {
        // Sử dụng Promise.allSettled để fetch song song và xử lý lỗi từng cái
        // Lưu ý: getUserById hiện tại không yêu cầu token theo code bạn cung cấp
        const promises = idsToFetch.map(id => getUserById(id));
        const results = await Promise.allSettled(promises);

        // Cập nhật state details map
        setPersistedPartnerDetails(prevMap => {
            const newMap = new Map(prevMap);
            results.forEach((result, index) => {
                if (result.status === 'fulfilled' && result.value) {
                    newMap.set(idsToFetch[index], result.value);
                } else if (result.status === 'rejected') {
                    console.error(`Failed to fetch persisted partner ${idsToFetch[index]}:`, result.reason);
                    // Có thể xóa ID này khỏi persistedPartnerIds nếu fetch lỗi? (Tùy chọn)
                }
            });
            return newMap;
        });
    } catch (error) {
        console.error("Error fetching details for persisted partners:", error);
        toast.error("Lỗi tải thông tin người dùng đã lưu.");
    }
  }, [persistedPartnerIds, allUsers, persistedPartnerDetails]); // Phụ thuộc vào các state này

  // Chạy fetch details khi danh sách persistedPartnerIds thay đổi
  useEffect(() => {
    fetchPersistedPartnerDetails();
  }, [fetchPersistedPartnerDetails]); // Chạy khi hàm thay đổi (ít khi)

  // --- Xử lý tìm kiếm người dùng ---
  const handleSearch = useCallback(async (query: string) => {
    setSearchTerm(query); // Cập nhật state search term
    if (!query.trim()) { // Nếu ô tìm kiếm trống
      setSearchResults([]); // Xóa kết quả tìm kiếm
      setIsSearching(false);
      return; // `displayedContacts` sẽ tự động tính lại ở useMemo
    }

    setIsSearching(true); // Bắt đầu trạng thái loading tìm kiếm
    setSearchResults([]); // Xóa kết quả cũ

    try {
        let foundUsers: User[] = [];
        const lowerCaseQuery = query.toLowerCase();

        if (currentUser?.role === 'admin') {
            // Admin: Lọc client-side từ `allUsers`
            foundUsers = allUsers.filter(u =>
                u.id !== currentUser.id &&
                (u.username.toLowerCase().includes(lowerCaseQuery) ||
                 u.email.toLowerCase().includes(lowerCaseQuery) ||
                 `${u.firstName} ${u.lastName}`.toLowerCase().includes(lowerCaseQuery)
                )
            );
        } else if (currentUser?.role === 'officer') {
            // Officer:
            // 1. Tìm Admin (client-side)
            const foundAdmins = allUsers.filter(u =>
                u.role === 'admin' &&
                u.id !== currentUser.id &&
                 (`${u.firstName} ${u.lastName}`.toLowerCase().includes(lowerCaseQuery) ||
                  u.email.toLowerCase().includes(lowerCaseQuery))
            );

            // 2. Tìm Officer khác *chỉ bằng email* (dùng API `searchUser`)
            let foundOfficersByEmail: User[] = [];
            const isEmailQuery = query.includes('@') && query.includes('.');
            if (isEmailQuery) {
                 const searchParams = { options: { filters: { email: query }, includeDeleted: false, limit: 10, offset: 0, sortBy: 'username', sortDesc: false } };
                try {
                    const searchResult = await searchUser(searchParams);
                    foundOfficersByEmail = (searchResult.users || []).filter(u =>
                        u.role === 'officer' && u.id !== currentUser.id
                    );
                } catch (searchError) {
                     console.error("Error searching officers by email:", searchError);
                     toast.error("Lỗi tìm kiếm Officer qua email.");
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
        setSearchResults(foundUsers); // Lưu kết quả tìm kiếm vào state
    } catch (error) {
        console.error("Error during user search:", error);
        toast.error("Lỗi tìm kiếm người dùng.");
    } finally {
        setIsSearching(false); // Kết thúc trạng thái loading tìm kiếm
    }
  }, [currentUser, allUsers]); // Phụ thuộc vào currentUser và allUsers

  // --- Xử lý khi chọn một người dùng từ sidebar ---
  const handleSelectUser = useCallback((user: User) => {
    setSelectedChatPartner(user); // Cập nhật người đang được chọn

    // Logic "ghim" user: Nếu người dùng hiện tại là officer và người được chọn không phải admin
    if (currentUser && currentUser.role === 'officer' && user.role !== 'admin') {
      // Thêm ID vào Set persistedPartnerIds nếu chưa có
      setPersistedPartnerIds(prev => {
        if (prev.has(user.id)) return prev; // Không thay đổi nếu đã tồn tại
        const newSet = new Set(prev);
        newSet.add(user.id);
        console.log("Persisted partner ID added:", user.id);
        return newSet;
      });
      // Thêm/Cập nhật thông tin chi tiết vào Map persistedPartnerDetails
      if (!persistedPartnerDetails.has(user.id)) {
          setPersistedPartnerDetails(prevMap => new Map(prevMap).set(user.id, user));
      }
    }

    // Tùy chọn: Xóa nội dung tìm kiếm sau khi chọn
    setSearchTerm('');
    setSearchResults([]);
    setIsSearching(false);

  }, [currentUser, persistedPartnerDetails]); // Phụ thuộc currentUser và persistedPartnerDetails

  // --- Tính toán danh sách contacts cuối cùng để hiển thị (dùng useMemo) ---
  const displayedContacts = useMemo(() => {
    if (!currentUser) return []; // Trả về mảng rỗng nếu chưa có currentUser

    // 1. Nếu đang tìm kiếm (searchTerm không rỗng), hiển thị kết quả tìm kiếm
    if (searchTerm.trim()) {
      return searchResults;
    }

    // 2. Nếu không tìm kiếm:
    let baseContacts: User[] = [];
    // Xác định danh sách cơ sở dựa trên vai trò
    if (currentUser.role === 'admin') {
      // Admin: Lấy tất cả người dùng khác
      baseContacts = allUsers.filter(u => u.id !== currentUser.id);
    } else {
      // Officer: Lấy tất cả người dùng có vai trò 'admin'
      baseContacts = allUsers.filter(u => u.role === 'admin' && u.id !== currentUser.id);
    }

    // Lấy danh sách các partner đã được "ghim" (persisted)
    const persistedPartners: User[] = Array.from(persistedPartnerIds)
      .map(id => persistedPartnerDetails.get(id) || allUsers.find(u => u.id === id)) // Ưu tiên details đã fetch, fallback về allUsers
      .filter((user): user is User => !!user && user.id !== currentUser.id); // Lọc bỏ null/undefined và chính mình

    // Kết hợp danh sách cơ sở và danh sách persisted, loại bỏ trùng lặp
    const combined = [...baseContacts, ...persistedPartners];
    const uniqueMap = new Map<string, User>();
    combined.forEach(user => {
      if (!uniqueMap.has(user.id)) {
        uniqueMap.set(user.id, user);
      }
    });

    // Sắp xếp (ví dụ theo tên) - tùy chọn
    const sortedContacts = Array.from(uniqueMap.values()).sort((a, b) =>
        `${a.firstName} ${a.lastName}`.localeCompare(`${b.firstName} ${b.lastName}`)
    );

    return sortedContacts;

  }, [currentUser, searchTerm, searchResults, allUsers, persistedPartnerIds, persistedPartnerDetails]); // Các dependencies

  // --- Render ---
  // Hiển thị loader nếu đang xác thực
  if (isAuthLoading) {
    return <PageLoader message="Đang tải thông tin xác thực..." />;
  }

  // Hiển thị thông báo nếu chưa đăng nhập
  if (!isLoggedIn || !currentUser) {
      return (
          <div className="flex items-center justify-center h-full text-center p-4">
              Vui lòng đăng nhập để sử dụng tính năng chat.
          </div>
      );
  }

  // Render layout chính
  return (
    <div className="flex h-full border-t"> {/* Đảm bảo layout linh hoạt và có border */}
      <ChatSidebar
        contacts={displayedContacts} // Truyền danh sách đã tính toán
        currentUser={currentUser}
        onSelectUser={handleSelectUser} // Hàm callback khi chọn user
        isLoading={isLoadingUsers || isSearching} // Trạng thái loading
        selectedUserId={selectedChatPartner?.id} // ID của người đang được chọn
        onSearch={handleSearch} // Hàm callback khi tìm kiếm
        searchTerm={searchTerm} // Giá trị hiện tại của ô tìm kiếm
      />
      {/* Render ChatWindow, đảm bảo currentUser tồn tại */}
      {currentUser && (
          <ChatWindow
            currentUser={currentUser}
            chatPartner={selectedChatPartner}
            token={token ?? null} // Chuyển undefined thành null
            key={selectedChatPartner?.id} // Rerender khi chọn người khác
          />
       )}
       {/* Fallback nếu currentUser chưa kịp load */}
       {!currentUser && !isAuthLoading && (
           <div className="flex-1 flex items-center justify-center bg-white h-full">
               <p className="text-gray-500">Đang tải dữ liệu người dùng...</p>
           </div>
       )}
    </div>
  );
};

export default ChatLayout;