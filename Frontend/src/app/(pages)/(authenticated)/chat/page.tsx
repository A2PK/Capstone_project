
import ChatLayout from '@/components/chat/ChatLayout';
import { Toaster } from 'react-hot-toast';

export default function ChatPage() {
  return (
    <div className="h-screen w-full flex flex-col">
       <Toaster position="top-right" />
       {/* Có thể thêm Header chung của ứng dụng ở đây nếu cần */}
       {/* <AppHeader /> */}
       <ChatLayout />
    </div>
  );
}