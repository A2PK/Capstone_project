
import ChatLayout from '@/components/chat/ChatLayout';
import { Toaster } from 'react-hot-toast';

export default function ChatPage() {
  return (
    <div className="h-screen w-full flex flex-col">
       <Toaster position="top-right" />
       <ChatLayout />
    </div>
  );
}