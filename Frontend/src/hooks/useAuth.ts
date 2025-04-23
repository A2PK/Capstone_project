// src/hooks/useAuth.ts
import { useState, useEffect } from 'react';
import axios from 'axios';
import type { User } from '@/types/user'; // Đảm bảo đường dẫn type User đúng

interface SessionData {
  isLoggedIn: boolean;
  user: User | null;
  token: string | null;
}

export function useAuth() {
  const [session, setSession] = useState<SessionData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSession = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await axios.get<SessionData>('/api/auth/session');
        setSession(response.data);
        // console.log("Auth Hook: Session data fetched", response.data);
      } catch (err) {
        console.error("Auth Hook: Failed to fetch session:", err);
        setError('Failed to load session data.');
        setSession({ isLoggedIn: false, user: null, token: null }); // Set default state on error
      } finally {
        setIsLoading(false);
      }
    };

    fetchSession();
  }, []);

  return {
    session,
    isLoading,
    error,
    token: session?.token,
    user: session?.user,
    isLoggedIn: session?.isLoggedIn ?? false, // Default to false if session is null
  };
}