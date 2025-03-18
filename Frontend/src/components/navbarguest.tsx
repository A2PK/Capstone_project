'use client'
import Image from "next/image";
import { Button } from "@/components/ui/button";
import logo from "/public/wqm.jpg";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Bell, Mail, Menu, X } from "lucide-react";
import Link from "next/link";

export default function Navbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const router = useRouter();

  const handleLogin = () => {
    router.push("/auth/login"); // Redirect to login page
  };

  return (
    <nav className="bg-white shadow-md border-b border-gray-200">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo Section */}
          <div className="flex items-center">
            <Image
              src={logo}
              alt="Water Monitoring System Logo"
              width={90}
              height={40}
            />
          </div>

          {/* Desktop Icons + Login */}
          <div className="hidden sm:flex items-center space-x-4">
            {/* <Button variant="ghost" size="icon">
              <Bell className="h-5 w-5 text-gray-500" />
            </Button>
            <Button variant="ghost" size="icon">
              <Mail className="h-5 w-5 text-gray-500" />
            </Button> */}
            <Button
              onClick={handleLogin}
              className="bg-blue-600 text-white hover:bg-blue-700"
            >
              Login
            </Button>
          </div>

          {/* Mobile Menu Toggle */}
          <div className="flex sm:hidden">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              {isMobileMenuOpen ? (
                <X className="block h-6 w-6" aria-hidden="true" />
              ) : (
                <Menu className="block h-6 w-6" aria-hidden="true" />
              )}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="sm:hidden bg-white shadow-md absolute top-16 right-0 w-full max-w-xs z-50 p-4">
            <div className="flex flex-col space-y-4 py-2 px-4">
              {/* <Button variant="ghost" className="flex items-center justify-start">
                <Bell className="h-5 w-5 text-gray-500 mr-2" />
                Notifications
              </Button>

              <Button variant="ghost" className="flex items-center justify-start">
                <Mail className="h-5 w-5 text-gray-500 mr-2" />
                Messages
              </Button> */}

              <Button
                onClick={handleLogin}
                className="bg-blue-600 text-white hover:bg-blue-700 w-full"
              >
                Login
              </Button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
