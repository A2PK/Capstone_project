'use client'
import Image from "next/image";
import { redirect } from "next/navigation";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import logo from "/public/wqm.jpg";
import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { LogOut, Bell, Mail, Menu, X, User } from "lucide-react";
import Link from "next/link";

export default function Navbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const router = useRouter();
  let isLoggedIn = true;
  const handleLogout = () => {
    router.push("/auth/login"); // Redirect to the login page
  };
  // const onLogoutSubmit = async (_: FormData) => {
  //   "use server";
  //   redirect("/auth/login");
  // };

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
            {/* <span className="ml-2 text-xl font-bold text-blue-600">
              WQM
            </span> */}
          </div>

          {/* Desktop Icons and User Section */}
          <div className="hidden sm:flex items-center space-x-4">
            <Button variant="ghost" size="icon">
              <Bell className="h-5 w-5 text-gray-500" />
            </Button>
            <Button variant="ghost" size="icon">
              <Mail className="h-5 w-5 text-gray-500" />
            </Button>
            {isLoggedIn && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    className="relative rounded-full bg-white p-1 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  >
                    <Avatar>
                      <AvatarImage
                        src="https://github.com/shadcn.png"
                        alt="User Avatar"
                      />
                      <AvatarFallback>U</AvatarFallback>
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem>
                    <Link
                      href="/dashboard/profile"
                      className="flex items-center w-full px-4 py-2 text-gray-700 hover:bg-blue-100 hover:text-blue-600 rounded-md transition-all duration-200 ease-in-out"
                    >
                      <User className="mr-2 h-4 w-4" />
                      My Profile
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    {/* <form action={onLogoutSubmit}>
                      <Button type="submit" className="flex items-center">
                        <LogOut className="mr-2 h-4 w-4" />
                        Log out
                      </Button>
                    </form> */}
                    <Button
                      type="button" // Changed to "button" as "submit" is unnecessary here
                      onClick={handleLogout}
                      className="flex items-center w-full px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 transition-all duration-200 ease-in-out"
                    >
                      <LogOut className="mr-2 h-4 w-4" />
                      Log out
                    </Button>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>

          {/* Mobile Menu Button */}
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
        {isMobileMenuOpen && (
          <div className="sm:hidden bg-white shadow-md absolute top-16 right-0 w-full max-w-xs z-50 p-4">
            <div className="flex space-x-4 py-2 px-4 items-center">

              <Button variant="ghost" className="flex items-center justify-center">
                <Bell className="h-5 w-5 text-gray-500" />
              </Button>

              <Button variant="ghost" className="flex items-center justify-center">
                <Mail className="h-5 w-5 text-gray-500" />
              </Button>

              {isLoggedIn && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" className="flex items-center justify-center">
                      <Avatar className="mr-2">
                        <AvatarImage
                          src="https://github.com/shadcn.png"
                          alt="User Avatar"
                        />
                        <AvatarFallback>U</AvatarFallback>
                      </Avatar>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem>
                      <Link href="/dashboard/profile" className="flex items-center w-full">
                        <User className="mr-2 h-4 w-4" />
                        My Profile
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                      <Button
                        type="button" // Changed to "button" as "submit" is unnecessary here
                        onClick={handleLogout}
                        className="flex items-center w-full px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 transition-all duration-200 ease-in-out"
                      >
                        <LogOut className="mr-2 h-4 w-4" />
                        Log out
                      </Button>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>
          </div>
        )}


      </div>
    </nav>
  );
}