"use client"

import { useState } from "react"
import Link from "next/link"
import { Home, Newspaper, Monitor, Database, ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed)
  }

  return (
    <div
      className={`flex ${isCollapsed ? "w-16" : "w-64"} transition-all duration-200 ease-in-out`}
    >
      {/* Sidebar with Gentle White Background */}
      <div className={`bg-white h-screen p-4 border-r border-gray-200 ${isCollapsed ? "w-16" : "w-64"}`}>
        {/* Sidebar Header */}
        <div className="flex justify-between items-center text-gray-800">
          {/* If not collapsed, show the title */}
          {!isCollapsed && <span className="text-xl font-bold">Dashboard</span>}
          <Button
            variant="ghost"
            size="icon"
            className="text-gray-800"
            onClick={toggleSidebar}
          >
            {/* Icons change based on sidebar state */}
            {isCollapsed ? <ChevronRight /> : <ChevronLeft />}
          </Button>
        </div>

        {/* Sidebar Items */}
        <div className="mt-4 space-y-4">
          <Link
            href="/dashboard/homepage"
            className={`flex items-center text-gray-800 py-2 px-3 rounded-md hover:bg-blue-100 ${isCollapsed ? "justify-center" : "pl-2"}`}
          >
            <Home className={`mr-3 ${isCollapsed && "hidden"}`} />
            {!isCollapsed && "Trang chủ"}
          </Link>
          <Link
            href="/dashboard/newspaper"
            className={`flex items-center text-gray-800 py-2 px-3 rounded-md hover:bg-blue-100 ${isCollapsed ? "justify-center" : "pl-2"}`}
          >
            <Newspaper className={`mr-3 ${isCollapsed && "hidden"}`} />
            {!isCollapsed && "Bản tin"}
          </Link>
          <Link
            href="/dashboard/stations"
            className={`flex items-center text-gray-800 py-2 px-3 rounded-md hover:bg-blue-100 ${isCollapsed ? "justify-center" : "pl-2"}`}
          >
            <Monitor className={`mr-3 ${isCollapsed && "hidden"}`} />
            {!isCollapsed && "Trạm quan trắc"}
          </Link>
          <Link
            href="/dashboard/realtimedata"
            className={`flex items-center text-gray-800 py-2 px-3 rounded-md hover:bg-blue-100 ${isCollapsed ? "justify-center" : "pl-2"}`}
          >
            <Database className={`mr-3 ${isCollapsed && "hidden"}`} />
            {!isCollapsed && "Dữ liệu thực"}
          </Link>
        </div>
      </div>
    </div>
  )
}
