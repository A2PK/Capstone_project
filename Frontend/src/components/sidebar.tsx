import Link from "next/link"
import { Home, Newspaper, Monitor, Database, Inbox } from "lucide-react"
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"

export default function Sidebar() {
  const navItems = [
    { href: "/dashboardofficer/homepage", icon: <Home />, label: "Trang chủ" },
    { href: "/dashboardofficer/newspaper", icon: <Newspaper />, label: "Bản tin" },
    { href: "/dashboardofficer/stations", icon: <Monitor />, label: "Trạm quan trắc" },
    { href: "/dashboardofficer/realtimedata", icon: <Database />, label: "Dữ liệu thực" },
    { href: "/dashboardofficer/request", icon: <Inbox />, label: "Yêu cầu" },
  ]

  return (
    <div className="flex w-16 transition-all duration-200 ease-in-out">
      <div className="bg-white h-screen p-4 border-r border-gray-200 w-16 flex flex-col items-center space-y-6">
        {navItems.map((item, index) => (
          <Tooltip key={index} delayDuration={0}>
            <TooltipTrigger asChild>
              <Link href={item.href}>
                <div className="text-gray-800 hover:text-blue-600 cursor-pointer">
                  {item.icon}
                </div>
              </Link>
            </TooltipTrigger>
            <TooltipContent side="right" className="z-[9999]">
              {item.label}
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </div>
  )
}
