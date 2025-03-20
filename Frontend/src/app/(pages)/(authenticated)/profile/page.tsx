"use client"
import PageLoader from "@/components/pageloader"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { UserCircle, Mail, MapPin, Briefcase, Calendar, LogOut } from "lucide-react"
import Link from "next/link"
import { useEffect, useState } from "react"

export default function ProfilePage() {
  const [isLoading, setIsLoading] = useState(true);
      useEffect(() => {
              // Simulate loading delay (e.g., fetching data)
              const timeout = setTimeout(() => {
                setIsLoading(false);
              }, 1000); // 1.5s delay
              return () => clearTimeout(timeout);
            }, []);
          if (isLoading) return <PageLoader message="Đang tải trang thông tin cá nhân..." />;
  return (
    <div className="container mx-auto py-10">
      <div className="flex flex-col gap-8 md:flex-row">
        {/* Profile Sidebar */}
        <div className="md:w-1/3">
          <Card>
            <CardHeader className="flex flex-col items-center space-y-2">
              <div className="relative">
                <Avatar className="h-24 w-24">
                  <AvatarImage src="https://github.com/shadcn.png" alt="Profile picture" />
                  <AvatarFallback>
                    <UserCircle className="h-24 w-24" />
                  </AvatarFallback>
                </Avatar>
                <Link href={"profile/edit"}>
                
                <Button size="sm" variant="outline" className="absolute -bottom-2 -right-2 rounded-full">
                  Thay đổi
                </Button></Link>
              </div>
              <div className="space-y-1 text-center">
                <h2 className="text-2xl font-bold">Nguyen Tien</h2>
                <p className="text-sm text-muted-foreground">@vina</p>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">user@gmail.com</span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Vịnh thừa thiên</span>
                </div>
                <div className="flex items-center gap-2">
                  <Briefcase className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Quan chức cấp thấp</span>
                </div>
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Tham gia ngày 22, tháng 3, 2025</span>
                </div>
              </div>
            </CardContent>
            <CardFooter>
              {/* <Button variant="outline" className="w-full gap-2">
                <LogOut className="h-4 w-4" />
                Sign Out
              </Button> */}
            </CardFooter>
          </Card>
        </div>

        {/* Profile Content */}
        <div className="flex-1">
          <Tabs defaultValue="profile" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="profile">Thông tin cá nhân</TabsTrigger>
              <TabsTrigger value="account">Tài khoản</TabsTrigger>
              <TabsTrigger value="security">Bảo mật</TabsTrigger>
            </TabsList>
            <TabsContent value="profile" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Thông tin cá nhân</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="firstName">Tên</Label>
                      <Input id="firstName" defaultValue="Tien" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="lastName">Họ</Label>
                      <Input id="lastName" defaultValue="Nguyen" />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="username">Username</Label>
                    <Input id="username" defaultValue="vina" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="bio">Bio</Label>
                    <Input id="bio" defaultValue="Admin" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="location">Location</Label>
                    <Input id="location" defaultValue="Thừa Thiên Huế" />
                  </div>
                </CardContent>
                <CardFooter className="flex justify-end">
                  <Button>Lưu thay đổi</Button>
                </CardFooter>
              </Card>
            </TabsContent>
            <TabsContent value="account" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Cập nhật tài khoản</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input id="email" defaultValue="" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="phone">Số điện thoại</Label>
                    <Input id="phone" defaultValue="" />
                  </div>
                </CardContent>
                <CardFooter className="flex justify-end">
                  <Button>Lưu thay đổi</Button>
                </CardFooter>
              </Card>
            </TabsContent>
            <TabsContent value="security" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Bảo mật mật khẩu</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="currentPassword">Mật khẩu hiện tại</Label>
                    <Input id="currentPassword" type="password" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="newPassword">Mật khẩu mới</Label>
                    <Input id="newPassword" type="password" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword">Nhập lại mật khẩu</Label>
                    <Input id="confirmPassword" type="password" />
                  </div>
                  
                </CardContent>
                <CardFooter className="flex justify-end">
                  <Button>Cập nhật mật khẩu</Button>
                </CardFooter>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}

