"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/hooks/use-toast"
import { UserCircle, Upload } from "lucide-react"
import { updateUser } from "@/lib/user"
import { User } from "@/types/user"
export default function EditProfilePage() {
  
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [profileImage, setProfileImage] = useState<string | null>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setProfileImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setIsLoading(true)

    // Simulate API call
    setTimeout(() => {
      setIsLoading(false)
      toast({
        variant: "success",
        title: "Cập nhật thành công",
        description: "Thông tin cá nhân của bạn đã được thay đổi",
      })
      router.push("/profile")
    }, 1500)
  }

  return (
    <div className="container mx-auto py-10">
      <Card>
        <CardHeader>
          <CardTitle>Thay đổi thông tin cá nhân</CardTitle>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-6">
            <div className="flex flex-col items-center space-y-4">
              <Avatar className="h-24 w-24">
                <AvatarImage src={profileImage || "https://github.com/shadcn.png"} alt="Profile picture" />
                <AvatarFallback>
                  <UserCircle className="h-24 w-24" />
                </AvatarFallback>
              </Avatar>
              <div className="flex items-center gap-2">
                <Label htmlFor="picture" className="cursor-pointer">
                  <div className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground">
                    <Upload className="h-4 w-4" />
                    Tải ảnh Avatar
                  </div>
                  <Input id="picture" type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                </Label>
                {profileImage && (
                  <Button type="button" variant="outline" onClick={() => setProfileImage(null)}>
                    Huỷ bỏ
                  </Button>
                )}
              </div>
            </div>

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
              <p className="text-sm text-muted-foreground">
                Nickname hay biệt danh của bạn
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" defaultValue="user@gmail.com" />
            </div>

            <div className="space-y-2">
              <Label htmlFor="bio">Tiểu sử</Label>
              <Textarea
                id="bio"
                placeholder="Tell us a little about yourself"
                defaultValue="Quan chức cấp thấp"
                className="min-h-[120px] resize-y"
              />
              <p className="text-sm text-muted-foreground">Quan chức cấp thấp</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="location">Địa điểm sống</Label>
              <Input id="location" defaultValue="Vịnh thừa thiên" />
            </div>

          </CardContent>
          <CardFooter className="flex justify-between">
            <Button type="button" variant="outline" onClick={() => router.push("/profile")}>
              Huỷ bỏ
            </Button>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? "Đang lưu..." : "Lưu thay đổi"}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  )
}

