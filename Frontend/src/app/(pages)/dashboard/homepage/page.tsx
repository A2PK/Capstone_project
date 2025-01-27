import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MapPin } from "lucide-react";
import Link from "next/link";

const articles = [
  {
    title: "Bản tin khẩn chất lượng nước sông Tiền !!",
    date: "4 Nov 2024",
    description:
      "Dữ liệu quan trắc được từ trạm sông Tiền cho thấy rằng chất lượng nước nơi này đang có sự ô nhiễm nhẹ...",
    imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
  },
  {
    title: "Báo cáo hàng tuần về chất lượng nước sông Hậu!!",
    date: "4 Nov 2024",
    description:
      "Dữ liệu quan trắc được từ trạm sông Hồng cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
    imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
  },
  {
    title: "Báo cáo hàng tuần về chất lượng nước sông Cần!!",
    date: "4 Nov 2024",
    description:
      "Dữ liệu quan trắc được từ trạm sông Cần cho thấy rằng chất lượng nước nơi này vẫn đang bình thường...",
    imageUrl: "https://i.ibb.co/fM2bq9V/z6267544864718-caf04a0ab4ae130a41a82852880185d9.jpg",
  },
];

export default function Homepage() {
  return (
    <div className="space-y-12">
      <section>
        <h1 className="text-4xl font-bold text-orange-600 mb-8">Bản tin</h1>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {articles.map((article, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <img
                src={article.imageUrl}
                alt={article.title}
                className="w-full h-64 object-cover rounded-t-md"
              />
              <CardContent>
                <CardHeader>
                  <CardTitle className="text-lg font-semibold">{article.title}</CardTitle>
                </CardHeader>
                <p className="text-gray-600 text-sm mb-2">{article.date}</p>
                <p className="text-gray-700 text-sm">{article.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="flex justify-center mt-6">
        <Link href="/dashboard/newspaper">
          <Button className="px-6 py-2 text-white bg-black hover:bg-gray-800">Xem thêm</Button>
        </Link>
        </div>
      </section>
      <section>
  <h2 className="text-4xl font-bold text-orange-600 mb-8">Bản đồ địa lý</h2>
  <div className="relative w-full h-[500px] border border-gray-300 rounded-md overflow-hidden">
    <iframe
      src="https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d25173.56829474514!2d106.3820079!3d10.5364105!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e0!3m2!1sen!2s!4v1688659195067!5m2!1sen!2s"
      width="100%"
      height="100%"
      style={{ border: 0 }}
      allowFullScreen
      loading="lazy"
    ></iframe>
  </div>
</section>
    </div>
  );
}
