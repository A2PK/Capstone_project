"use client";

import Image from "next/image";
import Link from "next/link";
import logo from "/public/atlogo.png"; // Đảm bảo đường dẫn logo đúng
import { MapPin, Phone, Mail, Facebook, Twitter, Linkedin, Github } from 'lucide-react'; // Import các icon cần thiết

const AppFooter = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-slate-900 text-slate-300 pt-12 pb-8 px-4"> {/* Màu nền tối hơn, padding lớn hơn */}
      <div className="container mx-auto grid gap-10 md:gap-12 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 mb-8"> {/* Responsive grid, 4 cột trên màn lớn */}

        {/* 1. Logo và Mô tả */}
        <div className="flex flex-col items-center sm:items-start text-center sm:text-left">
          <Link href="/" className="inline-block mb-4">
            {/* Giảm kích thước logo một chút cho phù hợp với footer */}
            <Image src={logo} alt="Water Quality Monitoring Logo" width={120} quality={90} />
          </Link>
          <p className="text-sm leading-relaxed">
            Nâng cao nghiên cứu môi trường và giải pháp dựa trên dữ liệu cho quản lý chất lượng nước.
          </p>
        </div>

        {/* 2. Liên kết nhanh */}
        <div className="flex flex-col items-center sm:items-start gap-3">
          <h3 className="font-semibold text-white uppercase tracking-wider text-sm mb-2"> {/* Tiêu đề cột rõ ràng hơn */}
            Khám phá
          </h3>
          <nav className="flex flex-col items-center sm:items-start gap-2">
            <Link href="/" className="hover:text-white transition-colors duration-200">Trang chủ</Link>
            <Link href="/" className="hover:text-white transition-colors duration-200">Về chúng tôi</Link>
            <Link href="/" className="hover:text-white transition-colors duration-200">Tính năng</Link>
            <Link href="/" className="hover:text-white transition-colors duration-200">Liên hệ</Link>
          </nav>
        </div>

        {/* 3. Thông tin Liên hệ */}
        <div className="flex flex-col items-center sm:items-start gap-3">
          <h3 className="font-semibold text-white uppercase tracking-wider text-sm mb-2">
            Liên hệ
          </h3>
          <address className="not-italic text-sm flex flex-col items-center sm:items-start gap-3">
            <div className="flex items-center gap-2">
              <MapPin size={16} className="text-slate-400 flex-shrink-0" />
              <span>12 Lê Lợi, Quận 1, TPHCM</span>
            </div>
            <div className="flex items-center gap-2">
              <Phone size={16} className="text-slate-400 flex-shrink-0" />
              {/* Thay đổi số điện thoại nếu cần */}
              <a href="tel:+18005551234" className="hover:text-white transition-colors duration-200">+1 (800) 555-1234</a>
            </div>
            <div className="flex items-center gap-2">
              <Mail size={16} className="text-slate-400 flex-shrink-0" />
              <a
                href="mailto:info@watermonitoring.com"
                className="hover:text-white transition-colors duration-200" // Thống nhất màu hover
              >
                info@watermonitoring.com
              </a>
            </div>
          </address>
        </div>

        {/* 4. Mạng xã hội */}
        <div className="flex flex-col items-center sm:items-start gap-3">
           <h3 className="font-semibold text-white uppercase tracking-wider text-sm mb-2">
            Theo dõi chúng tôi
          </h3>
          <div className="flex gap-4">
             {/* Thay đổi href thành link MXH thực tế của bạn */}
            <Link href="https://facebook.com" target="_blank" rel="noopener noreferrer" aria-label="Facebook" className="text-slate-400 hover:text-white transition-colors duration-200">
              <Facebook size={20} />
            </Link>
            <Link href="https://twitter.com" target="_blank" rel="noopener noreferrer" aria-label="Twitter" className="text-slate-400 hover:text-white transition-colors duration-200">
              <Twitter size={20} />
            </Link>
            <Link href="https://linkedin.com" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className="text-slate-400 hover:text-white transition-colors duration-200">
              <Linkedin size={20} />
            </Link>
             <Link href="https://github.com" target="_blank" rel="noopener noreferrer" aria-label="Github" className="text-slate-400 hover:text-white transition-colors duration-200">
              <Github size={20} />
            </Link>
          </div>
        </div>

      </div>

      {/* Dòng Copyright */}
      <div className="border-t border-slate-800 pt-6 mt-8"> {/* Đường kẻ phân cách tinh tế hơn */}
        <p className="text-center text-xs text-slate-500"> {/* Chữ copyright rõ hơn chút */}
          &copy; {currentYear} Water Quality Monitoring. Đã đăng ký Bản quyền.
        </p>
      </div>
    </footer>
  );
};

export default AppFooter;