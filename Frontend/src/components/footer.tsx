"use client";

import Image from "next/image";
import Link from "next/link";
import logo from "/public/atlogo.png";

const AppFooter = () => {
  return (
    <footer className="bg-gray-800 text-gray-200">
      <div className="container mx-auto px-4 py-10 grid gap-8 md:grid-cols-3">
        {/* Logo and Application Name */}
        <div className="flex flex-col items-center text-center md:items-start md:text-left">
          <div className="flex justify-center items-center">
            <Image src={logo} alt="Water Quality Monitoring Logo" width={150} />
          </div>
          <h2 className="text-xl font-semibold mt-4">
            Quan trắc chất lượng nước
          </h2>
          <p className="text-sm">
          Nâng cao nghiên cứu môi trường và giải pháp dựa trên dữ liệu cho quản lý chất lượng nước.
          </p>
        </div>

        {/* Quick Links */}
        <div className="flex flex-col gap-4">
          <h2 className="text-lg font-semibold">Đường Links</h2>
          <nav className="flex flex-col gap-2">
            <Link href="/" className="hover:underline">
              Trang chủ
            </Link>
            <Link href="/about" className="hover:underline">
              Lược sử về chúng tôi
            </Link>
            <Link href="/features" className="hover:underline">
              Đặc điểm
            </Link>
            <Link href="/contact" className="hover:underline">
              Liện hệ với
            </Link>
          </nav>
        </div>

        {/* Contact Information */}
        <div className="flex flex-col gap-4">
          <h2 className="text-lg font-semibold">Liên hệ tại</h2>
          <address className="not-italic text-sm">
            12 Lê Lợi, Quận 1, TPHCM
            <br />
            Phone: +1 (800) 555-1234
            <br />
            Email:{" "}
            <a
              href="mailto:info@watermonitoring.com"
              className="hover:underline text-blue-400"
            >
              info@watermonitoring.com
            </a>
          </address>
        </div>
      </div>
      <div className="border-t border-gray-700 py-4">
        <p className="text-center text-xs text-gray-500">
          &copy; {new Date().getFullYear()} Water Quality Monitoring. All
          rights reserved.
        </p>
      </div>
    </footer>
  );
};

export default AppFooter;
