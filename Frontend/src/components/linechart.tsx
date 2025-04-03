"use client";
import dynamic from "next/dynamic";
import { useMemo, useState, useEffect, useCallback } from "react";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button"; // Giả sử bạn có Button từ shadcn/ui
import { Maximize, Minimize, LineChart, BarChart, AreaChart } from 'lucide-react'; // Icons ví dụ
import type { ApexOptions } from "apexcharts";

// --- Dynamic Import Chart ---
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false, loading: () => <p>Loading chart...</p> });

// --- Props Interface ---
interface ChartlineProps {
  trend: number[];
  prediction: number[];
  title: string;
}

// --- Chart Type Definition ---
type ChartType = "line" | "area" | "bar";

// --- Component Implementation ---
const Chartline: React.FC<ChartlineProps> = ({ trend, prediction, title }) => {
  const [predictMode, setPredictMode] = useState(false);
  const [predictRange, setPredictRange] = useState<number>(1);
  const [isClient, setIsClient] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [chartType, setChartType] = useState<ChartType>("line"); // Mặc định là biểu đồ đường

  useEffect(() => {
    setIsClient(true); // Mark component as client-rendered

    // --- Fullscreen Exit Handler ---
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    document.addEventListener('keydown', handleEsc);

    // Cleanup listener on component unmount
    return () => {
      document.removeEventListener('keydown', handleEsc);
    };
  }, [isFullscreen]); // Re-run effect if isFullscreen changes

  // --- Data Preparation ---
  const historicalData = useMemo(() => ({
    name: "Historical",
    data: trend.map((value, index) => ({
      x: new Date(2024, 0, index + 1).getTime(), // Giữ nguyên logic tạo date
      y: value,
    })),
  }), [trend]);

  const predictionData = useMemo(() => ({
    name: "Prediction",
    data: prediction.map((value, index) => ({
      // Đảm bảo ngày dự đoán bắt đầu sau ngày lịch sử cuối cùng
      x: new Date(2024, 0, trend.length + index + 1).getTime(),
      y: value,
    })),
  }), [prediction, trend.length]);

  const chartData = useMemo(() => {
    const data: ApexAxisChartSeries = [{ ...historicalData }]; // Luôn bắt đầu với historical (bản sao nông)

    if (predictMode && predictionData?.data?.length > 0) {
      // Lấy các điểm dữ liệu dự đoán gốc để xử lý
      const rawPredictionPoints = predictionData.data.slice(0, predictRange);

      let processedPredictionData: ApexAxisChartSeries[0]['data'] = []; // Khởi tạo mảng data cho series prediction

      // Logic thêm điểm nối và chỉnh màu marker của nó
      if (chartType !== 'bar' && historicalData?.data?.length > 0) {
          const connectionPoint = historicalData.data[historicalData.data.length - 1];
          if (connectionPoint) {
              // --- BẮT ĐẦU CHỈNH MÀU MARKER ---
              processedPredictionData = [
                  {
                      ...connectionPoint, // Sao chép x, y từ điểm historical cuối
                      // Ghi đè màu marker CHO RIÊNG ĐIỂM NỐI NÀY trong series Prediction
                      // Sử dụng màu của series đầu tiên (Historical - màu xanh)
                      // Lưu ý: Đảm bảo 'var(--color-primary)' được định nghĩa đúng trong CSS của bạn
                      fillColor: "var(--color-primary)", // Màu nền của marker
                      strokeColor: "var(--color-primary)", // Màu viền của marker (nếu có)
                  },
                  // Thêm phần còn lại của các điểm dự đoán gốc
                  ...rawPredictionPoints
              ];
              // --- KẾT THÚC CHỈNH MÀU MARKER ---
          } else {
              // Nếu không có điểm nối, chỉ dùng các điểm dự đoán gốc
              processedPredictionData = rawPredictionPoints;
          }
      } else {
          // Đối với biểu đồ cột hoặc không có dữ liệu historical, chỉ dùng các điểm dự đoán gốc
          processedPredictionData = rawPredictionPoints;
      }


      // Chỉ thêm series Prediction nếu thực sự có dữ liệu sau khi xử lý
      if (processedPredictionData && processedPredictionData.length > 0) {
        data.push({
            name: "Prediction", // Giữ tên series gốc
            data: processedPredictionData,
        });
      }
    }

    return data;
  }, [predictMode, predictRange, historicalData, predictionData, chartType]);

  // --- Toggle Fullscreen ---
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((prev) => !prev);
  }, []);

  // --- Chart Options ---
// --- Chart Options ---
const chartOptions = useMemo((): ApexOptions => {
  let gridPaddingRight = 15;
    let markersSize = 0;

    if (chartType === 'line' || chartType === 'area') {
      gridPaddingRight = 40;
      markersSize = 3;
    }

    // --- Cấu hình fill dựa trên loại biểu đồ ---
    let finalFillOptions: ApexFill = {}; // Khởi tạo rỗng hoặc mặc định cho line

    if (chartType === 'area') {
        // --- THỬ LẠI VỚI GRADIENT, ĐIỀU CHỈNH OPACITY TRONG GRADIENT ---
        finalFillOptions = {
            type: 'gradient',
            gradient: {
                // shade: 'light', // Bạn có thể thử thêm 'light' shade
                shadeIntensity: 0.2, // Giảm cường độ nếu dùng shade, 0=không shade
                inverseColors: false,
                // --- ĐẶT GIÁ TRỊ OPACITY THẤP ĐỂ MÀU NHẠT ---
                // Bắt đầu từ mờ hơn và kết thúc rất mờ
                opacityFrom: predictMode ? 0.35 : 0.2,  // Ví dụ: Bắt đầu hơi rõ hơn (35% / 20% opacity)
                opacityTo: predictMode ? 0.3 : 0.15,    // Ví dụ: Kết thúc rất nhạt (10% / 5% opacity)
                // Bạn có thể điều chỉnh các giá trị 0.35, 0.2, 0.1, 0.05 này để đạt độ nhạt mong muốn
                stops: [0, 100] // Áp dụng gradient trên toàn bộ vùng
            }
            // Khi dùng gradient.opacityFrom/To, thường không cần đặt fill.opacity riêng nữa
        };
    } else if (chartType === 'bar') {
        finalFillOptions = { type: 'solid', opacity: 1 }; // Bar vẫn fill đặc
    } else { // line chart
        finalFillOptions = { type: 'solid', opacity: 1 }; // Line không cần fill area
    }
  // Nếu chartType là 'line', finalFillOptions vẫn là { type: 'solid', opacity: 1 } như mặc định ban đầu.

  const options: ApexOptions = {
    chart: {
      fontFamily: "inherit",
      foreColor: "#adb0bb",
      offsetX: 0,
      offsetY: 10,
      animations: { speed: 500 },
      toolbar: {
        show: true,
        tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true },
        autoSelected: 'zoom'
      },
      zoom: { enabled: true, type: 'x', autoScaleYaxis: true },
    },
    colors: ["var(--color-primary)", "#a6a6a6"],
    dataLabels: { enabled: false },
    grid: {
      show: true,
      strokeDashArray: 3,
      borderColor: "#90A4AE50",
      padding: {
        right: gridPaddingRight,
        left: 15
      }
    },
    stroke: {
      curve: "smooth",
      width: chartType === 'bar' ? 0 : 2,
      dashArray: (chartType !== 'bar' && chartData.length > 1) ? [0, 8] : 0,
    },
    xaxis: {
      type: "datetime",
      axisBorder: { show: false },
      axisTicks: { show: false },
      labels: {
        style: { fontWeight: 'bold', colors: "#1b1c1c" },
        datetimeUTC: false,
        trim: false,
        hideOverlappingLabels: false,
      },
      tickPlacement: 'on',
    },
    yaxis: {
      min: 0,
      labels: {
        style: { fontWeight: 'bold', colors: "#1b1c1c" },
      }
    },
    legend: {
      show: true,
      position: 'bottom',
      labels: { colors: "#1b1c1c", useSeriesColors: false },
      fontWeight: 700,
      offsetY: 20, // Tăng khoảng cách cho dễ nhìn hơn
    },
    tooltip: {
      theme: "dark",
      x: { format: "dd MMM yy" }
    },
    // --- Sử dụng cấu hình fill đã được đơn giản hóa ---
    fill: finalFillOptions,
    plotOptions: {
      bar: {
        horizontal: false,
        columnWidth: '55%',
      },
    },
    markers: {
      size: markersSize,
      strokeWidth: 0,
      hover: {
        sizeOffset: 4
      }
    },
  };

  return options;

}, [chartType, chartData, isFullscreen, predictMode]); // Dependencies giữ nguyên// Đã thêm predictMode


  // --- Render ---
  return (
    <div
      className={`rounded-lg dark:shadow-dark-md shadow-md bg-white dark:bg-darkgray p-6 relative w-full break-words transition-all duration-300 ease-in-out ${
        isFullscreen
          ? "fixed inset-0 z-50 overflow-auto" // CSS cho chế độ toàn màn hình
          : "" // CSS bình thường
      }`}
    >
      {/* Header */}
      <div className="flex flex-wrap justify-between items-center gap-4 mb-4">
        {/* Title */}
        <h5 className={`card-title text-lg font-semibold ${isFullscreen ? 'text-xl' : ''}`}>Chỉ số {title}</h5>

        {/* Controls */}
        <div className="flex flex-wrap justify-end items-center gap-3 sm:gap-5">
          {/* Chart Type Selector */}
          <div className="flex items-center gap-2">
            <Select value={chartType} onValueChange={(value: ChartType) => setChartType(value)}>
              <SelectTrigger className="h-8 w-32 text-sm">
                <SelectValue placeholder="Loại biểu đồ" />
              </SelectTrigger>
              <SelectContent className="w-32 text-sm">
                <SelectItem value="line" className="text-sm">
                  <div className="flex items-center gap-2"><LineChart size={16} /> Đường</div>
                </SelectItem>
                <SelectItem value="bar" className="text-sm">
                  <div className="flex items-center gap-2"><BarChart size={16} /> Cột</div>
                </SelectItem>
                <SelectItem value="area" className="text-sm">
                  <div className="flex items-center gap-2"><AreaChart size={16} /> Miền</div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Predict Controls */}
          <div className="flex items-center gap-2">
            <label htmlFor="predict-mode" className="text-sm cursor-pointer">Dự đoán</label>
            <Switch
              checked={predictMode}
              onCheckedChange={() => setPredictMode((prev) => !prev)}
              id="predict-mode"
              className="data-[state=unchecked]:bg-gray-300 data-[state=checked]:bg-blue-500"
            />
          </div>
          <Select
            value={predictRange.toString()}
            disabled={!predictMode}
            onValueChange={(value) => setPredictRange(parseInt(value))}
          >
            <SelectTrigger className="h-8 w-32 text-sm" disabled={!predictMode}>
              <SelectValue placeholder="Ngày 1" />
            </SelectTrigger>
            <SelectContent className="w-32 text-sm">
              {Array.from({ length: 10 }, (_, i) => (
                <SelectItem key={i + 1} value={(i + 1).toString()} className="text-sm">{`${i + 1} Ngày`}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Fullscreen Button */}
            <Button
              variant="outline"
              size="icon"
              onClick={toggleFullscreen}
              className="h-8 w-8"
              aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            >
              {isFullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
            </Button>
        </div>
      </div>

      {/* Chart Area */}
      {/* Có thể thử bỏ margin âm nếu vẫn gặp vấn đề layout rìa */}
      <div className={`-ms-4 -me-3 mt-2 ${isFullscreen ? 'h-[calc(100vh-100px)]' : ''}`}>
        {isClient && (
          <Chart
            options={chartOptions}
            series={chartData} // Dữ liệu đã xử lý
            type={chartType}
            height={isFullscreen ? "100%" : "315px"}
            width="100%"
          />
        )}
        {!isClient && <p>Loading chart...</p>} {/* Hiển thị loading nếu chưa render phía client */}
      </div>
    </div>
  );
};

export default Chartline;