"use client";
import { AIModel } from "@/types/aimodel";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic"; // Import dynamic
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import PageLoader from "@/components/pageloader";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { RefreshCw, Cpu, BarChartHorizontal } from "lucide-react";
import type { ApexOptions } from "apexcharts"; // Import ApexOptions type

// --- Dynamic Import Chart ---
// Chỉ load chart ở phía client
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false, loading: () => <p className="text-center text-muted-foreground">Đang tải biểu đồ...</p> });
// --- Dữ liệu giả lập cho nhiều models (Thêm performanceHistory) ---
const MOCK_MODELS: AIModel[] = [
    {
      id: "itransformer-v1",
      name: "Itransformer",
      type: "Transformer",
      version: "1.0",
      lastTrained: "2025-03-15",
      status: "active",
      description: "Mô hình Transformer tối ưu cho dự đoán chuỗi thời gian.",
      useCase: "Dự đoán chỉ số chất lượng nước (WQI) cho ngày tiếp theo.", // <-- Mới
      inputFeatures: ["DO", "pH", "Nhiệt độ", "TSS", "COD", "Lịch sử WQI"], // <-- Mới
      outputTarget: "Chỉ số WQI (0-100)", // <-- Mới
      lastTrainingDuration: "2 giờ 15 phút", // <-- Mới
      deploymentDate: "2025-03-20", // <-- Mới
      metrics: { accuracy: 90.2, precision: 90.8, recall: 90.5 }, // Giả sử đây là model phân loại WQI tốt/xấu
      usageStats: { totalRequests: 2400, avgResponseTimeMs: 4000, errorRatePercent: 3, activeUsers: 20, peakUsageTime: "17/03/2025", peakUsageCount: 210 },
      trainingDataSize: "3K mẫu",
      performanceHistory: [
        { date: "2025-01-10", metrics: { accuracy: 88.5, precision: 89.0, recall: 88.0 } },
        { date: "2025-01-25", metrics: { accuracy: 89.1, precision: 89.5, recall: 88.8 } },
        { date: "2025-02-10", metrics: { accuracy: 89.8, precision: 90.1, recall: 89.5 } },
        { date: "2025-02-28", metrics: { accuracy: 90.0, precision: 90.5, recall: 90.0 } },
        { date: "2025-03-15", metrics: { accuracy: 90.2, precision: 90.8, recall: 90.5 } },
      ]
    },
    {
      id: "lstm-v2",
      name: "LSTM Predictor",
      type: "LSTM",
      version: "2.1",
      lastTrained: "2025-04-01",
      status: "active",
      description: "Mạng LSTM cho dự báo nhu cầu sản phẩm.",
      useCase: "Dự báo doanh số bán hàng tháng tới.", // <-- Mới
      inputFeatures: ["Doanh số lịch sử", "Chiến dịch marketing", "Dữ liệu mùa vụ"], // <-- Mới
      outputTarget: "Doanh số dự kiến (số lượng)", // <-- Mới
      lastTrainingDuration: "5 giờ", // <-- Mới
      deploymentDate: "2025-04-05", // <-- Mới
      metrics: { mae: 15.5, rmse: 20.1, r2: 0.85 },
      usageStats: { totalRequests: 5500, avgResponseTimeMs: 6500, errorRatePercent: 2, activeUsers: 45, peakUsageTime: "02/04/2025", peakUsageCount: 400 },
      trainingDataSize: "10K mẫu",
       performanceHistory: [
        { date: "2025-02-05", metrics: { mae: 18.2, rmse: 23.5, r2: 0.80 } },
        { date: "2025-02-20", metrics: { mae: 17.0, rmse: 22.0, r2: 0.82 } },
        { date: "2025-03-10", metrics: { mae: 16.1, rmse: 21.0, r2: 0.84 } },
        { date: "2025-03-25", metrics: { mae: 15.8, rmse: 20.5, r2: 0.845 } },
        { date: "2025-04-01", metrics: { mae: 15.5, rmse: 20.1, r2: 0.85 } },
      ]
    },
     {
      id: "prophet-daily",
      name: "Prophet Daily Trends",
      type: "Statistical",
      version: "1.5",
      lastTrained: "2025-03-28",
      status: "inactive",
      description: "Mô hình Prophet của Facebook để dự báo xu hướng hàng ngày.",
      useCase: "Phát hiện xu hướng truy cập website.", // <-- Mới
      inputFeatures: ["Lượt truy cập hàng ngày"], // <-- Mới
      outputTarget: "Dự báo lượt truy cập & điểm bất thường", // <-- Mới
      lastTrainingDuration: "30 phút", // <-- Mới
      deploymentDate: "2024-12-01", // <-- Mới
      metrics: { mae: 5.2, rmse: 7.8 },
      usageStats: { totalRequests: 120, avgResponseTimeMs: 1500, errorRatePercent: 1, activeUsers: 5 },
      trainingDataSize: "1K mẫu",
    },
      {
      id: "itransformer-v2-training",
      name: "Itransformer Enhanced",
      type: "Transformer",
      version: "2.0 Beta",
      lastTrained: "Đang huấn luyện...",
      status: "training",
      description: "Phiên bản nâng cấp đang được huấn luyện.",
      useCase: "Dự đoán WQI với độ chính xác cao hơn.", // <-- Mới
      inputFeatures: ["Dữ liệu cảm biến mở rộng", "Dữ liệu thời tiết"], // <-- Mới
      outputTarget: "Chỉ số WQI (0-100)", // <-- Mới
      // Không có duration và deployment date khi đang training
      metrics: {},
      usageStats: { totalRequests: 0, avgResponseTimeMs: 0, errorRatePercent: 0, activeUsers: 0 },
      trainingDataSize: "5K mẫu",
    },
  ];
  

export default function ModelManagement() {
  const [models, setModels] = useState<AIModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [retrainingStatus, setRetrainingStatus] = useState<{ [key: string]: boolean }>({});
  const [activeTab, setActiveTab] = useState("overview");
  const [isClient, setIsClient] = useState(false); // State để kiểm tra client-side rendering

  // --- State cho dữ liệu và tùy chọn biểu đồ ---
  const [performanceChartData, setPerformanceChartData] = useState<ApexAxisChartSeries>([]);
  const [performanceChartOptions, setPerformanceChartOptions] = useState<ApexOptions>({});
  const [usageChartData, setUsageChartData] = useState<ApexAxisChartSeries>([]); // State cho biểu đồ sử dụng
  const [usageChartOptions, setUsageChartOptions] = useState<ApexOptions>({}); // State cho biểu đồ sử dụng


  // Effect chạy 1 lần khi component mount để đánh dấu là client-side
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Effect để tải dữ liệu model ban đầu
  useEffect(() => {
    const timeout = setTimeout(() => {
      setModels(MOCK_MODELS);
      if (MOCK_MODELS.length > 0) {
        setSelectedModelId(MOCK_MODELS[0].id);
      }
      setIsLoading(false);
    }, 1000);
    return () => clearTimeout(timeout);
  }, []);

  // --- Effect để cập nhật dữ liệu biểu đồ khi model được chọn thay đổi ---
  useEffect(() => {
    if (!selectedModelId || models.length === 0) return;

    const model = models.find(m => m.id === selectedModelId);
    if (!model || !model.performanceHistory || model.performanceHistory.length === 0) {
        setPerformanceChartData([]); // Reset chart data nếu không có history
        setUsageChartData([]); // Reset chart usage data
        return; // Không có model hoặc history, không cần làm gì thêm
    }

    // --- Xử lý Performance Chart ---
    const history = model.performanceHistory;
    const performanceSeries: ApexAxisChartSeries = [];
    const performanceCategories = history.map(h => new Date(h.date).getTime()); // Chuyển date thành timestamp cho trục X

    // Thêm series dựa trên metrics có sẵn
    if (history[0].metrics.accuracy !== undefined) {
      performanceSeries.push({ name: "Accuracy (%)", data: history.map(h => ({ x: new Date(h.date).getTime(), y: h.metrics.accuracy ?? 0 })) });
    }
    if (history[0].metrics.precision !== undefined) {
      performanceSeries.push({ name: "Precision (%)", data: history.map(h => ({ x: new Date(h.date).getTime(), y: h.metrics.precision ?? 0 })) });
    }
    if (history[0].metrics.recall !== undefined) {
      performanceSeries.push({ name: "Recall (%)", data: history.map(h => ({ x: new Date(h.date).getTime(), y: h.metrics.recall ?? 0 })) });
    }
    if (history[0].metrics.mae !== undefined) {
      performanceSeries.push({ name: "MAE", data: history.map(h => ({ x: new Date(h.date).getTime(), y: h.metrics.mae ?? 0 })) });
    }
     if (history[0].metrics.rmse !== undefined) {
      performanceSeries.push({ name: "RMSE", data: history.map(h => ({ x: new Date(h.date).getTime(), y: h.metrics.rmse ?? 0 })) });
    }
    // Thêm R2 nếu muốn
     if (history[0].metrics.r2 !== undefined) {
       // performanceSeries.push({ name: "R-squared", data: history.map(h => ({ x: new Date(h.date).getTime(), y: h.metrics.r2 ?? 0 })) });
     }

    setPerformanceChartData(performanceSeries);

    // Cấu hình options chung cho performance chart
    setPerformanceChartOptions({
      chart: {
        height: 300,
        type: 'line',
        zoom: { enabled: false },
        toolbar: { show: true, tools: { download: true, selection: false, zoom: false, zoomin: false, zoomout: false, pan: false, reset: true } },
        fontFamily: 'inherit',
      },
      stroke: { curve: 'smooth', width: 2 },
      dataLabels: { enabled: false },
      markers: { size: 4, hover: { sizeOffset: 2 } },
      xaxis: {
        type: 'datetime',
        categories: performanceCategories, // Dùng timestamp
         labels: { datetimeUTC: false, format: 'dd MMM yy' }, // Format ngày tháng
        tooltip: { enabled: false }
      },
      yaxis: {
        title: { text: 'Giá trị Metric' },
        labels: {
          formatter: (val) => val.toFixed(1) // Format số thập phân
        }
      },
      tooltip: {
        x: { format: 'dd MMM yyyy' }, // Format tooltip ngày
        y: { formatter: (val) => val !== undefined ? val.toFixed(2) : 'N/A' } // Format tooltip giá trị
      },
      legend: { position: 'top' },
      grid: {
          borderColor: '#e7e7e7',
          row: { colors: ['#f3f3f3', 'transparent'], opacity: 0.5 },
      },
    });

    // --- Xử lý Usage Chart (Ví dụ: Total Requests theo ngày - cần dữ liệu usage history) ---
    // Giả sử bạn có dữ liệu usage history (tương tự performance history)
    // const usageHistory = model.usageHistory || [];
    // if (usageHistory.length > 0) { ... }
    // --- Tạo dữ liệu giả lập cho usage chart ---
     const usageSeries = [{
        name: "Số yêu cầu",
        data: history.map((h, index) => ({
            x: new Date(h.date).getTime(),
            // Tạo số yêu cầu giả lập tăng dần
            y: Math.floor((model.usageStats.totalRequests / history.length) * (index + 1) + (Math.random() - 0.5) * 100)
        }))
     }];
     setUsageChartData(usageSeries);
     setUsageChartOptions({
        chart: { type: 'area', height: 300, zoom: { enabled: false }, toolbar: { show: false }, fontFamily: 'inherit', },
        dataLabels: { enabled: false },
        stroke: { curve: 'smooth', width: 2 },
        xaxis: {
            type: 'datetime',
             labels: { datetimeUTC: false, format: 'dd MMM yy' },
        },
        yaxis: { title: { text: 'Số lượng yêu cầu' } },
        tooltip: { x: { format: 'dd MMM yyyy' } },
        grid: { borderColor: '#e7e7e7' },
     });


  }, [selectedModelId, models]); // Chạy lại khi selectedModelId hoặc models thay đổi


  const handleRetrain = (modelId: string) => {
    // ... (logic handleRetrain giữ nguyên)
    if (!modelId) return;
    setRetrainingStatus(prev => ({ ...prev, [modelId]: true }));
    console.log(`Bắt đầu huấn luyện lại model: ${modelId}`);
    setTimeout(() => {
      console.log(`Hoàn thành huấn luyện lại model: ${modelId}`);
       setModels(prevModels => prevModels.map(model =>
           model.id === modelId
             ? {
                 ...model,
                 status: 'active',
                 lastTrained: new Date().toLocaleDateString('vi-VN'), // Cập nhật ngày theo locale VN
                 // Quan trọng: Cập nhật lại metrics và performanceHistory sau khi huấn luyện
                 metrics: { ...model.metrics, accuracy: (model.metrics.accuracy ?? 85) + Math.random() * 5 }, // ví dụ tăng accuracy
                 performanceHistory: [ // Thêm điểm dữ liệu mới
                   ...(model.performanceHistory ?? []),
                   { date: new Date().toISOString(), metrics: { accuracy: (model.metrics.accuracy ?? 85) + Math.random() * 5 } } // Chỉ ví dụ accuracy
                 ]
             }
             : model
       ));
      setRetrainingStatus(prev => ({ ...prev, [modelId]: false }));
    }, 5000);
  };

  const selectedModel = models.find(m => m.id === selectedModelId);

  const getStatusBadge = (status: AIModel['status']) => {
    // ... (logic getStatusBadge giữ nguyên)
     switch (status) {
       case "active":
         return <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-800">Hoạt động</span>;
       case "inactive":
         return <span className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-800">Không hoạt động</span>;
        case "training":
            return <span className="inline-flex items-center rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-800">Đang huấn luyện</span>;
       case "error":
         return <span className="inline-flex items-center rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-medium text-red-800">Lỗi</span>;
       default:
         return <span className="inline-flex items-center rounded-full bg-yellow-100 px-2.5 py-0.5 text-xs font-medium text-yellow-800">Không xác định</span>;
     }
  }

  if (isLoading) return <PageLoader message="Đang tải trang quản lý mô hình..." />;

  return (
    <div className="flex min-h-screen flex-col">
      <main className="flex-1">
        <div className="container py-6">
           {/* Header: Title, Select, Button (giữ nguyên) */}
           <div className="mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <h1 className="text-3xl font-bold">Quản lý Model AI</h1>
                <div className="flex w-full sm:w-auto flex-col sm:flex-row items-stretch sm:items-center gap-2">
                    <Select value={selectedModelId} onValueChange={setSelectedModelId} disabled={models.length === 0}>
                        <SelectTrigger className="w-full sm:w-[250px]">
                        <SelectValue placeholder="Chọn model để quản lý" />
                        </SelectTrigger>
                        <SelectContent>
                        {models.map(model => (
                            <SelectItem key={model.id} value={model.id}>
                            <div className="flex items-center gap-2">
                                {model.type === 'Transformer' && <Cpu size={16} className="text-blue-500"/>}
                                {model.type === 'LSTM' && <BarChartHorizontal size={16} className="text-green-500"/>}
                                {model.type === 'Statistical' && <RefreshCw size={16} className="text-orange-500"/>}
                                <span>{model.name} ({model.version})</span>
                            </div>
                            </SelectItem>
                        ))}
                        {models.length === 0 && <SelectItem value="loading" disabled>Đang tải...</SelectItem>}
                        </SelectContent>
                    </Select>
                    {selectedModel && (
                        <AlertDialog>
                            <AlertDialogTrigger asChild>
                                <Button disabled={retrainingStatus[selectedModelId] || selectedModel.status === 'training'}>
                                    {retrainingStatus[selectedModelId] || selectedModel.status === 'training' ? (
                                        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                    ) : (
                                        <RefreshCw className="mr-2 h-4 w-4" />
                                    )}
                                Huấn luyện lại
                                </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                                <AlertDialogHeader>
                                <AlertDialogTitle>Xác nhận huấn luyện lại Model?</AlertDialogTitle>
                                <AlertDialogDescription>
                                    Việc huấn luyện lại model <strong>{selectedModel?.name} ({selectedModel?.version})</strong> sẽ sử dụng dữ liệu mới nhất...
                                </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                    <AlertDialogCancel>Huỷ bỏ</AlertDialogCancel>
                                    <AlertDialogAction onClick={() => handleRetrain(selectedModelId)} disabled={retrainingStatus[selectedModelId]}>
                                        {retrainingStatus[selectedModelId] ? "Đang huấn luyện..." : "Bắt đầu huấn luyện"}
                                    </AlertDialogAction>
                                </AlertDialogFooter>
                            </AlertDialogContent>
                        </AlertDialog>
                    )}
                </div>
           </div>

           {/* Tabs & Content */}
           {selectedModel ? (
             <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
               <TabsList className="mb-4">
                 <TabsTrigger value="overview">Tổng quan</TabsTrigger>
                 <TabsTrigger value="performance">Hiệu quả hoạt động</TabsTrigger>
                 <TabsTrigger value="usage">Thống kê sử dụng</TabsTrigger>
               </TabsList>

               {/* Tab Tổng quan (giữ nguyên) */}
               <TabsContent value="overview">
                <Card>
                   <CardHeader>
                     <CardTitle>Thông tin Model: {selectedModel.name}</CardTitle>
                     {/* Sử dụng useCase nếu có, nếu không dùng description */}
                     <CardDescription>{selectedModel.useCase ?? selectedModel.description}</CardDescription>
                   </CardHeader>
                   <CardContent>
                     {/* Chia thành các nhóm thông tin */}
                     <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">

                        {/* Nhóm 1: Thông tin cơ bản */}
                        <div className="space-y-3 border-b md:border-b-0 md:border-r md:pr-8 pb-4 md:pb-0">
                             <h4 className="text-sm font-semibold text-primary mb-2">Thông tin cơ bản</h4>
                             <div className="grid grid-cols-[150px,1fr] gap-2 items-center"> {/* Cố định chiều rộng cột label */}
                                 <div className="text-sm font-medium text-muted-foreground">Loại Model:</div>
                                 <div>{selectedModel.type}</div>
                             </div>
                             <div className="grid grid-cols-[150px,1fr] gap-2 items-center">
                                 <div className="text-sm font-medium text-muted-foreground">Phiên bản:</div>
                                 <div>{selectedModel.version}</div>
                             </div>
                             <div className="grid grid-cols-[150px,1fr] gap-2 items-center">
                                 <div className="text-sm font-medium text-muted-foreground">Trạng thái:</div>
                                 <div>{getStatusBadge(selectedModel.status)}</div>
                             </div>
                             {selectedModel.deploymentDate && (
                                <div className="grid grid-cols-[150px,1fr] gap-2 items-center">
                                    <div className="text-sm font-medium text-muted-foreground">Ngày triển khai:</div>
                                    <div>{selectedModel.deploymentDate}</div>
                                </div>
                             )}
                             {selectedModel.outputTarget && (
                                <div className="grid grid-cols-[150px,1fr] gap-2 items-start">
                                    <div className="text-sm font-medium text-muted-foreground">Mục tiêu đầu ra:</div>
                                    <div>{selectedModel.outputTarget}</div>
                                </div>
                            )}
                            {selectedModel.inputFeatures && selectedModel.inputFeatures.length > 0 && (
                               <div className="grid grid-cols-[150px,1fr] gap-2 items-start">
                                   <div className="text-sm font-medium text-muted-foreground">Đặc trưng đầu vào:</div>
                                   {/* Hiển thị dạng danh sách hoặc tag nhỏ */}
                                   <div className="flex flex-wrap gap-1">
                                       {selectedModel.inputFeatures.map((feature, index) => (
                                           <span key={index} className="text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded">
                                               {feature}
                                           </span>
                                       ))}
                                   </div>
                               </div>
                           )}
                        </div>

                        {/* Nhóm 2: Thông tin huấn luyện & Hiệu suất chính */}
                        <div className="space-y-3">
                           <h4 className="text-sm font-semibold text-primary mb-2">Huấn luyện & Hiệu suất</h4>
                            <div className="grid grid-cols-[180px,1fr] gap-2 items-center">
                                 <div className="text-sm font-medium text-muted-foreground">Lần huấn luyện gần nhất:</div>
                                 <div>{selectedModel.lastTrained}</div>
                             </div>
                              {selectedModel.lastTrainingDuration && (
                                <div className="grid grid-cols-[180px,1fr] gap-2 items-center">
                                    <div className="text-sm font-medium text-muted-foreground">Thời gian huấn luyện:</div>
                                    <div>{selectedModel.lastTrainingDuration}</div>
                                </div>
                             )}
                              {selectedModel.trainingDataSize && (
                                <div className="grid grid-cols-[180px,1fr] gap-2 items-center">
                                <div className="text-sm font-medium text-muted-foreground">Dữ liệu huấn luyện:</div>
                                <div>{selectedModel.trainingDataSize}</div>
                                </div>
                             )}
                             {/* Hiển thị các metric chính */}
                              {selectedModel.metrics.accuracy !== undefined && (
                                <div className="grid grid-cols-[180px,1fr] gap-2 items-center">
                                    <div className="text-sm font-medium text-muted-foreground">Độ chính xác:</div>
                                    <div className="font-semibold">{selectedModel.metrics.accuracy}%</div>
                                </div>
                               )}
                               {selectedModel.metrics.mae !== undefined && (
                                <div className="grid grid-cols-[180px,1fr] gap-2 items-center">
                                    <div className="text-sm font-medium text-muted-foreground">MAE:</div>
                                    <div className="font-semibold">{selectedModel.metrics.mae}</div>
                                </div>
                               )}
                                {selectedModel.metrics.r2 !== undefined && (
                                <div className="grid grid-cols-[180px,1fr] gap-2 items-center">
                                    <div className="text-sm font-medium text-muted-foreground">R-squared:</div>
                                    <div className="font-semibold">{selectedModel.metrics.r2}</div>
                                </div>
                               )}
                               {/* Bạn có thể thêm các metric khác nếu muốn */}
                        </div>

                     </div>
                   </CardContent>
                 </Card>
              </TabsContent>


               {/* --- Tab Hiệu quả hoạt động --- */}
               <TabsContent value="performance">
                  <Card>
                    <CardHeader>
                      <CardTitle>Chỉ số hiệu suất: {selectedModel.name}</CardTitle>
                      <CardDescription>Phân tích chi tiết hiệu suất của mô hình</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-6">
                        {/* Phần hiển thị các metric hiện tại (giữ nguyên) */}
                        <div className="grid gap-4 md:grid-cols-3">
                          {/* ... hiển thị các Card metric nhỏ ... */}
                           {selectedModel.metrics.accuracy !== undefined && ( <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-4"><div className="text-sm font-medium text-muted-foreground">Độ chính xác</div><div className="mt-2 text-2xl font-bold">{selectedModel.metrics.accuracy}%</div></div> )}
                           {selectedModel.metrics.precision !== undefined && ( <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-4"><div className="text-sm font-medium text-muted-foreground">Precision</div><div className="mt-2 text-2xl font-bold">{selectedModel.metrics.precision}%</div></div> )}
                           {/* ... các metric khác ... */}
                           {selectedModel.metrics.mae !== undefined && ( <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-4"><div className="text-sm font-medium text-muted-foreground">MAE</div><div className="mt-2 text-2xl font-bold">{selectedModel.metrics.mae}</div></div> )}

                        </div>

                        {/* --- Phần Biểu đồ --- */}
                        <div className="rounded-lg border p-4">
                          <h3 className="mb-4 text-lg font-medium">Hiệu suất theo thời gian</h3>
                          <div className="h-72 w-full"> {/* Đặt chiều cao cố định cho container biểu đồ */}
                            {isClient && performanceChartData.length > 0 ? (
                              <Chart
                                options={performanceChartOptions}
                                series={performanceChartData}
                                type="line" // Có thể đổi thành "area" nếu muốn
                                height="100%"
                                width="100%"
                              />
                            ) : (
                              // Hiển thị khi đang tải hoặc không có dữ liệu
                              <div className="flex items-center justify-center h-full bg-muted rounded">
                                <p className="text-muted-foreground">
                                    {isClient ? "Không có dữ liệu lịch sử hiệu suất cho model này." : "Đang tải biểu đồ..."}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
               </TabsContent>

               {/* --- Tab Thống kê sử dụng --- */}
               <TabsContent value="usage">
                   <Card>
                   <CardHeader>
                     <CardTitle>Thống kê sử dụng: {selectedModel.name}</CardTitle>
                     <CardDescription>Số liệu sử dụng mô hình</CardDescription>
                   </CardHeader>
                   <CardContent>
                      {/* Phần hiển thị các chỉ số sử dụng (giữ nguyên) */}
                     <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 mb-6">
                         {/* ... hiển thị các Card usage nhỏ ... */}
                         <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-4"><div className="text-sm font-medium text-muted-foreground">Tổng số yêu cầu</div><div className="mt-2 text-2xl font-bold">{selectedModel.usageStats.totalRequests.toLocaleString()}</div></div>
                         {/* ... các stats khác ... */}
                     </div>

                      {/* --- Phần Biểu đồ sử dụng --- */}
                      <div className="rounded-lg border p-4">
                         <h3 className="mb-4 text-lg font-medium">Lượng sử dụng theo thời gian</h3>
                          <div className="h-72 w-full">
                             {isClient && usageChartData.length > 0 ? (
                               <Chart
                                 options={usageChartOptions}
                                 series={usageChartData}
                                 type="area" // Dùng area chart cho usage
                                 height="100%"
                                 width="100%"
                               />
                             ) : (
                               <div className="flex items-center justify-center h-full bg-muted rounded">
                                 <p className="text-muted-foreground">
                                     {isClient ? "Không có dữ liệu lịch sử sử dụng." : "Đang tải biểu đồ..."}
                                 </p>
                               </div>
                             )}
                           </div>
                       </div>
                   </CardContent>
                 </Card>
              </TabsContent>

             </Tabs>
           ) : (
             // Hiển thị khi chưa chọn model (giữ nguyên)
             <div className="text-center py-10">
                <p className="text-muted-foreground">
                    {models.length > 0 ? "Vui lòng chọn một model từ danh sách ở trên." : "Không có model nào để hiển thị."}
                 </p>
             </div>
           )}
        </div>
      </main>
    </div>
  );
}