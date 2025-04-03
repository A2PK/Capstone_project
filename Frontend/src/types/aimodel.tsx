export interface AIModel {
    id: string;
    name: string;
    type: string; // Ví dụ: "Transformer", "LSTM", "Statistical"
    version: string;
    lastTrained: string; // Nên dùng kiểu Date hoặc ISO string trong thực tế
    status: "active" | "inactive" | "training" | "error";
    description: string;
    useCase?: string; // Mục đích sử dụng cụ thể
    inputFeatures?: string[]; // Danh sách các đặc trưng đầu vào
    outputTarget?: string;  // Mô tả đầu ra của model
    lastTrainingDuration?: string; // Ví dụ: "45 phút", "3 giờ"
    deploymentDate?: string; // Ngày triển khai phiên bản này, ví dụ: "2025-03-20"
    metrics: {
      accuracy?: number; // Độ chính xác tổng thể
      precision?: number; // Độ chính xác phân loại
      recall?: number; // Khả năng nhận diện
      mae?: number; // Mean Absolute Error (cho mô hình hồi quy)
      rmse?: number; // Root Mean Squared Error (cho mô hình hồi quy)
      r2?: number; // R-squared (cho mô hình hồi quy)
      // Thêm các metric khác nếu cần
    };
    usageStats: {
      totalRequests: number;
      avgResponseTimeMs: number;
      errorRatePercent: number;
      activeUsers: number;
      peakUsageTime?: string; // Thời điểm sử dụng cao nhất
      peakUsageCount?: number; // Số lượng yêu cầu lúc cao điểm
    };
    trainingDataSize?: string; // Kích thước dữ liệu huấn luyện
    performanceHistory?: {
        date: string; // ISO string date
        metrics: AIModel['metrics'];
    }[];
  }