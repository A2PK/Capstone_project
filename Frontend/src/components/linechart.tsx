// File: src/components/linechart.tsx
"use client";

import dynamic from "next/dynamic";
import { useMemo, useState, useEffect, useCallback } from "react";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Maximize, Minimize, LineChart, BarChart, AreaChart } from 'lucide-react';
import type { ApexOptions} from "apexcharts"; // Thêm ApexMarkerShape nếu dùng
import type { DataPoint } from "@/types/station2"; // Đảm bảo đường dẫn này đúng
import { format } from 'date-fns';
import { getDonvi } from "@/lib/utils";
// Dynamic Import Chart
const Chart = dynamic(() => import("react-apexcharts"), {
    ssr: false,
    loading: () => <div className="flex justify-center items-center h-[300px] sm:h-[315px]"><p>Đang tải biểu đồ...</p></div>
});

// Props Interface
interface ChartlineProps {
    historicalDataPoints: DataPoint[];
    groupedPredictionDataPoints: Map<string, DataPoint[]>;
    selectedFeature: string;
    title: string;
}

// Chart Type Definition
type ChartType = "line" | "area" | "bar";

// Hàm trợ giúp lấy giá trị
const getPointValue = (dp: DataPoint, featureName: string): number | null => {
    if (!dp) return null;
    if (featureName === "WQI") {
        const wqiValue = dp.wqi;
        return typeof wqiValue === 'number' && !isNaN(wqiValue) ? wqiValue : null;
    } else {
        const feature = Array.isArray(dp.features) ? dp.features.find(f => f.name === featureName) : undefined;
        if (feature) {
            if (typeof feature.textualValue === 'string' && feature.textualValue.trim() !== '') {
                const parsedTextual = parseFloat(feature.textualValue);
                if (!isNaN(parsedTextual)) return parsedTextual;
            }
            if (typeof feature.value === 'number' && !isNaN(feature.value)) {
                return feature.value;
            }
        }
    }
    return null;
};


// --- Định nghĩa kiểu cho một phần tử trong mảng series của ApexCharts ---
type ApexChartSeriesDataElement = {
    x: number;
    y: number | null;
};
type ApexChartSeries = {
    name?: string;
    data: ApexChartSeriesDataElement[];
    type?: string;
    color?: string;
};

// Component Implementation
const Chartline: React.FC<ChartlineProps> = ({
    historicalDataPoints,
    groupedPredictionDataPoints,
    selectedFeature,
    title
}) => {
    const [predictMode, setPredictMode] = useState(false);
    const [predictWeeks, setPredictWeeks] = useState<number>(1); // State lưu số tuần
    const [isClient, setIsClient] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [chartType, setChartType] = useState<ChartType>("line");

    // Định nghĩa helper functions bên ngoài useMemo
    const getBaseName = (name: string | undefined): string | undefined => {
        if (!name) return undefined;
        return name.replace(/_connector$/, '');
    };
    const isConnector = (name: string | undefined): boolean => {
        return !!name && name.endsWith('_connector');
    };
    const isHistory = (name: string | undefined): boolean => {
        return name === 'Lịch sử';
    };
    const SERIES_DISPLAY_NAMES: { [key: string]: string } = {
        'Lịch sử': 'Lịch sử', // Giữ nguyên hoặc đổi nếu muốn
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'ETSformer':'ETSformer',
        'ETSformerPar':'ETSformer Parallel',
        // Thêm các ánh xạ khác nếu cần
        // 'ten_goc_khac': 'Tên Hiển Thị Khác'
      };
    useEffect(() => {
        setIsClient(true);
        const handleEsc = (event: KeyboardEvent) => {
            if (event.key === 'Escape' && isFullscreen) setIsFullscreen(false);
        };
        document.addEventListener('keydown', handleEsc);
        return () => document.removeEventListener('keydown', handleEsc);
    }, [isFullscreen]);

    // --- Tính toán số tuần dự đoán tối đa có sẵn (đếm số điểm) --- <<<--- CẬP NHẬT LOGIC
    const maxAvailablePredictWeeks = useMemo((): number => {
        let maxWeeks = 0;
        if (!groupedPredictionDataPoints || groupedPredictionDataPoints.size === 0) {
            return 0;
        }
        // Sắp xếp historicalDataPoints một lần để lấy điểm cuối cùng
        const sortedHistory = [...historicalDataPoints].sort((a, b) => new Date(a.monitoringTime).getTime() - new Date(b.monitoringTime).getTime());
        const lastHistoricalTimestamp = sortedHistory.length > 0
            ? new Date(sortedHistory[sortedHistory.length - 1].monitoringTime).getTime()
            : undefined;

        groupedPredictionDataPoints.forEach((predictionPoints) => {
            if (!Array.isArray(predictionPoints)) return;

            // Đếm số điểm dự đoán thực sự nằm sau điểm lịch sử cuối cùng
            let futurePointsCount = 0;
            predictionPoints.forEach(dp => {
                try {
                    const timestamp = new Date(dp.monitoringTime).getTime();
                    // Chỉ đếm các điểm dự đoán SAU lịch sử cuối
                    if (!isNaN(timestamp) && (lastHistoricalTimestamp === undefined || timestamp > lastHistoricalTimestamp)) {
                        futurePointsCount++; // Mỗi điểm hợp lệ là một tuần dự đoán
                    }
                } catch (e) {
                     console.error("Error processing timestamp for max week calculation:", dp.monitoringTime, e);
                }
            });
            maxWeeks = Math.max(maxWeeks, futurePointsCount); // Tìm số tuần tối đa
        });

        return maxWeeks; // Trả về số tuần tối đa tìm được

    }, [groupedPredictionDataPoints, historicalDataPoints]); // Phụ thuộc vào dữ liệu đầu vào

    // --- useEffect để đảm bảo predictWeeks hợp lệ ---
     useEffect(() => {
        if (maxAvailablePredictWeeks > 0 && predictWeeks > maxAvailablePredictWeeks) {
            setPredictWeeks(maxAvailablePredictWeeks);
        } else if (maxAvailablePredictWeeks === 0 && predictWeeks !== 1) {
             setPredictWeeks(1);
        } else if (predictWeeks < 1) {
            setPredictWeeks(1);
        }
     }, [maxAvailablePredictWeeks, predictWeeks]);


    // Data Preparation with Connector Series Logic
    const chartData = useMemo((): ApexChartSeries[] => {
        const series: ApexChartSeries[] = [];
        const connectorSeries: ApexChartSeries[] = [];

        // 1. Historical Data (Giữ nguyên)
        const historicalSeriesData: ApexChartSeriesDataElement[] = [];
        historicalDataPoints.forEach(dp => {
            try {
                const timestamp = new Date(dp.monitoringTime).getTime();
                if (!isNaN(timestamp)) {
                    historicalSeriesData.push({ x: timestamp, y: getPointValue(dp, selectedFeature) });
                }
            } catch (e) { console.error("Error processing historical point:", dp.monitoringTime, e); }
        });
        const sortedHistoricalData = historicalSeriesData.sort((a, b) => a.x - b.x);
        const lastHistoricalPoint = sortedHistoricalData.length > 0 ? sortedHistoricalData[sortedHistoricalData.length - 1] : null;
        if (sortedHistoricalData.length > 0) {
            series.push({ name: "Lịch sử", data: sortedHistoricalData });
        }
        const lastHistoricalTimestamp = lastHistoricalPoint ? lastHistoricalPoint.x : undefined;

        // 2. Prediction Data & Connectors
        if (predictMode && groupedPredictionDataPoints && groupedPredictionDataPoints.size > 0) {
            groupedPredictionDataPoints.forEach((predictionPoints, source) => {
                const predictionSourceSeriesData: ApexChartSeriesDataElement[] = [];
                if (Array.isArray(predictionPoints)) {
                    predictionPoints.forEach(dp => {
                        try {
                            const timestamp = new Date(dp.monitoringTime).getTime();
                            if (!isNaN(timestamp) && (lastHistoricalTimestamp === undefined || timestamp > lastHistoricalTimestamp)) {
                                predictionSourceSeriesData.push({ x: timestamp, y: getPointValue(dp, selectedFeature) });
                            }
                        } catch (e) { console.error(`Error processing prediction point for source ${source}:`, dp.monitoringTime, e); }
                    });
                }
                // Sắp xếp các điểm dự đoán (tuần) theo thời gian
                const sortedSourcePredictions = predictionSourceSeriesData.sort((a, b) => a.x - b.x);

                // --- BỎ LỌC NGÀY DUY NHẤT ---

                // --- Cắt theo predictWeeks --- <<<--- CẬP NHẬT SLICE
                // Sử dụng trực tiếp mảng đã sắp xếp và lọc thời gian
                const finalWeeksToShow = sortedSourcePredictions.slice(0, predictWeeks); // Cắt theo số tuần

                // --- Thêm chuỗi dự đoán chính ---
                if (finalWeeksToShow.length > 0) {
                    series.push({
                        name: source,
                        data: finalWeeksToShow // Dữ liệu là các tuần đã chọn
                    });

                    // --- TẠO CHUỖI NỐI (CONNECTOR) ---
                    const firstPredictionPoint = finalWeeksToShow[0];
                    if (lastHistoricalPoint && firstPredictionPoint && (chartType === 'line' || chartType === 'area')) {
                         if (lastHistoricalPoint.x !== firstPredictionPoint.x) {
                            if (lastHistoricalPoint.y !== null && firstPredictionPoint.y !== null) {
                                connectorSeries.push({
                                    name: `${source}_connector`,
                                    data: [lastHistoricalPoint, firstPredictionPoint]
                                });
                            }
                         }
                    }
                }
            });
        }
        return [...series, ...connectorSeries];

    }, [
        historicalDataPoints, groupedPredictionDataPoints, selectedFeature,
        predictMode, predictWeeks, chartType, getPointValue // <<<--- Dùng predictWeeks
    ]);

    const toggleFullscreen = useCallback(() => setIsFullscreen((prev) => !prev), []);

    // Chart Options Calculation
    const chartOptions = useMemo((): ApexOptions => {
        const currentChartData = chartData; // Sử dụng chartData từ useMemo trước đó

        // Trả về options tối thiểu nếu không có dữ liệu hợp lệ
        if (!currentChartData || currentChartData.length === 0 || currentChartData.every(s => !s.data || s.data.length === 0)) {
             return {
                 chart: { id: `chart-${title}-${selectedFeature}-nodata`, toolbar: { show: false } },
                 noData: { text: 'Không có dữ liệu phù hợp để hiển thị.' }
             };
        }

        // --- Helpers đã được định nghĩa bên ngoài component: getBaseName, isConnector, isHistory ---

        // --- Colors ---
        const baseColors = ["#008FFB", "#FF4560", "#00E396", "#FEB019", "#775DD0", "#546E7A", "#D10CE8"];
        const predictionBaseNames = Array.from(new Set(
            currentChartData.map(s => getBaseName(s.name))
                           .filter((name): name is string => !!name && !isHistory(name))
        ));
        const colorMap = new Map<string, string>();
        colorMap.set('Lịch sử', baseColors[0]);
        const usedIndexes = [0];
        predictionBaseNames.forEach((baseName, index) => {
             let colorIndex = 0;
             if (baseName?.toLowerCase().includes('xgb')) colorIndex = 1;
             else if (baseName?.toLowerCase().includes('rf')) colorIndex = 2;
             else colorIndex = (index + 1) % baseColors.length;
             while (usedIndexes.includes(colorIndex)) {
                 colorIndex = (colorIndex + 1) % baseColors.length;
             }
             usedIndexes.push(colorIndex);
             colorMap.set(baseName, baseColors[colorIndex]);
        });
        const chartColors = currentChartData.map(s => {
             const baseName = getBaseName(s.name);
             return colorMap.get(baseName || '') || baseColors[baseColors.length - 1];
        });

        // --- Marker Sizes ---
        const markerSizes = currentChartData.map(s => isConnector(s.name) ? 0 : 3);

        // --- Grid Padding ---
        let gridPaddingRight = 15;
        if (chartType === 'line' || chartType === 'area') gridPaddingRight = 30;

        // --- Fill Options (for Area) ---
        let finalFillOptions: ApexOptions['fill'] = {};
         if (chartType === 'area') {
             const baseOpacityFrom = 0.45;
             const baseOpacityTo = 0.35;
             const stepDown = 0.05;
             const predictionIndexMap = new Map<string, number>();
             predictionBaseNames.forEach((name, index) => predictionIndexMap.set(name, index));

             const opacityFrom = currentChartData.map((s: ApexChartSeries) => {
                 if (isConnector(s.name)) return 0;
                 if (isHistory(s.name)) return baseOpacityFrom;
                 const baseName = getBaseName(s.name);
                 const predictIndex = baseName ? predictionIndexMap.get(baseName) : -1;
                 return Math.max(0.1, baseOpacityFrom * 0.8 - (predictIndex !== undefined && predictIndex >= 0 ? predictIndex : 0) * stepDown);
             });
             const opacityTo = currentChartData.map((s: ApexChartSeries) => {
                  if (isConnector(s.name)) return 0;
                  if (isHistory(s.name)) return baseOpacityTo;
                  const baseName = getBaseName(s.name);
                  const predictIndex = baseName ? predictionIndexMap.get(baseName) : -1;
                  return Math.max(0.05, baseOpacityTo * 0.8 - (predictIndex !== undefined && predictIndex >= 0 ? predictIndex : 0) * stepDown * 0.5);
             });

             finalFillOptions = {
                 type: 'gradient',
                 gradient: {
                     shadeIntensity: 0.3,
                     inverseColors: false,
                     opacityFrom: opacityFrom,
                     opacityTo: opacityTo,
                     stops: [0, 95, 100]
                 }
             };
         } else {
             finalFillOptions = { type: 'solid', opacity: (chartType === 'bar' ? 0.85 : 1) };
         }
         const currentUnit = getDonvi(selectedFeature);
        // --- Final Options Object ---
        return {
            chart: {
                id: `chart-${title}-${selectedFeature}`,
                fontFamily: "inherit",
                foreColor: "#adb0bb", // Màu chữ mặc định cho các phần tử chart (trục, etc.)
                offsetX: 0,
                offsetY: 0,
                // animations: {
                //     enabled: true,
                //     speed: 300 // Tốc độ animation
                // },
                animations: {
                    enabled: true // Tạm thời tắt animation
                },
                toolbar: {
                    show: true // Hiển thị thanh công cụ (zoom, pan, download)
                },
                zoom: {
                    enabled: true, // Cho phép zoom
                    type: 'x',      // Zoom theo trục X
                    autoScaleYaxis: true // Tự động điều chỉnh trục Y khi zoom X
                },
                events: {
                    // Xử lý click vào legend để đồng bộ ẩn/hiện series chính và connector
                    legendClick: function(chartContext, seriesIndex, config) {
                        try {
                             // seriesName có thể là undefined nếu config không đúng -> thêm kiểm tra
                             if (typeof seriesIndex !== 'number' || seriesIndex < 0) {
                                console.error("legendClick nhận được seriesIndex không hợp lệ:", seriesIndex);
                                return; // Thoát hàm nếu chỉ số không hợp lệ
                            }
                             const seriesName = config?.globals?.seriesNames?.[seriesIndex];
                             const allSeriesNames = config?.globals?.seriesNames;

                             if (!seriesName || !Array.isArray(allSeriesNames)) {
                                 console.error("Could not get series name or all series names from legend click config.");
                                 return; // Thoát nếu không lấy được thông tin cần thiết
                             }

                             // 1. Toggle series được click
                             chartContext.toggleSeries(seriesName);

                             // 2. Nếu là prediction chính, toggle connector tương ứng
                             if (!isHistory(seriesName) && !isConnector(seriesName)) {
                                 const connectorName = `${seriesName}_connector`;
                                 if (allSeriesNames.includes(connectorName)) {
                                     chartContext.toggleSeries(connectorName);
                                 }
                             }
                        } catch (error) {
                           console.error("Error in legendClick handler:", error);
                        }
                    }
                }
            },
            colors: chartColors, // Mảng màu đã tính toán
            dataLabels: {
                enabled: false // Tắt nhãn dữ liệu trên các điểm
            },
            grid: {
                show: true,
                strokeDashArray: 3, // Nét đứt cho đường lưới
                borderColor: "#e0e0e0", // Màu đường lưới
                padding: {
                    right: gridPaddingRight, // Padding bên phải (đã tính toán)
                    left: 15,
                    top: 5,
                    bottom: 5
                }
            },
            stroke: {
                curve: "smooth", // Đường cong mượt
                width: chartType === 'bar' ? 0 : 2, // Độ dày đường (0 cho bar chart)
                dashArray: currentChartData.map(s => isHistory(s.name) ? 0 : 5), // Nét đứt cho prediction/connector
            },
            xaxis: {
                type: "datetime", // Loại trục X là thời gian
                axisBorder: { show: false }, // Ẩn đường viền trục X
                axisTicks: { show: false }, // Ẩn các dấu tick trên trục X
                labels: {
                    style: { fontWeight: '500', colors: "#333" }, // Style cho nhãn trục X
                    format: 'dd MMM yy', // Định dạng ngày tháng
                    datetimeUTC: false, // Sử dụng giờ địa phương, không phải UTC
                    trim: true, // Cắt bớt nhãn nếu quá dài
                    hideOverlappingLabels: true, // Ẩn nhãn nếu chúng chồng chéo
                    rotate: 0, // Không xoay nhãn
                    maxHeight: 40, // Chiều cao tối đa cho nhãn (tránh chồng vào biểu đồ)
                },
                tickPlacement: 'on', // Đặt các tick trùng với điểm dữ liệu (thay vì giữa)
                tooltip: { // Tooltip riêng cho trục X khi hover
                     enabled: true,
                     formatter: function (val) {
                         try {
                             const timestamp = typeof val === 'string' ? parseInt(val, 10) : val;
                             if (!isNaN(timestamp)) {
                                 return format(new Date(Number(timestamp)), 'dd/MM/yyyy'); // Định dạng ngày trong tooltip trục X
                             }
                             return '';
                         } catch { return ''; }
                     },
                 }
            },
            yaxis: {
                min: 0, // Trục Y bắt đầu từ 0
                title: {
                    // Thêm tiêu đề cho trục Y (bao gồm đơn vị nếu có)
                    text: currentUnit ? `Giá trị (${currentUnit})` : "Giá trị",
                    style: {
                        fontSize: '10px',
                        // color: '#666'
                    }
                },
                labels: {
                    style: { fontWeight: '500', colors: "#333" }, // Style nhãn trục Y
                    formatter: (value) => { // Định dạng số trên trục Y
                        if (value === null || value === undefined || isNaN(value)) return "";
                        return value.toFixed(2); // Hiển thị 2 chữ số thập phân
                    }
                },
            },
            legend: {
                 show: true, // Hiển thị legend
                 position: 'bottom', // Vị trí ở dưới
                 horizontalAlign: 'center', // Căn giữa
                 fontWeight: 500,
                 offsetY: 5, // Khoảng cách từ biểu đồ
                 itemMargin: { horizontal: 10, vertical: 5 }, // Khoảng cách giữa các mục legend
                 labels: {
                     colors: "#1b1c1c", // Màu chữ legend
                     useSeriesColors: false, // Không tự động dùng màu series cho chữ (vì đã có marker)
                 },
                 markers: { /* Giữ trống, không ghi đè marker mặc định */ },
                 formatter: function(seriesName, opts) {
                    if (isConnector(seriesName)) {
                        return ""; // Ẩn connector
                    }
                    // Tra cứu tên hiển thị, nếu không có thì dùng tên gốc
                    return SERIES_DISPLAY_NAMES[seriesName] || seriesName;
                },
                 onItemClick: {
                     toggleDataSeries: false // Tắt hành vi toggle mặc định
                 },
                 onItemHover: {
                     highlightDataSeries: true // Highlight series khi hover legend
                 },
            },
            tooltip: {
    theme: "light",
    shared: true,
    intersect: false,
    custom: function({ series, seriesIndex, dataPointIndex, w }) {
        // console.log("--- Tooltip Custom Fired ---", { seriesIndex, dataPointIndex }); // Có thể giữ lại log này

        // --- KIỂM TRA AN TOÀN NÂNG CAO ---
        if (!w || !w.globals || dataPointIndex < 0 || seriesIndex < 0 || !w.globals.seriesX) {
             console.log("Tooltip Exit: Early globals check failed"); return '';
        }
        // Quan trọng: Kiểm tra seriesNames vì sẽ dùng length của nó
        if (!Array.isArray(w.globals.seriesNames) ||
            !Array.isArray(w.globals.collapsedSeriesIndices) ||
            !Array.isArray(w.globals.colors) ||
            !Array.isArray(series)
           ) {
             console.log("Tooltip Exit: Required arrays check failed (including seriesNames)"); return '';
        }
        // *** KHÔNG CÒN KIỂM TRA w.globals.seriesIndices Ở ĐÂY ***
        // --- KẾT THÚC KIỂM TRA AN TOÀN ---


        const hoveredTimestamp = w.globals.seriesX?.[seriesIndex]?.[dataPointIndex];
        // console.log(" Hovered Timestamp:", hoveredTimestamp);
        if (hoveredTimestamp === undefined || hoveredTimestamp === null) {
            console.log("Tooltip Exit: Invalid hovered timestamp"); return '';
        }

        let headerDate = '';
        try {
             headerDate = format(new Date(hoveredTimestamp), 'dd/MM/yyyy');
             // console.log(" Header Date:", headerDate);
        } catch (e) {
             console.error("Lỗi khi format ngày:", hoveredTimestamp, e);
             return '';
        }


        let htmlContent = `<div class="apexcharts-tooltip-title" style="font-family: Helvetica, Arial, sans-serif; font-size: 12px; padding: 5px 10px; background: #f3f3f3; border-bottom: 1px solid #e3e3e3;">
                               ${headerDate}
                           </div>
                           <div class="apexcharts-tooltip-series-group apexcharts-active" style="padding: 5px 10px; display: flex; flex-direction: column;">`;


        let seriesHtml = '';
        let visibleSeriesFound = false;

        // --- *** LẶP DỰA TRÊN ĐỘ DÀI CỦA seriesNames *** ---
        const numberOfSeries = w.globals.seriesNames.length;
        // console.log(`Iterating based on numberOfSeries: ${numberOfSeries}`);

        for (let i = 0; i < numberOfSeries; i++) { // Dùng vòng lặp for
            // console.log(` Processing Index ${i}`);

            if (!w.globals.seriesNames[i]) { /*console.log(`  Skipping ${i}: No name`);*/ continue; }
            if (w.globals.collapsedSeriesIndices.includes(i)) { /*console.log(`  Skipping ${i}: Collapsed`);*/ continue; }

            const currentSeriesName = w.globals.seriesNames[i];
            if (isConnector(currentSeriesName)) { /*console.log(`  Skipping ${i}: Connector`);*/ continue; }

            let valueForThisSeries = null;
            let pointIndexInSeriesI = -1;
            if (w.globals.seriesX && Array.isArray(w.globals.seriesX[i])) {
                pointIndexInSeriesI = w.globals.seriesX[i].findIndex((ts: any) => ts === hoveredTimestamp);
            } else {
                /*console.log(`  Skipping value search for ${i}: seriesX[${i}] is not an array`);*/
                 continue;
            }

            // console.log(`  Found index ${pointIndexInSeriesI} in series ${i} for timestamp ${hoveredTimestamp}`);

            if (pointIndexInSeriesI !== -1 && Array.isArray(series[i]) && series[i][pointIndexInSeriesI] !== undefined) {
                valueForThisSeries = series[i][pointIndexInSeriesI];
            } else {
                 valueForThisSeries = null;
            }
            // console.log(`  Value for series ${i}: ${valueForThisSeries}`);

            if (valueForThisSeries !== null && !isNaN(valueForThisSeries)) {
                 const color = Array.isArray(w.globals.colors) && w.globals.colors[i] ? w.globals.colors[i] : '#000';
                 const baseName = getBaseName(currentSeriesName) || currentSeriesName;
                 const displayName = SERIES_DISPLAY_NAMES[baseName] || baseName;
                 const valStr = typeof valueForThisSeries === 'number' ? valueForThisSeries.toFixed(4) : 'N/A';

                 seriesHtml += `<div class="apexcharts-tooltip-series-item" style="display:flex; align-items:center; padding: 3px 0;">
                                    <span class="apexcharts-tooltip-marker" style="background-color:${color}; width:10px; height:10px; border-radius:50%; margin-right:6px; flex-shrink: 0;"></span>
                                    <span style="font-family: Helvetica, Arial, sans-serif; font-size: 12px; color: #505050;">${displayName}:&nbsp;<strong>${valStr}</strong></span>
                               </div>`;
                 visibleSeriesFound = true;
            } else {
                /*console.log(`  Skipping HTML for series ${i}: Value is null or NaN`);*/
            }
        } // --- *** KẾT THÚC VÒNG LẶP for *** ---

        // console.log("Loop finished. Found visible series:", visibleSeriesFound);

        if (!visibleSeriesFound) {
            console.log("Tooltip Exit: No valid series data found to display for this point");
            return '';
        }

        htmlContent += seriesHtml;
        htmlContent += `</div>`;
        // console.log("Returning HTML content.");
        return htmlContent;
    } // Kết thúc custom
},
            fill: finalFillOptions, // Áp dụng tùy chọn fill đã tính (cho area/bar)
            plotOptions: {
                bar: { // Tùy chọn riêng cho bar chart
                    horizontal: false,
                    columnWidth: '60%', // Độ rộng cột
                    borderRadius: 4 // Bo góc cột
                }
            },
            markers: { // Tùy chọn cho marker trên các điểm dữ liệu của line/area chart
                size: markerSizes, // Kích thước marker (connector = 0)
                strokeWidth: 0, // Không có viền cho marker
                hover: {
                    sizeOffset: 3 // Tăng kích thước marker khi hover
                },
            },
        };
    }, [chartData, isFullscreen, chartType, predictWeeks, title, selectedFeature, getBaseName, isConnector, isHistory]);

    // Render Logic
    const hasHistoricalData = useMemo(() => chartData?.some(s => isHistory(s.name) && s.data && s.data.length > 0) ?? false, [chartData, isHistory]);
    const hasAnyPredictionDataInChart = useMemo(() => chartData?.some(s => !isHistory(s.name) && !isConnector(s.name) && s.data && s.data.length > 0) ?? false, [chartData, isHistory, isConnector]);
    const hasRawPredictionData = useMemo(() => groupedPredictionDataPoints && groupedPredictionDataPoints.size > 0, [groupedPredictionDataPoints]);
    const canRenderChart = isClient && (hasHistoricalData || (predictMode && hasAnyPredictionDataInChart));

    // No Data Message
    const noDataMessage = useMemo(() => {
        if (!isClient) return "Đang tải biểu đồ...";
        if (!hasHistoricalData && !hasRawPredictionData) return "Không có dữ liệu lịch sử và dự đoán.";
        if (!hasHistoricalData && !predictMode) return "Không có dữ liệu lịch sử.";
        if (predictMode && !hasAnyPredictionDataInChart && hasHistoricalData) return "Không có điểm dự đoán phù hợp sau thời điểm lịch sử cuối cùng.";
        if (!canRenderChart) return "Không có dữ liệu phù hợp để hiển thị.";
        return '';
    }, [isClient, hasHistoricalData, hasRawPredictionData, predictMode, hasAnyPredictionDataInChart, canRenderChart]);

    // --- Tạo options động cho Select theo Tuần ---
    const weekOptions = useMemo(() => {
        if (maxAvailablePredictWeeks <= 0) return [];
        return Array.from({ length: maxAvailablePredictWeeks }, (_, i) => i + 1);
    }, [maxAvailablePredictWeeks]);


    return (
        <div
            className={`rounded-lg shadow-md bg-white dark:bg-gray-800 p-4 sm:p-6 relative w-full break-words transition-all duration-300 ease-in-out ${
                isFullscreen ? "fixed inset-0 overflow-auto bg-white dark:bg-gray-800" : ""
            }`}
        >
            {/* Header */}
             <div className="flex flex-wrap justify-between items-center gap-y-3 gap-x-4 mb-4">
                 <h5 className={`card-title text-base sm:text-lg font-semibold ${isFullscreen ? 'text-xl' : ''} text-gray-800 dark:text-gray-100`}>
                     Xu hướng {title}
                 </h5>
                 <div className="flex flex-wrap justify-end items-center gap-y-3 gap-x-4">
                     {/* Select Chart Type */}
                     <div className="flex items-center gap-2">
                         <Label htmlFor="chart-type-select" className="text-sm font-medium hidden sm:inline text-gray-700 dark:text-gray-300">Loại:</Label>
                         <Select value={chartType} onValueChange={(value: ChartType) => setChartType(value)}>
                             <SelectTrigger id="chart-type-select" className="h-8 w-auto px-3 text-xs sm:text-sm bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100">
                                 <SelectValue placeholder="Loại biểu đồ" />
                             </SelectTrigger>
                             <SelectContent className="min-w-[8rem] text-xs sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                                 <SelectItem value="line"><div className="flex items-center gap-2"><LineChart size={14} /> Đường</div></SelectItem>
                                 <SelectItem value="bar"><div className="flex items-center gap-2"><BarChart size={14} /> Cột</div></SelectItem>
                                 <SelectItem value="area"><div className="flex items-center gap-2"><AreaChart size={14} /> Miền</div></SelectItem>
                             </SelectContent>
                         </Select>
                     </div>
                     {/* Switch Predict Mode */}
                      <div className="flex items-center gap-2">
                          <Label htmlFor="predict-mode" className={`text-sm select-none ${!hasRawPredictionData ? 'cursor-not-allowed text-gray-400 dark:text-gray-500' : 'cursor-pointer text-gray-700 dark:text-gray-300'}`}>
                              Dự đoán
                          </Label>
                          <Switch
                              checked={predictMode}
                              onCheckedChange={setPredictMode}
                              id="predict-mode"
                              disabled={!hasRawPredictionData}
                              className="data-[state=unchecked]:bg-gray-300 dark:data-[state=unchecked]:bg-gray-600 data-[state=checked]:bg-[#FF4560]"
                              aria-label="Bật/tắt chế độ dự đoán"
                          />
                      </div>
                     {/* Select Weeks (Dynamic Options) */}
                      {hasRawPredictionData && maxAvailablePredictWeeks > 0 && (
                          <div className={`flex items-center gap-2 transition-opacity duration-300 ${predictMode ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
                              <Label htmlFor="predict-weeks" className="text-sm font-medium hidden sm:inline text-gray-700 dark:text-gray-300">Xem:</Label>
                              <Select
                                  value={predictWeeks.toString()} // Sử dụng predictWeeks
                                  onValueChange={(value) => {
                                      const selected = parseInt(value, 10);
                                      if (!isNaN(selected)) {
                                          setPredictWeeks(selected); // Sử dụng setPredictWeeks
                                      }
                                  }}
                                  disabled={!predictMode || weekOptions.length === 0}
                              >
                                  <SelectTrigger
                                      id="predict-weeks"
                                      className="h-8 w-[7rem] px-3 text-xs sm:text-sm bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100"
                                      disabled={!predictMode || weekOptions.length === 0}
                                      aria-label="Chọn số tuần dự đoán"
                                  >
                                     <SelectValue placeholder={weekOptions.length > 0 ? "Số tuần" : "Không có"} />
                                  </SelectTrigger>
                                  <SelectContent className="min-w-[7rem] text-xs sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                                      {weekOptions.map((weeks) => (
                                          <SelectItem key={weeks} value={weeks.toString()}>{`${weeks} Tuần`}</SelectItem>
                                      ))}
                                      {weekOptions.length === 0 && (
                                          <div className="px-4 py-2 text-sm text-gray-500 italic">Không có dữ liệu dự đoán</div>
                                      )}
                                  </SelectContent>
                              </Select>
                          </div>
                      )}
                     {/* Fullscreen Button */}
                      <Button variant="outline" size="icon" onClick={toggleFullscreen} className="h-8 w-8 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700">
                          {isFullscreen ? <Minimize size={14} /> : <Maximize size={14} />}
                          <span className="sr-only">{isFullscreen ? "Thoát toàn màn hình" : "Xem toàn màn hình"}</span>
                      </Button>
                 </div>
            </div>

            {/* Chart Area */}
             <div className={`relative mt-2 ${isFullscreen ? 'h-[calc(100vh-120px)]' : 'h-[300px] sm:h-[315px]'}`}>
                 {canRenderChart ? (
                     <Chart
                         key={chartOptions.chart?.id + '-' + chartData.length}
                         options={chartOptions}
                         series={chartData ?? []}
                         type={chartType}
                         height="100%"
                         width="100%"
                     />
                 ) : (
                      <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400 italic px-4 text-center">
                          {noDataMessage}
                      </div>
                 )}
            </div>
        </div>
    );
};

export default Chartline;