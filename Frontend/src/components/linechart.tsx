// File: src/components/linechart.tsx
"use client";

import dynamic from "next/dynamic";
import { useMemo, useState, useEffect, useCallback } from "react";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Maximize, Minimize, LineChart, BarChart, AreaChart } from 'lucide-react';
import type { ApexOptions } from "apexcharts";
import type { DataPoint } from "@/types/station2"; // Ensure this path is correct
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

// Helper function to get value (defined outside component, stable)
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

// Define SERIES_DISPLAY_NAMES outside the component if it's truly constant
const SERIES_DISPLAY_NAMES: { [key: string]: string } = {
    'Lịch sử': 'Lịch sử',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'ETSformer':'ETSformer',
    'ETSformerPar':'ETSformer Parallel',
};

// Type definitions for ApexCharts series data
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
    const [predictWeeks, setPredictWeeks] = useState<number>(1);
    const [isClient, setIsClient] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [chartType, setChartType] = useState<ChartType>("line");

    // --- Memoized Helper Functions ---
    const getBaseName = useCallback((name: string | undefined): string | undefined => {
        if (!name) return undefined;
        return name.replace(/_connector$/, '');
    }, []);

    const isConnector = useCallback((name: string | undefined): boolean => {
        return !!name && name.endsWith('_connector');
    }, []);

    const isHistory = useCallback((name: string | undefined): boolean => {
        return name === 'Lịch sử';
    }, []);

    // Effect for client-side check and fullscreen escape handler
    useEffect(() => {
        setIsClient(true);
        const handleEsc = (event: KeyboardEvent) => {
            if (event.key === 'Escape' && isFullscreen) setIsFullscreen(false);
        };
        document.addEventListener('keydown', handleEsc);
        return () => document.removeEventListener('keydown', handleEsc);
    }, [isFullscreen]); // isFullscreen is the dependency here

    // Calculate max available prediction weeks
    const maxAvailablePredictWeeks = useMemo((): number => {
        let maxWeeks = 0;
        if (!groupedPredictionDataPoints || groupedPredictionDataPoints.size === 0) {
            return 0;
        }
        const sortedHistory = [...historicalDataPoints].sort((a, b) => new Date(a.monitoringTime).getTime() - new Date(b.monitoringTime).getTime());
        const lastHistoricalTimestamp = sortedHistory.length > 0
            ? new Date(sortedHistory[sortedHistory.length - 1].monitoringTime).getTime()
            : undefined;

        groupedPredictionDataPoints.forEach((predictionPoints) => {
            if (!Array.isArray(predictionPoints)) return;
            let futurePointsCount = 0;
            predictionPoints.forEach(dp => {
                try {
                    const timestamp = new Date(dp.monitoringTime).getTime();
                    if (!isNaN(timestamp) && (lastHistoricalTimestamp === undefined || timestamp > lastHistoricalTimestamp)) {
                        futurePointsCount++;
                    }
                } catch (e) {
                    console.error("Error processing timestamp for max week calculation:", dp.monitoringTime, e);
                }
            });
            maxWeeks = Math.max(maxWeeks, futurePointsCount);
        });
        return maxWeeks;
    }, [groupedPredictionDataPoints, historicalDataPoints]);

    // Effect to adjust selected prediction weeks if needed
    useEffect(() => {
        if (maxAvailablePredictWeeks > 0 && predictWeeks > maxAvailablePredictWeeks) {
            setPredictWeeks(maxAvailablePredictWeeks);
        } else if (maxAvailablePredictWeeks === 0 && predictWeeks !== 1) {
             setPredictWeeks(1);
        } else if (predictWeeks < 1) {
            setPredictWeeks(1);
        }
    }, [maxAvailablePredictWeeks, predictWeeks]);

    // --- Memoized Data Preparation ---
    const chartData = useMemo((): ApexChartSeries[] => {
        const series: ApexChartSeries[] = [];
        const connectorSeries: ApexChartSeries[] = [];

        const historicalSeriesData: ApexChartSeriesDataElement[] = [];
        historicalDataPoints.forEach(dp => {
            try {
                const timestamp = new Date(dp.monitoringTime).getTime();
                if (!isNaN(timestamp)) {
                    // getPointValue is stable, defined outside
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

        if (predictMode && groupedPredictionDataPoints && groupedPredictionDataPoints.size > 0) {
            groupedPredictionDataPoints.forEach((predictionPoints, source) => {
                const predictionSourceSeriesData: ApexChartSeriesDataElement[] = [];
                if (Array.isArray(predictionPoints)) {
                    predictionPoints.forEach(dp => {
                        try {
                            const timestamp = new Date(dp.monitoringTime).getTime();
                            if (!isNaN(timestamp) && (lastHistoricalTimestamp === undefined || timestamp > lastHistoricalTimestamp)) {
                                // getPointValue is stable
                                predictionSourceSeriesData.push({ x: timestamp, y: getPointValue(dp, selectedFeature) });
                            }
                        } catch (e) { console.error(`Error processing prediction point for source ${source}:`, dp.monitoringTime, e); }
                    });
                }
                const sortedSourcePredictions = predictionSourceSeriesData.sort((a, b) => a.x - b.x);
                const finalWeeksToShow = sortedSourcePredictions.slice(0, predictWeeks);

                if (finalWeeksToShow.length > 0) {
                    series.push({ name: source, data: finalWeeksToShow });
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
        predictMode, predictWeeks, chartType
        // getPointValue is removed - defined outside, stable
    ]);

    const toggleFullscreen = useCallback(() => setIsFullscreen((prev) => !prev), []);

    // --- Memoized Chart Options ---
    const chartOptions = useMemo((): ApexOptions => {
        const currentChartData = chartData;

        if (!currentChartData || currentChartData.length === 0 || currentChartData.every(s => !s.data || s.data.length === 0)) {
             return {
                 chart: { id: `chart-${title}-${selectedFeature}-nodata`, toolbar: { show: false } },
                 noData: { text: 'Không có dữ liệu phù hợp để hiển thị.' }
             };
        }

        // --- Using memoized helpers ---
        const baseColors = ["#008FFB", "#FF4560", "#00E396", "#FEB019", "#775DD0", "#546E7A", "#D10CE8"];
        const predictionBaseNames = Array.from(new Set(
            currentChartData.map(s => getBaseName(s.name)) // Uses memoized getBaseName
                           .filter((name): name is string => !!name && !isHistory(name)) // Uses memoized isHistory
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
             const baseName = getBaseName(s.name); // Uses memoized getBaseName
             return colorMap.get(baseName || '') || baseColors[baseColors.length - 1];
        });

        const markerSizes = currentChartData.map(s => isConnector(s.name) ? 0 : 3); // Uses memoized isConnector

        let gridPaddingRight = 15;
        if (chartType === 'line' || chartType === 'area') gridPaddingRight = 30;

        let finalFillOptions: ApexOptions['fill'] = {};
        if (chartType === 'area') {
            const baseOpacityFrom = 0.45;
            const baseOpacityTo = 0.35;
            const stepDown = 0.05;
            const predictionIndexMap = new Map<string, number>();
            predictionBaseNames.forEach((name, index) => predictionIndexMap.set(name, index));

            const opacityFrom = currentChartData.map((s: ApexChartSeries) => {
                if (isConnector(s.name)) return 0; // Uses memoized isConnector
                if (isHistory(s.name)) return baseOpacityFrom; // Uses memoized isHistory
                const baseName = getBaseName(s.name); // Uses memoized getBaseName
                const predictIndex = baseName ? predictionIndexMap.get(baseName) : -1;
                return Math.max(0.1, baseOpacityFrom * 0.8 - (predictIndex !== undefined && predictIndex >= 0 ? predictIndex : 0) * stepDown);
            });
            const opacityTo = currentChartData.map((s: ApexChartSeries) => {
                 if (isConnector(s.name)) return 0; // Uses memoized isConnector
                 if (isHistory(s.name)) return baseOpacityTo; // Uses memoized isHistory
                 const baseName = getBaseName(s.name); // Uses memoized getBaseName
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

        const currentUnit = getDonvi(selectedFeature); // getDonvi is likely stable, defined outside

        return {
            chart: {
                id: `chart-${title}-${selectedFeature}`,
                fontFamily: "inherit",
                foreColor: "#adb0bb",
                offsetX: 0,
                offsetY: 0,
                animations: { enabled: true },
                toolbar: { show: true },
                zoom: { enabled: true, type: 'x', autoScaleYaxis: true },
                events: {
                    legendClick: function(chartContext, seriesIndex, config) {
                         try {
                              if (typeof seriesIndex !== 'number' || seriesIndex < 0) return;
                              const seriesName = config?.globals?.seriesNames?.[seriesIndex];
                              const allSeriesNames = config?.globals?.seriesNames;
                              if (!seriesName || !Array.isArray(allSeriesNames)) return;

                              chartContext.toggleSeries(seriesName);
                              // isHistory/isConnector are stable, no need to pass from outside here
                              if (seriesName !== 'Lịch sử' && !seriesName.endsWith('_connector')) {
                                   const connectorName = `${seriesName}_connector`;
                                   if (allSeriesNames.includes(connectorName)) {
                                        chartContext.toggleSeries(connectorName);
                                   }
                              }
                         } catch (error) { console.error("Error in legendClick handler:", error); }
                    }
                }
            },
            colors: chartColors,
            dataLabels: { enabled: false },
            grid: {
                show: true,
                strokeDashArray: 3,
                borderColor: "#e0e0e0",
                padding: { right: gridPaddingRight, left: 15, top: 5, bottom: 5 }
            },
            stroke: {
                curve: "smooth",
                width: chartType === 'bar' ? 0 : 2,
                dashArray: currentChartData.map(s => isHistory(s.name) ? 0 : 5), // Uses memoized isHistory
            },
            xaxis: {
                type: "datetime",
                axisBorder: { show: false },
                axisTicks: { show: false },
                labels: {
                    style: { fontWeight: '500', colors: "#333" },
                    format: 'dd MMM yy',
                    datetimeUTC: false,
                    trim: true,
                    hideOverlappingLabels: true,
                    rotate: 0,
                    maxHeight: 40,
                },
                tickPlacement: 'on',
                tooltip: {
                    enabled: true,
                    formatter: function (val) {
                         try {
                              const timestamp = typeof val === 'string' ? parseInt(val, 10) : val;
                              if (!isNaN(timestamp)) {
                                   return format(new Date(Number(timestamp)), 'dd/MM/yyyy');
                              }
                              return '';
                         } catch { return ''; }
                    },
               }
            },
            yaxis: {
                min: 0,
                title: {
                    text: currentUnit ? `Giá trị (${currentUnit})` : "Giá trị",
                    style: { fontSize: '10px' }
                },
                labels: {
                    style: { fontWeight: '500', colors: "#333" },
                    formatter: (value) => {
                         if (value === null || value === undefined || isNaN(value)) return "";
                         return value.toFixed(2);
                    }
                },
            },
            legend: {
                 show: true,
                 position: 'bottom',
                 horizontalAlign: 'center',
                 fontWeight: 500,
                 offsetY: 5,
                 itemMargin: { horizontal: 10, vertical: 5 },
                 labels: { colors: "#1b1c1c", useSeriesColors: false },
                 markers: { /* Use default */ },
                 formatter: function(seriesName, opts) {
                     if (isConnector(seriesName)) return ""; // Uses memoized isConnector
                     // SERIES_DISPLAY_NAMES is stable, defined outside/top-level
                     return SERIES_DISPLAY_NAMES[seriesName] || seriesName;
                 },
                 onItemClick: { toggleDataSeries: false },
                 onItemHover: { highlightDataSeries: true },
            },
            tooltip: {
                 theme: "light",
                 shared: true,
                 intersect: false,
                 custom: function({ series, seriesIndex, dataPointIndex, w }) {
                    // ... (tooltip logic - ensure it uses stable functions if needed) ...
                    // Inside here, calls to getBaseName, isConnector, isHistory will use memoized versions
                    // SERIES_DISPLAY_NAMES is accessed directly as it's stable
                    // ... (rest of tooltip logic from previous version) ...

                    // Example usage within tooltip (ensure these are used correctly)
                    // const baseName = getBaseName(currentSeriesName); // Uses memoized
                    // if (isConnector(currentSeriesName)) { ... } // Uses memoized
                    // const displayName = SERIES_DISPLAY_NAMES[baseName] || baseName; // Uses stable const

                    // --- RE-INCLUDE FULL TOOLTIP LOGIC HERE (Simplified for brevity) ---
                    if (!w || !w.globals || dataPointIndex < 0 || seriesIndex < 0 || !w.globals.seriesX || !Array.isArray(w.globals.seriesNames)) return '';
                    const hoveredTimestamp = w.globals.seriesX?.[seriesIndex]?.[dataPointIndex];
                    if (hoveredTimestamp === undefined || hoveredTimestamp === null) return '';
                    let headerDate = ''; try { headerDate = format(new Date(hoveredTimestamp), 'dd/MM/yyyy'); } catch { return ''; }
                    let htmlContent = `<div class="apexcharts-tooltip-title" style="font-family: Helvetica, Arial, sans-serif; font-size: 12px; padding: 5px 10px; background: #f3f3f3; border-bottom: 1px solid #e3e3e3;">${headerDate}</div><div class="apexcharts-tooltip-series-group apexcharts-active" style="padding: 5px 10px; display: flex; flex-direction: column;">`;
                    let seriesHtml = ''; let visibleSeriesFound = false; const numberOfSeries = w.globals.seriesNames.length;
                    for (let i = 0; i < numberOfSeries; i++) {
                        if (!w.globals.seriesNames[i] || w.globals.collapsedSeriesIndices.includes(i)) continue;
                        const currentSeriesName = w.globals.seriesNames[i];
                        if (isConnector(currentSeriesName)) continue; // Uses memoized isConnector
                        let valueForThisSeries = null; let pointIndexInSeriesI = -1;
                        if (w.globals.seriesX && Array.isArray(w.globals.seriesX[i])) { pointIndexInSeriesI = w.globals.seriesX[i].findIndex((ts: any) => ts === hoveredTimestamp); } else continue;
                        if (pointIndexInSeriesI !== -1 && Array.isArray(series[i]) && series[i][pointIndexInSeriesI] !== undefined) { valueForThisSeries = series[i][pointIndexInSeriesI]; } else { valueForThisSeries = null; }
                        if (valueForThisSeries !== null && !isNaN(valueForThisSeries)) {
                            const color = Array.isArray(w.globals.colors) && w.globals.colors[i] ? w.globals.colors[i] : '#000';
                            const baseName = getBaseName(currentSeriesName) || currentSeriesName; // Uses memoized getBaseName
                            const displayName = SERIES_DISPLAY_NAMES[baseName] || baseName; // Uses stable const
                            const valStr = typeof valueForThisSeries === 'number' ? valueForThisSeries.toFixed(4) : 'N/A';
                            seriesHtml += `<div class="apexcharts-tooltip-series-item" style="display:flex; align-items:center; padding: 3px 0;"><span class="apexcharts-tooltip-marker" style="background-color:${color}; width:10px; height:10px; border-radius:50%; margin-right:6px; flex-shrink: 0;"></span><span style="font-family: Helvetica, Arial, sans-serif; font-size: 12px; color: #505050;">${displayName}:&nbsp;<strong>${valStr}</strong></span></div>`;
                            visibleSeriesFound = true;
                        }
                    } if (!visibleSeriesFound) return ''; htmlContent += seriesHtml; htmlContent += `</div>`; return htmlContent;
                    // --- END OF RE-INCLUDED TOOLTIP LOGIC ---
                 }
            },
            fill: finalFillOptions,
            plotOptions: {
                bar: { horizontal: false, columnWidth: '60%', borderRadius: 4 }
            },
            markers: {
                size: markerSizes,
                strokeWidth: 0,
                hover: { sizeOffset: 3 },
            },
        };
    }, [
        chartData, isFullscreen, chartType, predictWeeks, title, selectedFeature,
        getBaseName, isConnector, isHistory, // Include memoized helpers
        SERIES_DISPLAY_NAMES // Include the constant
    ]);

    // --- Other useMemo hooks need checking too ---
    const hasHistoricalData = useMemo(() => chartData?.some(s => isHistory(s.name) && s.data && s.data.length > 0) ?? false, [chartData, isHistory]); // Added isHistory dependency
    const hasAnyPredictionDataInChart = useMemo(() => chartData?.some(s => !isHistory(s.name) && !isConnector(s.name) && s.data && s.data.length > 0) ?? false, [chartData, isHistory, isConnector]); // Added isHistory, isConnector dependencies
    const hasRawPredictionData = useMemo(() => groupedPredictionDataPoints && groupedPredictionDataPoints.size > 0, [groupedPredictionDataPoints]);
    const canRenderChart = useMemo(() => isClient && (hasHistoricalData || (predictMode && hasAnyPredictionDataInChart)), [isClient, hasHistoricalData, predictMode, hasAnyPredictionDataInChart]);

    const noDataMessage = useMemo(() => {
        if (!isClient) return "Đang tải biểu đồ...";
        if (!hasHistoricalData && !hasRawPredictionData) return "Không có dữ liệu lịch sử và dự đoán.";
        if (!hasHistoricalData && !predictMode) return "Không có dữ liệu lịch sử.";
        // Ensure dependencies are included if used:
        if (predictMode && !hasAnyPredictionDataInChart && hasHistoricalData) return "Không có điểm dự đoán phù hợp sau thời điểm lịch sử cuối cùng.";
        if (!canRenderChart) return "Không có dữ liệu phù hợp để hiển thị.";
        return '';
    }, [isClient, hasHistoricalData, hasRawPredictionData, predictMode, hasAnyPredictionDataInChart, canRenderChart]); // Dependencies seem correct here

    const weekOptions = useMemo(() => {
        if (maxAvailablePredictWeeks <= 0) return [];
        return Array.from({ length: maxAvailablePredictWeeks }, (_, i) => i + 1);
    }, [maxAvailablePredictWeeks]);

    // --- Render ---
    return (
        <div
            className={`rounded-lg shadow-md bg-white dark:bg-gray-800 p-4 sm:p-6 relative w-full break-words transition-all duration-300 ease-in-out ${
                isFullscreen ? "fixed inset-0 z-50 overflow-auto bg-white dark:bg-gray-800" : "" // Added z-index for fullscreen
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
                                   value={predictWeeks.toString()}
                                   onValueChange={(value) => {
                                       const selected = parseInt(value, 10);
                                       if (!isNaN(selected)) setPredictWeeks(selected);
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
                          key={chartOptions.chart?.id + '-' + chartData.length + '-' + chartType} // Added chartType to key
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