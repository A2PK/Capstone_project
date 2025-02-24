"use client";
import dynamic from "next/dynamic";
import { useMemo, useState, useEffect } from "react";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false, loading: () => <p>Loading chart...</p> });

const Chartline = () => {
  const [predictMode, setPredictMode] = useState(false);
  const [predictRange, setPredictRange] = useState<number>(1);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Fake data for historical and predicted values
  const fakeHistoricalData = Array.from({ length: 20 }, (_, i) => ({
    x: new Date(2024, 0, i + 1).getTime(),
    y: Math.floor(Math.random() * 50) + 50,
  }));

  const fakePredictionData = Array.from({ length: 10 }, (_, i) => ({
    x: new Date(2024, 0, i + 21).getTime(),
    y: Math.floor(Math.random() * 50) + 50,
  }));

  // Ensure chart data structure is correct
  const historicalData = {
    name: "Historical",
    data: fakeHistoricalData,
  };

  const predictionData = {
    name: "Prediction",
    data: fakePredictionData,
  };

  const chartData = useMemo(() => {
    const data = [historicalData];
    if (predictMode) {
      let processedPredictionData = predictionData.data.slice(0, predictRange);
      if (historicalData.data.length > 0) {
        processedPredictionData.unshift(historicalData.data[historicalData.data.length - 1]);
      }
      data.push({
        ...predictionData,
        data: processedPredictionData,
      });
    }
    return data;
  }, [predictMode, predictRange]);

  return (
    <div className="rounded-lg dark:shadow-dark-md shadow-md bg-white dark:bg-darkgray p-6 relative w-full break-words">
      <div className="flex justify-between items-center">
        <h5 className="card-title">Water Monitoring</h5>
        <div className="flex justify-end items-center gap-5">
          <label htmlFor="predict-mode">Prediction Mode</label>
          <Switch
            checked={predictMode}
            onCheckedChange={() => setPredictMode((prev) => !prev)}
            id="predict-mode"
            className="data-[state=unchecked]:bg-gray-300 data-[state=checked]:bg-blue-500"
          />
          <Select disabled={!predictMode} onValueChange={(value) => setPredictRange(parseInt(value))}>
            <SelectTrigger className="h-8 w-32 text-sm"> {/* Adjust size here */}
              <SelectValue placeholder = "Week 1" />
            </SelectTrigger>
            <SelectContent className="w-32 text-sm"> {/* Adjust size here */}
              {Array.from({ length: 10 }, (_, i) => (
                <SelectItem key={i + 1} value={(i + 1).toString()} className="text-sm">{`${i + 1} Week${i + 1 > 1 ? "s" : ""}`}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="-ms-4 -me-3 mt-2">
        {isClient && (
          <Chart
            options={{
              chart: {
                fontFamily: "inherit",
                foreColor: "#adb0bb",
                offsetX: 0,
                offsetY: 10,
                animations: { speed: 500 },
                toolbar: { show: false },
              },
              colors: ["var(--color-primary)", "#a6a6a6"],
              dataLabels: { enabled: false },
              fill: {
                type: "gradient",
                gradient: {
                  shadeIntensity: 0,
                  inverseColors: false,
                  opacityFrom: 0.1,
                  opacityTo: 0.3,
                  stops: [100],
                },
              },
              grid: {
                show: true,
                strokeDashArray: 3,
                borderColor: "#90A4AE50",
              },
              stroke: { curve: "smooth", width: 2, dashArray: [0, 8] },
              xaxis: { axisBorder: { show: false }, axisTicks: { show: false }, type: "datetime" },
              yaxis: { min: 0, tickAmount: 10 },
              legend: { show: true, labels: { colors: "#adb0bb" } },
              tooltip: { theme: "dark", x: { format: "dd MMM yyyy" } },
            }}
            series={chartData}
            type="area"
            height="315px"
            width="100%"
          />
        )}
      </div>
    </div>
  );
};

export default Chartline;
