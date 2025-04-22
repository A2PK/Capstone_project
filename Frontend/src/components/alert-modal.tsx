"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogClose,
  DialogDescription,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Activity, Droplets, FlaskRound, Zap, BarChart3, Waves, Beaker, AlertCircle, HelpCircle } from "lucide-react"

// Định nghĩa kiểu cho cấu hình cảnh báo của một metric
export interface MetricAlertConfig {
  min: number | null
  max: number | null
}

// Định nghĩa kiểu cho toàn bộ cấu hình cảnh báo
export interface AlertConfiguration {
  ph: MetricAlertConfig
  ec: MetricAlertConfig
  do: MetricAlertConfig
  nh4: MetricAlertConfig
  no2: MetricAlertConfig
  po4: MetricAlertConfig
  tss: MetricAlertConfig
  cod: MetricAlertConfig
  ah: MetricAlertConfig
}

// Tên các metric có thể cấu hình
export const alertableMetrics: (keyof AlertConfiguration)[] = [
  "ph",
  "ec",
  "do",
  "nh4",
  "no2",
  "po4",
  "tss",
  "cod",
  "ah",
]

// Thông tin mô tả cho từng metric
const metricInfo: Record<
  keyof AlertConfiguration,
  {
    label: string
    description: string
    icon: React.ReactNode
    color: string
  }
> = {
  ph: {
    label: "pH",
    description: "Độ pH đo mức độ axit hoặc kiềm của nước",
    icon: <FlaskRound className="h-5 w-5" />,
    color: "bg-purple-100 text-purple-700",
  },
  ec: {
    label: "EC",
    description: "Độ dẫn điện, đo lượng muối hòa tan trong nước",
    icon: <Zap className="h-5 w-5" />,
    color: "bg-yellow-100 text-yellow-700",
  },
  do: {
    label: "DO",
    description: "Oxy hòa tan, đo lượng oxy có trong nước",
    icon: <Droplets className="h-5 w-5" />,
    color: "bg-blue-100 text-blue-700",
  },
  nh4: {
    label: "NH4",
    description: "Amoni, chỉ số đo lượng amoni trong nước",
    icon: <Beaker className="h-5 w-5" />,
    color: "bg-green-100 text-green-700",
  },
  no2: {
    label: "NO2",
    description: "Nitrit, chỉ số đo lượng nitrit trong nước",
    icon: <Beaker className="h-5 w-5" />,
    color: "bg-teal-100 text-teal-700",
  },
  po4: {
    label: "PO4",
    description: "Phosphat, chỉ số đo lượng phosphat trong nước",
    icon: <Beaker className="h-5 w-5" />,
    color: "bg-indigo-100 text-indigo-700",
  },
  tss: {
    label: "TSS",
    description: "Tổng chất rắn lơ lửng trong nước",
    icon: <Waves className="h-5 w-5" />,
    color: "bg-orange-100 text-orange-700",
  },
  cod: {
    label: "COD",
    description: "Nhu cầu oxy hóa học, đo lượng oxy cần thiết để oxy hóa các chất hữu cơ",
    icon: <Activity className="h-5 w-5" />,
    color: "bg-red-100 text-red-700",
  },
  ah: {
    label: "AH",
    description: "Độ kiềm tổng, đo khả năng đệm của nước",
    icon: <BarChart3 className="h-5 w-5" />,
    color: "bg-pink-100 text-pink-700",
  },
}

// Props cho component modal
interface AlertConfigModalProps {
  isOpen: boolean
  onClose: () => void
  initialConfig: AlertConfiguration
  onSave: (config: AlertConfiguration) => void
}

export function AlertConfigModal({ isOpen, onClose, initialConfig, onSave }: AlertConfigModalProps) {
  const [localConfig, setLocalConfig] = useState<AlertConfiguration>(initialConfig)

  // Cập nhật state nội bộ khi prop initialConfig thay đổi (quan trọng khi mở lại modal)
  useEffect(() => {
    if (isOpen) {
      setLocalConfig(initialConfig)
    }
  }, [initialConfig, isOpen])

  const handleInputChange = (metric: keyof AlertConfiguration, type: "min" | "max", value: string) => {
    const numericValue = value === "" ? null : Number.parseFloat(value)
    // Đảm bảo không gán NaN
    const finalValue = isNaN(numericValue as number) || (numericValue as number < 0) ? null : numericValue

    setLocalConfig((prevConfig) => ({
      ...prevConfig,
      [metric]: {
        ...prevConfig[metric],
        [type]: finalValue,
      },
    }))
  }

  const handleSave = () => {
    onSave(localConfig)
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-2xl md:max-w-3xl">
        <DialogHeader className="pb-4 border-b mb-4">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-6 w-6 text-blue-600" />
            <DialogTitle className="text-xl font-bold text-blue-600">Thiết Lập Miền Giá Trị</DialogTitle>
          </div>
          <DialogDescription className="mt-2 text-gray-600">
            Đặt giới hạn dưới (Min) và giới hạn trên (Max) cho các chỉ số. Hệ thống sẽ cảnh báo khi giá trị nằm ngoài
            khoảng này.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-y-6 py-4 max-h-[65vh] overflow-y-auto pr-4 -mr-2">
          {alertableMetrics.map((metric) => (
            <div key={metric} className="space-y-3 border-b pb-6 last:border-b-0">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={cn("p-2 rounded-lg", metricInfo[metric].color)}>{metricInfo[metric].icon}</div>
                  <Label className="text-base font-bold uppercase text-gray-900 block">
                    {metricInfo[metric].label || metric}
                  </Label>
                </div>

                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <HelpCircle className="h-4 w-4 text-gray-500" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p>{metricInfo[metric].description}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
                <div className="space-y-1.5">
                  <Label htmlFor={`${metric}-min`} className="text-sm font-medium text-gray-600">
                    Giá trị nhỏ nhất (Min)
                  </Label>
                  <Input
                    id={`${metric}-min`}
                    type="number"
                    step="any"
                    min="0"
                    value={localConfig[metric]?.min ?? ""}
                    onChange={(e) => handleInputChange(metric, "min", e.target.value)}
                    className="w-full border-gray-300 focus:border-blue-500 focus:ring-blue-500 transition-all"
                    placeholder="Để trống = không giới hạn"
                  />
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor={`${metric}-max`} className="text-sm font-medium text-gray-600">
                    Giá trị lớn nhất (Max)
                  </Label>
                  <Input
                    id={`${metric}-max`}
                    type="number"
                    min="0"
                    step="any"
                    value={localConfig[metric]?.max ?? ""}
                    onChange={(e) => handleInputChange(metric, "max", e.target.value)}
                    className="w-full border-gray-300 focus:border-blue-500 focus:ring-blue-500 transition-all"
                    placeholder="Để trống = không giới hạn"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        <DialogFooter className="mt-6 pt-4 border-t flex justify-between sm:justify-end gap-2">
          <DialogClose asChild>
            <Button type="button" variant="outline" className="border-gray-300 hover:bg-gray-100 transition-colors">
              Hủy
            </Button>
          </DialogClose>
          <Button
            type="button"
            onClick={handleSave}
            className="bg-blue-600 hover:bg-blue-700 text-white transition-colors"
          >
            Lưu Thay Đổi
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
