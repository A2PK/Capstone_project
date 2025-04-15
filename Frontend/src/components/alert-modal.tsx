import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
    DialogClose,
    DialogDescription // Thêm DialogDescription nếu muốn có mô tả
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from '@/lib/utils'; // Import cn nếu cần kết hợp class

// Định nghĩa kiểu cho cấu hình cảnh báo của một metric
export interface MetricAlertConfig {
    min: number | null;
    max: number | null;
}

// Định nghĩa kiểu cho toàn bộ cấu hình cảnh báo
export interface AlertConfiguration {
    ph: MetricAlertConfig;
    ec: MetricAlertConfig;
    do: MetricAlertConfig;
    nh4: MetricAlertConfig;
    no2: MetricAlertConfig;
    po4: MetricAlertConfig;
    tss: MetricAlertConfig;
    cod: MetricAlertConfig;
    ah: MetricAlertConfig;
}

// Tên các metric có thể cấu hình
export const alertableMetrics: (keyof AlertConfiguration)[] = [
    'ph', 'ec', 'do', 'nh4', 'no2', 'po4', 'tss', 'cod','ah'
];

// Props cho component modal
interface AlertConfigModalProps {
    isOpen: boolean;
    onClose: () => void;
    initialConfig: AlertConfiguration;
    onSave: (config: AlertConfiguration) => void;
}

export function AlertConfigModal({ isOpen, onClose, initialConfig, onSave }: AlertConfigModalProps) {
    const [localConfig, setLocalConfig] = useState<AlertConfiguration>(initialConfig);

    // Cập nhật state nội bộ khi prop initialConfig thay đổi (quan trọng khi mở lại modal)
    useEffect(() => {
        if (isOpen) {
            setLocalConfig(initialConfig);
        }
    }, [initialConfig, isOpen]);

    const handleInputChange = (metric: keyof AlertConfiguration, type: 'min' | 'max', value: string) => {
        const numericValue = value === '' ? null : parseFloat(value);
        // Đảm bảo không gán NaN
        const finalValue = isNaN(numericValue as number) ? null : numericValue;

        setLocalConfig(prevConfig => ({
            ...prevConfig,
            [metric]: {
                ...prevConfig[metric],
                [type]: finalValue,
            }
        }));
    };

    const handleSave = () => {
        onSave(localConfig);
        onClose();
    };

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            {/* Tăng chiều rộng modal: sm:max-w-xl, md:max-w-2xl, lg:max-w-3xl */}
            <DialogContent className="sm:max-w-2xl md:max-w-3xl">
                <DialogHeader className="pb-4 border-b mb-4"> {/* Thêm border và margin */}
                    <DialogTitle className="text-xl font-bold text-blue-800"> {/* Tăng kích thước, đậm, màu */}
                        Thiết Lập Miền Giá Trị Cảnh Báo
                    </DialogTitle>
                     {/* <DialogDescription>
                       Đặt giới hạn dưới (Min) và giới hạn trên (Max) cho các chỉ số. Các giá trị nằm ngoài khoảng này sẽ được cảnh báo.
                    </DialogDescription> */}
                </DialogHeader>

                {/* Phần nội dung form */}
                <div className="grid gap-y-6 py-4 max-h-[65vh] overflow-y-auto pr-4 -mr-2"> {/* Tăng gap-y, max-h và thêm padding right */}
                    {alertableMetrics.map(metric => (
                        <div key={metric} className="space-y-3 border-b pb-4 last:border-b-0">
                            {/* Label chính của Metric */}
                            <Label className="text-base font-bold uppercase text-gray-900 block"> {/* Căn trái, đậm, màu */}
                                {metric}
                            </Label>

                            {/* Nhóm Input Min và Max */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3"> {/* Grid 2 cột trên màn hình vừa và lớn */}
                                {/* Input Min */}
                                <div className="space-y-1.5">
                                    <Label htmlFor={`${metric}-min`} className="text-sm font-medium text-gray-600">
                                        Giá trị nhỏ nhất (Min)
                                    </Label>
                                    <Input
                                        id={`${metric}-min`}
                                        type="number"
                                        step="any" // Cho phép số thập phân
                                        value={localConfig[metric]?.min ?? ''}
                                        onChange={(e) => handleInputChange(metric, 'min', e.target.value)}
                                        className="w-full border-gray-300 focus:border-blue-500 focus:ring-blue-500" // Style input
                                        placeholder="Để trống = không giới hạn" // Placeholder rõ hơn
                                    />
                                </div>

                                {/* Input Max */}
                                <div className="space-y-1.5">
                                    <Label htmlFor={`${metric}-max`} className="text-sm font-medium text-gray-600">
                                        Giá trị lớn nhất (Max)
                                    </Label>
                                    <Input
                                        id={`${metric}-max`}
                                        type="number"
                                        step="any"
                                        value={localConfig[metric]?.max ?? ''}
                                        onChange={(e) => handleInputChange(metric, 'max', e.target.value)}
                                        className="w-full border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                                        placeholder="Để trống = không giới hạn"
                                    />
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Phần Footer với nút bấm */}
                <DialogFooter className="mt-6 pt-4 border-t"> {/* Thêm khoảng cách và border */}
                    <DialogClose asChild>
                        <Button type="button" variant="outline">Hủy</Button>
                    </DialogClose>
                    <Button type="button" onClick={handleSave} className="bg-blue-600 hover:bg-blue-700 text-white">Lưu Thay Đổi</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}