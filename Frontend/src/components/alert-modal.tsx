"use client";

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter,
    DialogClose,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Trash2, PlusCircle, Loader2, AlertCircle, HelpCircle, FlaskRound, Zap, Droplets, Beaker, Waves, Activity,BarChart3 } from 'lucide-react'; // Giữ lại icons cũ + thêm mới
import { ElementRange, CreateElementRangeDto } from '@/types/threshold'; // Sử dụng types mới
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils'; // Dùng cn nếu cần

// Có thể giữ lại metricInfo nếu bạn muốn hiển thị icon/tooltip dựa trên elementName
// Nhưng cần hàm để tìm thông tin dựa trên elementName thay vì key cố định
// Ví dụ:
const defaultMetricInfo = {
    description: "Chỉ số chất lượng nước.",
    icon: <HelpCircle className="h-5 w-5" />,
    color: "bg-gray-100 text-gray-700",
};
const metricInfoLookup: Record<string, { description: string; icon: React.ReactNode; color: string }> = {
    ph: { description: "Độ pH đo mức độ axit hoặc kiềm của nước", icon: <FlaskRound className="h-5 w-5" />, color: "bg-purple-100 text-purple-700" },
    ec: { description: "Độ dẫn điện, đo lượng muối hòa tan trong nước", icon: <Zap className="h-5 w-5" />, color: "bg-yellow-100 text-yellow-700" },
    do: { description: "Oxy hòa tan, đo lượng oxy có trong nước", icon: <Droplets className="h-5 w-5" />, color: "bg-blue-100 text-blue-700" },
    'n-nh4': { description: "Amoni (tính theo Nitơ)", icon: <Beaker className="h-5 w-5" />, color: "bg-green-100 text-green-700" }, // Chú ý key có dấu gạch nối
    'n-no2': { description: "Nitrit (tính theo Nitơ)", icon: <Beaker className="h-5 w-5" />, color: "bg-teal-100 text-teal-700" }, // Chú ý key có dấu gạch nối
    'p-po4': { description: "Phosphat (tính theo Photpho)", icon: <Beaker className="h-5 w-5" />, color: "bg-indigo-100 text-indigo-700" }, // Chú ý key có dấu gạch nối
    tss: { description: "Tổng chất rắn lơ lửng trong nước", icon: <Waves className="h-5 w-5" />, color: "bg-orange-100 text-orange-700" },
    cod: { description: "Nhu cầu oxy hóa học", icon: <Activity className="h-5 w-5" />, color: "bg-red-100 text-red-700" },
    ah: { description: "Độ kiềm tổng", icon: <BarChart3 className="h-5 w-5" />, color: "bg-pink-100 text-pink-700" },
    // Thêm các chỉ số khác nếu cần
};

const getMetricInfo = (elementName: string) => {
    return metricInfoLookup[elementName.toLowerCase()] || defaultMetricInfo;
};

// Props cho component modal (phiên bản mới)
interface AlertConfigModalProps {
    isOpen: boolean;
    onClose: () => void;
    currentThresholds: ElementRange[];
    onUpdate: (updatedConfigs: ElementRange[]) => Promise<void>;
    onCreate: (newConfigData: CreateElementRangeDto) => Promise<void>;
    onDelete: (idToDelete: string, elementName: string) => Promise<void>;
    validElementNames: string[];
    isProcessing?: boolean;
}

export function AlertConfigModal({
    isOpen,
    onClose,
    currentThresholds,
    onUpdate,
    onCreate,
    onDelete,
    validElementNames,
    isProcessing = false,
}: AlertConfigModalProps) {
    const { toast } = useToast();
    const [editableConfigs, setEditableConfigs] = useState<ElementRange[]>([]);
    const [newElementName, setNewElementName] = useState<string>('');
    const [newMinValue, setNewMinValue] = useState<string>('');
    const [newMaxValue, setNewMaxValue] = useState<string>('');

    useEffect(() => {
        if (isOpen) {
            // Deep copy để tránh mutate prop
            setEditableConfigs(JSON.parse(JSON.stringify(currentThresholds)));
            setNewElementName('');
            setNewMinValue('');
            setNewMaxValue('');
        }
    }, [isOpen, currentThresholds]);

    const handleInputChange = useCallback((index: number, field: 'minValue' | 'maxValue', value: string) => {
        setEditableConfigs((prevConfigs) => {
          console.log(`Input change: index=<span class="math-inline">\{index\}, field\=</span>{field}, value="${value}"`);
            const updatedConfigs = [...prevConfigs];
            const currentConfig = updatedConfigs[index];
            const numericValue = value.trim() === '' ? null : parseFloat(value);

            if (value.trim() !== '' && isNaN(numericValue as number)) {
                // Không thay đổi state nếu giá trị nhập vào không hợp lệ
                 toast({
                     variant: "destructive",
                     title: "Giá trị không hợp lệ",
                     description: `Vui lòng nhập một số hợp lệ.`,
                 });
                return prevConfigs; // Trả về state cũ
            }
            updatedConfigs[index] = { ...currentConfig, [field]: numericValue };
            console.log(`   -> State MỚI dự kiến (bên trong setter) tại index ${index}:`, updatedConfigs[index]);
            console.log(`   -> Toàn bộ state MỚI dự kiến (bên trong setter):`, updatedConfigs);
            return updatedConfigs;
        });
    }, [toast]);

    const handleDeleteClick = useCallback((id: string | undefined, name: string) => {
        if (!id) return;
        onDelete(id, name); // Confirmation nên được xử lý trong onDelete prop nếu cần
    }, [onDelete]);

    const handleAddNewThreshold = useCallback(async () => {
        if (!newElementName) {
            toast({ variant: "warning", title: "Thiếu thông tin", description: "Vui lòng chọn tên chỉ số." });
            return;
        }
        const minValue = newMinValue.trim() === '' ? null : parseFloat(newMinValue);
        const maxValue = newMaxValue.trim() === '' ? null : parseFloat(newMaxValue);

        if ((newMinValue.trim() !== '' && isNaN(minValue as number)) || (newMaxValue.trim() !== '' && isNaN(maxValue as number))) {
             toast({ variant: "destructive", title: "Giá trị không hợp lệ", description: "Giá trị Min/Max không hợp lệ." });
            return;
        }
        if (minValue !== null && maxValue !== null && minValue > maxValue) {
             toast({ variant: "warning", title: "Logic không hợp lệ", description: "Giá trị nhỏ nhất không thể lớn hơn giá trị lớn nhất." });
             return;
         }

        const newConfigData: CreateElementRangeDto = { elementName: newElementName, minValue, maxValue };
        await onCreate(newConfigData);
        // Reset form sau khi gọi API (thành công hay không thì parent sẽ fetch lại và cập nhật)
        setNewElementName('');
        setNewMinValue('');
        setNewMaxValue('');
    }, [newElementName, newMinValue, newMaxValue, onCreate, toast]);

    const handleSaveChanges = useCallback(async () => {
      console.log(">>> Bắt đầu handleSaveChanges. State hiện tại editableConfigs:", editableConfigs); 
         for (const config of editableConfigs) {
             if (config.minValue !== null && config.maxValue !== null && config.minValue > config.maxValue) {
                 toast({ variant: "warning", title: "Logic không hợp lệ", description: `Tại "${config.elementName}", Min không thể lớn hơn Max.` });
                 return;
             }
         }
        const configsToUpdate = editableConfigs.filter(c => c.id); // Chỉ gửi những cái có ID
        console.log(">>> Dữ liệu chuẩn bị gửi đi (configsToUpdate):", configsToUpdate);
        await onUpdate(configsToUpdate);
    }, [editableConfigs, onUpdate, toast]);

    const availableElementNames = useMemo(() => {
        const configuredNames = new Set(editableConfigs.map(c => c.elementName.toLowerCase()));
        return validElementNames.filter(name => !configuredNames.has(name.toLowerCase()));
    }, [editableConfigs, validElementNames]);

    return (
      <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      {/* ---- ÁP DỤNG CẤU TRÚC FLEXBOX CHO DIALOG CONTENT ---- */}
      <DialogContent
          className="sm:max-w-2xl md:max-w-3xl max-h-[90vh] flex flex-col"
          onInteractOutside={(e) => { if (isProcessing) e.preventDefault(); }}
      >
          {/* Header không co lại */}
          <DialogHeader className="pb-4 border-b flex-shrink-0">
              <div className="flex items-center gap-2">
                  <AlertCircle className="h-6 w-6 text-blue-600" />
                  <DialogTitle className="text-xl font-bold text-blue-600">Thiết Lập Ngưỡng Cảnh Báo</DialogTitle>
              </div>
              <DialogDescription className="mt-2 text-gray-600">
                  Đặt giới hạn dưới (Min) và giới hạn trên (Max). Giá trị nằm ngoài khoảng này sẽ được cảnh báo.
              </DialogDescription>
          </DialogHeader>

          {/* ---- VÙNG NỘI DUNG CHÍNH CÓ SCROLL ---- */}
          <div className="flex-grow overflow-y-auto space-y-6 py-4 pr-3 -mr-1">

              {/* --- Phần Ngưỡng Hiện Tại --- */}
              <div className="space-y-6">
                  <h4 className="text-sm font-semibold text-gray-700 top-0 bg-white/95 backdrop-blur-sm z-10 pt-1 -mt-1 pb-1"> 
                      Ngưỡng Hiện tại
                  </h4>
                  {editableConfigs.length === 0 ? (
                      <p className="text-sm text-gray-500 text-center py-4">Chưa có cấu hình ngưỡng nào.</p>
                  ) : (
                      editableConfigs.map((config, index) => {
                          const info = getMetricInfo(config.elementName);
                          const isLastItem = index === editableConfigs.length - 1;
                          return (
                              <div key={config.id || `new-${index}`} className="space-y-3">
                                  {/* Header của metric */}
                                  <div className="flex items-center justify-between">
                                      <div className="flex items-center gap-2">
                                          <div className={cn("p-1.5 rounded-lg", info.color)}>{info.icon}</div>
                                          <Label className="text-base font-semibold text-gray-800">
                                              {config.elementName}
                                          </Label>
                                      </div>
                                       <TooltipProvider delayDuration={100}>
                                           <Tooltip>
                                               <TooltipTrigger asChild>
                                                   <Button
                                                       variant="ghost" size="icon" className="h-8 w-8 text-red-500 hover:bg-red-100 hover:text-red-600"
                                                       onClick={() => handleDeleteClick(config.id, config.elementName)} disabled={isProcessing || !config.id} aria-label={`Xóa ngưỡng ${config.elementName}`}
                                                   > <Trash2 className="h-4 w-4" /> </Button>
                                               </TooltipTrigger>
                                               <TooltipContent><p>Xóa ngưỡng {config.elementName}</p></TooltipContent>
                                           </Tooltip>
                                       </TooltipProvider>
                                  </div>
                                  {/* Input Min/Max */}
                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2 pl-10">
                                      <div className="space-y-1">
                                          <Label htmlFor={`min-${index}`} className="text-xs font-medium text-gray-600">Giá trị Min</Label>
                                          <Input id={`min-${index}`} type="number" step="any" placeholder="Trống = không giới hạn" value={config.minValue ?? ""} onChange={(e) => handleInputChange(index, 'minValue', e.target.value)} className="h-9 text-sm" disabled={isProcessing} />
                                      </div>
                                      <div className="space-y-1">
                                           <Label htmlFor={`max-${index}`} className="text-xs font-medium text-gray-600">Giá trị Max</Label>
                                          <Input id={`max-${index}`} type="number" step="any" placeholder="Trống = không giới hạn" value={config.maxValue ?? ""} onChange={(e) => handleInputChange(index, 'maxValue', e.target.value)} className="h-9 text-sm" disabled={isProcessing} />
                                      </div>
                                  </div>
                                  {/* Đường kẻ ngăn cách */}
                                  {!isLastItem && <Separator className="mt-4"/>}
                              </div>
                          );
                      })
                  )}
              </div>
              {/* --- Kết thúc Phần Ngưỡng Hiện Tại --- */}

              {/* --- Phần Thêm Ngưỡng Mới --- */}
              <Separator className="my-4" />
              <div>
                  <h4 className="mb-3 text-sm font-semibold text-gray-700">Thêm Ngưỡng Mới</h4>
                  <div className="grid grid-cols-[1fr_auto_auto_auto] gap-3 items-end">
                     <div>
                         <Label htmlFor="new-element-name" className="text-xs mb-1 block">Chỉ số</Label>
                         <Select value={newElementName} onValueChange={setNewElementName} disabled={isProcessing || availableElementNames.length === 0}>
                             <SelectTrigger id="new-element-name" className="h-9 text-sm">
                                 <SelectValue placeholder={availableElementNames.length === 0 ? "Đã đủ" : "Chọn..."} />
                             </SelectTrigger>
                             <SelectContent>
                                 {availableElementNames.map(name => ( <SelectItem key={name} value={name} className="text-sm">{name}</SelectItem> ))}
                             </SelectContent>
                         </Select>
                     </div>
                     <div>
                         <Label htmlFor="new-min-value" className="text-xs mb-1 block">Giá trị Min</Label>
                         <Input id="new-min-value" type="number" step="any" placeholder="Min" value={newMinValue} onChange={(e) => setNewMinValue(e.target.value)} className="h-9 text-sm" disabled={isProcessing} />
                     </div>
                     <div>
                         <Label htmlFor="new-max-value" className="text-xs mb-1 block">Giá trị Max</Label>
                         <Input id="new-max-value" type="number" step="any" placeholder="Max" value={newMaxValue} onChange={(e) => setNewMaxValue(e.target.value)} className="h-9 text-sm" disabled={isProcessing} />
                     </div>
                     <Button size="sm" className="h-9" onClick={handleAddNewThreshold} disabled={isProcessing || !newElementName} aria-label="Thêm ngưỡng mới">
                         <PlusCircle className="h-4 w-4 mr-1" /> Thêm
                     </Button>
                  </div>
              </div>
               {/* --- Kết thúc Phần Thêm Ngưỡng Mới --- */}

          </div>
           {/* ---- KẾT THÚC VÙNG NỘI DUNG CHÍNH CÓ SCROLL ---- */}

          {/* Footer không co lại, đẩy xuống dưới */}
          <DialogFooter className="mt-auto pt-4 border-t sm:justify-end gap-2 flex-shrink-0">
              <DialogClose asChild>
                  <Button type="button" variant="outline" disabled={isProcessing}>
                      Hủy
                  </Button>
              </DialogClose>
              <Button type="button" onClick={handleSaveChanges} disabled={isProcessing}>
                  {isProcessing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Lưu Thay Đổi
              </Button>
          </DialogFooter>
      </DialogContent>
  </Dialog>
    );
}