// src/app/dashboard/modelManagement/page.tsx
"use client";

import { ModelAI } from "@/types/models";
import { Station, QueryOptions } from "@/types/station2";
import React, { useEffect, useState, useCallback, useMemo } from "react"; // Import useMemo
import dynamic from "next/dynamic";

// --- Các import khác giữ nguyên ---
// ... (imports for Shadcn/UI, Custom Components, Icons, API Functions) ...
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger, DialogClose } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import { Separator } from "@/components/ui/separator";

import PageLoader from "@/components/pageloader";
import { Pagination } from "@/components/pagination";

import { RefreshCw, Cpu, List, Info, AlertTriangle, Trash2, Edit, BrainCircuit, Filter, SortAsc, SortDesc, XCircle, SlidersHorizontal, CalendarDays, Tag, MapPin, Globe, Building } from "lucide-react";

import {
    trainStationsModels,
    getAllAIModels,
    predictStationsModels,
    updateAIModel,
    deleteAIModel
} from "@/lib/model";
import { getStations } from "@/lib/station";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";


// --- Định nghĩa kiểu cho form (giữ nguyên) ---
// ... (Interfaces: UpdateFormData, TrainFormData, PredictFormData) ...
interface UpdateFormData {
    name: string;
    description: string;
    availability: boolean;
}
interface TrainFormData {
    train_test_ratio: number;
    place_ids: string[];
    date_tag: string;
}
interface PredictFormData {
    num_step: number;
    freq_days: number;
    model_types: string[];
    place_ids: string[];
}


export default function ModelManagement() {
    const { toast } = useToast();
    const MODEL_TYPES_OPTIONS = ['xgb', 'rf', 'ETSformerPar', 'ETSformer'];
    const MODEL_DISPLAY_NAMES: { [key: string]: string } = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'ETSformer':'ETSformer',
        'ETSformerPar':'ETSformer Parallel',
      };
    // --- States (giữ nguyên) ---
    // ... (all states: allFetchedModels, displayedModels, stations, etc.) ...
    const [allFetchedModels, setAllFetchedModels] = useState<ModelAI[]>([]);
    const [displayedModels, setDisplayedModels] = useState<ModelAI[]>([]);
    const [stations, setStations] = useState<Station[]>([]);
    const [selectedModel, setSelectedModel] = useState<ModelAI | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isProcessing, setIsProcessing] = useState<{ [key: string]: boolean }>({});
    const [activeTab, setActiveTab] = useState("overview"); // Tab cho model details
    const [isClient, setIsClient] = useState(false);
    const [error, setError] = useState<{ title: string; message: string } | null>(null);
    const [currentPage, setCurrentPage] = useState(1);
    const [itemsPerPage, setItemsPerPage] = useState(10);
    const [totalItems, setTotalItems] = useState(0);
    const totalPages = useMemo(() => Math.ceil(totalItems / itemsPerPage), [totalItems, itemsPerPage]);
    const [filterStationId, setFilterStationId] = useState<string>('all');
    const [filterStatus, setFilterStatus] = useState<string>('all');
    const [sortBy, setSortBy] = useState<string>('created_at');
    const [sortDesc, setSortDesc] = useState<boolean>(true);
    const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
    const [isTrainModalOpen, setIsTrainModalOpen] = useState(false); // Modal Train toàn cục
    const [isPredictModalOpen, setIsPredictModalOpen] = useState(false); // Modal Predict toàn cục
    const [currentModelForAction, setCurrentModelForAction] = useState<ModelAI | null>(null); // Chỉ dùng cho Update/Delete
    const [updateFormData, setUpdateFormData] = useState<UpdateFormData>({ name: '', description: '', availability: false });
    const [trainFormData, setTrainFormData] = useState<TrainFormData>({ train_test_ratio: 0.7, place_ids: [], date_tag: '' });
    const [predictFormData, setPredictFormData] = useState<PredictFormData>({ num_step: 7, freq_days: 7, model_types: MODEL_TYPES_OPTIONS.length > 0 ? [MODEL_TYPES_OPTIONS[0]] : [], place_ids: [] });


    // --- Constants (giữ nguyên) ---
    // ... (AVAILABLE_STATUSES, SORTABLE_FIELDS, MODEL_TYPES_OPTIONS) ...
    const AVAILABLE_STATUSES = [
        { value: 'available', label: 'sẵn sàng' },
        { value: 'unavailable', label: 'Không sẵn sàng' },
    ];
    const SORTABLE_FIELDS = [
        { value: 'name', label: 'Tên Model' },
        { value: 'created_at', label: 'Ngày tạo' },
        { value: 'updated_at', label: 'Ngày cập nhật' },
        { value: 'trained_at', label: 'Ngày huấn luyện' },
        { value: 'version', label: 'Phiên bản' },
    ];



    useEffect(() => {
        setIsClient(true);
    }, []);

    // --- Fetching Functions (giữ nguyên) ---
    // ... (fetchAllModelsData, fetchStations) ...
    const fetchAllModelsData = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const fetchedModels = await getAllAIModels();
            setAllFetchedModels(fetchedModels);
        } catch (err: any) {
            console.error("Failed to fetch AI models:", err);
            setError({ title: "Lỗi Tải Dữ Liệu", message: err.message || "Không thể tải danh sách models." });
            toast({ variant: "destructive", title: "Lỗi Tải Dữ Liệu", description: err.message || "Không thể tải danh sách models." });
            setAllFetchedModels([]);
        } finally {
            setIsLoading(false);
        }
    }, [toast]);

    const fetchStations = useCallback(async () => {
        try {
            const options: QueryOptions = { limit: 500 };
            const fetchedStations = await getStations(options);
            setStations(fetchedStations);
        } catch (err: any) {
            console.error("Failed to fetch Stations:", err);
            setError(prev => ({
                title: prev?.title ?? "Lỗi Tải Stations",
                message: `${prev?.message ?? ''}\n${err.message || "Không thể tải danh sách stations."}`.trim()
            }));
            toast({ variant: "destructive", title: "Lỗi Tải Stations", description: err.message || "Không thể tải danh sách stations." });
        }
    }, [toast]);


    // --- Effect tải dữ liệu ban đầu (giữ nguyên) ---
    // ... (useEffect for initial fetch) ...
    useEffect(() => {
        if (isClient) {
            fetchAllModelsData();
            fetchStations();
        }
    }, [isClient, fetchAllModelsData, fetchStations]);


    // --- Sửa: Tính toán danh sách ID các trạm có model ---
    const stationIdsWithModels = useMemo(() => {
        const ids = new Set<string>();
        // Chỉ xem xét các model chưa bị xóa và có station_id
        allFetchedModels.forEach(model => {
            if (model.station_id && !model.deleted_at) {
                ids.add(model.station_id);
            }
        });
        return ids;
    }, [allFetchedModels]);

    // --- Sửa: Lọc danh sách trạm để hiển thị trong Predict form ---
    const stationsAvailableForPrediction = useMemo(() => {
        // Lọc danh sách stations dựa trên những ID có model
        return stations.filter(station => stationIdsWithModels.has(station.id));
    }, [stations, stationIdsWithModels]);


    // --- Effect xử lý lọc, sắp xếp, phân trang client-side (giữ nguyên) ---
    // ... (useEffect for client-side processing) ...
    useEffect(() => {
        if (!isClient || isLoading) return;

        let processedModels = [...allFetchedModels];

        // 1. Filtering
        if (filterStationId !== 'all') {
            processedModels = processedModels.filter(m => m.station_id === filterStationId);
        }
        if (filterStatus === 'deleted') {
            processedModels = processedModels.filter(m => m.deleted_at !== null);
        } else if (filterStatus !== 'all') {
            processedModels = processedModels.filter(m => m.availability === (filterStatus === 'available') && m.deleted_at === null);
        } else {
            processedModels = processedModels.filter(m => m.deleted_at === null);
        }

        // 2. Sorting
        processedModels.sort((a, b) => {
            const valA = a[sortBy as keyof ModelAI];
            const valB = b[sortBy as keyof ModelAI];

            const isANull = valA === null || valA === undefined;
            const isBNull = valB === null || valB === undefined;
            if (isANull && isBNull) return 0;
            if (isANull) return 1;
            if (isBNull) return -1;

            let comparison = 0;
            if (sortBy === 'created_at' || sortBy === 'updated_at' || sortBy === 'trained_at') {
                const dateA = new Date(valA as string);
                const dateB = new Date(valB as string);
                if (isNaN(dateA.getTime()) && isNaN(dateB.getTime())) comparison = 0;
                else if (isNaN(dateA.getTime())) comparison = 1;
                else if (isNaN(dateB.getTime())) comparison = -1;
                else comparison = dateA.getTime() - dateB.getTime();
            } else if (typeof valA === 'string' && typeof valB === 'string') {
                comparison = valA.localeCompare(valB);
            } else if (typeof valA === 'number' && typeof valB === 'number') {
                comparison = valA - valB;
            } else if (typeof valA === 'boolean' && typeof valB === 'boolean') {
                comparison = (valA === valB) ? 0 : valA ? 1 : -1;
            } else {
                if (valA > valB) comparison = 1;
                else if (valA < valB) comparison = -1;
            }
            return sortDesc ? comparison * -1 : comparison;
        });

        // 3. Pagination
        setTotalItems(processedModels.length);
        const calculatedOffset = (currentPage - 1) * itemsPerPage;
        const calculatedLimit = itemsPerPage;
        const paginatedModels = processedModels.slice(calculatedOffset, calculatedOffset + calculatedLimit);

        setDisplayedModels(paginatedModels);

        if (selectedModel && !paginatedModels.find(m => m.id === selectedModel.id)) {
            setSelectedModel(null);
        }
        if (!selectedModel && paginatedModels.length > 0) {
            setSelectedModel(paginatedModels[0]);
        } else if (paginatedModels.length === 0) {
            setSelectedModel(null);
        }

    }, [allFetchedModels, currentPage, itemsPerPage, filterStationId, filterStatus, sortBy, sortDesc, isClient, isLoading, selectedModel]);


    // --- Handlers cho Pagination (giữ nguyên) ---
    // ... (handlePageChange) ...
    const handlePageChange = (page: number) => {
        setCurrentPage(page);
    };


    // --- Handlers cho Filter/Sort (giữ nguyên) ---
    // ... (handleFilterChange, handleSortChange, clearFilters) ...
    const handleFilterChange = (type: 'station' | 'status', value: string) => {
        if (type === 'station') setFilterStationId(value);
        if (type === 'status') setFilterStatus(value);
        setCurrentPage(1);
    };
    const handleSortChange = (type: 'field' | 'direction', value: string | boolean) => {
        if (type === 'field') setSortBy(value as string);
        if (type === 'direction') setSortDesc(value as boolean);
        setCurrentPage(1);
    };
    const clearFilters = () => {
        setFilterStationId('all');
        setFilterStatus('all');
        setSortBy('created_at');
        setSortDesc(true);
        setCurrentPage(1);
    };


    // --- Handlers Mở Modal (giữ nguyên) ---
    // ... (openUpdateModal, openGlobalTrainModal, openGlobalPredictModal) ...
    // Mở modal Update (gắn với model cụ thể)
    const openUpdateModal = (model: ModelAI) => {
        setCurrentModelForAction(model); // Cần model này để biết ID cần update
        setError(null);
        setUpdateFormData({
            name: model.name,
            description: model.description ?? '',
            availability: model.availability
        });
        setIsUpdateModalOpen(true);
    };

    // Mở modal Train (hành động toàn cục)
    const openGlobalTrainModal = () => {
        setCurrentModelForAction(null); // Không có model context cụ thể
        setError(null);
        const today = new Date();
        const dateTag = `${today.getDate().toString().padStart(2, '0')}${(today.getMonth() + 1).toString().padStart(2, '0')}${today.getFullYear().toString().slice(-2)}`;
        setTrainFormData({ // Reset form, không prefill place_id
            train_test_ratio: 0.7,
            place_ids: [],
            date_tag: dateTag
        });
        setIsTrainModalOpen(true);
    };

    // Mở modal Predict (hành động toàn cục)
    const openGlobalPredictModal = () => {
        setCurrentModelForAction(null); // Không có model context cụ thể
        setError(null);
        setPredictFormData({ // Reset form, không prefill place_id
            num_step: 7,
            freq_days: 7,
            model_types: ['xgb', 'rf'],
            place_ids: []
        });
        setIsPredictModalOpen(true);
    };


    // --- Handlers cho Form Input (giữ nguyên) ---
    // ... (handleFormInputChange) ...
    const handleFormInputChange = (
        form: 'update' | 'train' | 'predict',
        field: keyof UpdateFormData | keyof TrainFormData | keyof PredictFormData,
        value: any
    ) => {
        if (form === 'update') {
            setUpdateFormData(prev => ({ ...prev, [field]: value as never }));
        } else if (form === 'train') {
            if (field === 'place_ids') {
                setTrainFormData(prev => ({
                    ...prev,
                    place_ids: prev.place_ids.includes(value)
                        ? prev.place_ids.filter(id => id !== value)
                        : [...prev.place_ids, value]
                }));
            } else {
                setTrainFormData(prev => ({ ...prev, [field]: value as never }));
            }
        } else if (form === 'predict') {
            if (field === 'place_ids') { // Xử lý place_ids (array) như cũ
                setPredictFormData(prev => {
                    const currentValues = prev.place_ids;
                    return {
                        ...prev,
                        place_ids: currentValues.includes(value)
                            ? currentValues.filter(id => id !== value)
                            : [...currentValues, value]
                    };
                });
            } else if (field === 'model_types') { // Xử lý khi RadioGroup thay đổi
                // value ở đây là chuỗi đơn ('xgb' hoặc 'rf') được chọn
                setPredictFormData(prev => ({
                    ...prev,
                    // Luôn đặt model_types thành một mảng chỉ chứa giá trị mới nhất được chọn
                    model_types: [value]
                }));
            } else { // Xử lý các trường khác (num_step, freq_days)
                setPredictFormData(prev => ({
                    ...prev,
                    [field]: value as never
                }));
            }
        }
    };


    // --- Handlers cho Submit Forms (giữ nguyên) ---
    // ... (handleTrainSubmit, handlePredictSubmit, handleUpdateSubmit) ...
    const handleTrainSubmit = async () => {
        const actionKey = 'global-train';
        if (!trainFormData.date_tag) {
            toast({ variant: "destructive", title: "Lỗi", description: "Vui lòng nhập Date Tag." });
            return;
        }
        if (trainFormData.train_test_ratio <= 0 || trainFormData.train_test_ratio >= 1) {
            toast({ variant: "destructive", title: "Lỗi", description: "Train/Test Ratio phải nằm trong khoảng (0, 1)." });
            return;
        }
        setIsProcessing(prev => ({ ...prev, [actionKey]: true }));
        setError(null);
        toast({ variant: "default", title: "Đang xử lý", description: "Đang gửi yêu cầu huấn luyện..." });
        try {
            const params: Parameters<typeof trainStationsModels>[0] = {
                train_test_ratio: Number(trainFormData.train_test_ratio),
                date_tag: trainFormData.date_tag,
                place_ids: trainFormData.place_ids.length > 0 ? trainFormData.place_ids : undefined,
            };
            console.log(params)
            const trainedModels = await trainStationsModels(params);
            console.log(trainedModels)
            toast({ variant: "success", title: "Thành công", description: `Yêu cầu huấn luyện thành công! ${trainedModels.length} model được xử lý.` });
            setIsTrainModalOpen(false);
            fetchAllModelsData();
        } catch (err: any) {
            console.error(`Lỗi khi huấn luyện:`, err);
            setError({ title: "Lỗi Huấn Luyện", message: err.message || "Không thể bắt đầu huấn luyện." });
            toast({ variant: "destructive", title: "Lỗi Huấn Luyện", description: err.message || "Không thể bắt đầu huấn luyện." });
        } finally {
            setIsProcessing(prev => ({ ...prev, [actionKey]: false }));
        }
    };

    const handlePredictSubmit = async () => {
        const actionKey = 'global-predict';
        if (predictFormData.num_step < 1 || predictFormData.num_step > 14) {
            toast({ variant: "destructive", title: "Lỗi", description: "Num Step phải từ 1 đến 14." });
            return;
        }
        if (predictFormData.freq_days < 1 || predictFormData.freq_days > 14) {
            toast({ variant: "destructive", title: "Lỗi", description: "Freq Days phải từ 1 đến 14." });
            return;
        }
        if (predictFormData.model_types.length === 0) {
            toast({ variant: "destructive", title: "Lỗi", description: "Vui lòng chọn một Model Type." });
            return;
        }
        // Sửa: Thêm kiểm tra nếu bắt buộc chọn trạm
        if (predictFormData.place_ids.length === 0) {
            toast({ variant: "destructive", title: "Lỗi", description: "Vui lòng chọn ít nhất một trạm để dự đoán." });
            return; // Ngăn submit nếu không chọn trạm nào
        }

        setIsProcessing(prev => ({ ...prev, [actionKey]: true }));
        setError(null);
        toast({ variant: "default", title: "Đang xử lý", description: "Đang gửi yêu cầu dự đoán..." });

        try {
            const params: Parameters<typeof predictStationsModels>[0] = {
                num_step: Number(predictFormData.num_step),
                freq_days: Number(predictFormData.freq_days),
                model_types: predictFormData.model_types,
                // Luôn gửi place_ids vì đã kiểm tra ở trên
                place_ids: predictFormData.place_ids,
            };
            const status = await predictStationsModels(params);
            if (status >= 200 && status < 300) {
                toast({ variant: "success", title: "Thành công", description: "Yêu cầu dự đoán đã được gửi thành công." });
                setIsPredictModalOpen(false);
            } else {
                throw new Error(`Server phản hồi với status ${status}`);
            }
        } catch (err: any) {
            console.error(`Lỗi khi dự đoán:`, err);
            setError({ title: "Lỗi Dự Đoán", message: err.message || "Không thể bắt đầu dự đoán." });
            toast({ variant: "destructive", title: "Lỗi Dự Đoán", description: err.message || "Không thể bắt đầu dự đoán." });
        } finally {
            setIsProcessing(prev => ({ ...prev, [actionKey]: false }));
        }
    };

    const handleUpdateSubmit = async () => {
        if (!currentModelForAction) return;
        const modelId = currentModelForAction.id;
        const actionKey = `update-${modelId}`;
        // Kiểm tra validation (giữ nguyên)
        if (!updateFormData.name) {
            toast({ variant: "destructive", title: "Lỗi", description: "Tên Model không được để trống." });
            return;
        }

        setIsProcessing(prev => ({ ...prev, [actionKey]: true }));
        setError(null);
        toast({ variant: "default", title: "Đang xử lý", description: `Đang cập nhật model ${modelId}...` });

        try {
            // Dữ liệu gửi đi để cập nhật
            const updates: Partial<ModelAI> = {
                name: updateFormData.name,
                description: updateFormData.description,
                availability: updateFormData.availability,
                // Lưu ý: updated_at sẽ do backend xử lý và được cập nhật chính xác khi fetchAllModelsData chạy
            };

            const status = await updateAIModel(updates, modelId);

            if (status >= 200 && status < 300) {
                toast({ variant: "success", title: "Thành công", description: `Model ${modelId} đã được cập nhật.` });
                setIsUpdateModalOpen(false); // Đóng modal

                // --- BẮT ĐẦU THAY ĐỔI ---
                // 1. Cập nhật trực tiếp state selectedModel nếu model đang hiển thị chính là model vừa cập nhật
                if (selectedModel && selectedModel.id === modelId) {
                    setSelectedModel(prevSelectedModel => {
                        if (!prevSelectedModel) return null; // Trường hợp hiếm gặp nhưng để an toàn
                        // Tạo object mới bằng cách kết hợp state cũ và dữ liệu vừa cập nhật thành công
                        return {
                            ...prevSelectedModel, // Giữ lại các thuộc tính cũ
                            ...updates           // Ghi đè các thuộc tính đã được cập nhật
                            // updated_at sẽ được cập nhật chính xác ở bước fetch dưới đây
                        };
                    });
                }
                // --- KẾT THÚC THAY ĐỔI ---

                // 2. Fetch lại toàn bộ dữ liệu để đảm bảo đồng bộ hoàn toàn với server
                // (Bao gồm cả updated_at mới nhất và đảm bảo danh sách được sắp xếp/lọc đúng)
                fetchAllModelsData();

            } else {
                // Xử lý lỗi server response (giữ nguyên)
                throw new Error(`Server phản hồi với status ${status}`);
            }
        } catch (err: any) {
            // Xử lý lỗi gọi API (giữ nguyên)
            console.error(`Lỗi khi cập nhật model ${modelId}:`, err);
            setError({ title: "Lỗi Cập Nhật", message: err.message || `Không thể cập nhật model ${modelId}.` });
            toast({ variant: "destructive", title: "Lỗi Cập Nhật", description: err.message || `Không thể cập nhật model ${modelId}.` });
        } finally {
            // Kết thúc trạng thái xử lý (giữ nguyên)
            setIsProcessing(prev => ({ ...prev, [actionKey]: false }));
        }
    };


    // --- Handler cho Delete (giữ nguyên) ---
    // ... (handleDelete) ...
    const handleDelete = async (modelIdToDelete: string) => {
        if (!modelIdToDelete) return;
        const actionKey = `delete-${modelIdToDelete}`;
        setIsProcessing(prev => ({ ...prev, [actionKey]: true }));
        setError(null);
        toast({ title: "Đang xử lý", description: `Đang xóa model ${modelIdToDelete}...` });
        try {
            const status = await deleteAIModel(modelIdToDelete);
            if (status >= 200 && status < 300) {
                toast({ title: "Thành công", description: `Model ${modelIdToDelete} đã được xóa.` });
                if (selectedModel?.id === modelIdToDelete) {
                    setSelectedModel(null);
                }
                fetchAllModelsData();
            } else {
                throw new Error(`Server phản hồi với status ${status}`);
            }
        } catch (err: any) {
            console.error(`Lỗi khi xóa model ${modelIdToDelete}:`, err);
            setError({ title: "Lỗi Xóa Model", message: err.message || `Không thể xóa model ${modelIdToDelete}.` });
            toast({ variant: "destructive", title: "Lỗi Xóa Model", description: err.message || `Không thể xóa model ${modelIdToDelete}.` });
        } finally {
            setIsProcessing(prev => ({ ...prev, [actionKey]: false }));
        }
    };


    // --- Helper Functions (giữ nguyên) ---
    // ... (getStationName, getStatusBadge) ...
    const getStationName = (stationId: string): string => {
        const station = stations.find(s => s.id === stationId);
        return station ? station.name : 'N/A';
    };

    const getStatusBadge = (model: ModelAI) => {
        const isDeleting = isProcessing[`delete-${model.id}`];
        if (isDeleting) {
            return <span className="inline-flex items-center rounded-full bg-yellow-100 px-2.5 py-0.5 text-xs font-medium text-yellow-800 animate-pulse">Đang xóa...</span>;
        }
        if (model.deleted_at) {
            return <span className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-800">Đã xóa</span>;
        }
        if (model.availability) {
            return <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-800">Sẵn sàng</span>;
        }
        return <span className="inline-flex items-center rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-medium text-red-800">Không sẵn sàng</span>;

    };


    // --- Render ---
    if (!isClient) {
        return <PageLoader message="Đang khởi tạo giao diện..." />;
    }

    // Tìm thông tin chi tiết của trạm đang được chọn (nếu có)
    const selectedStationInfo = selectedModel ? stations.find(s => s.id === selectedModel.station_id) : null;


    return (
        <TooltipProvider>
            <div className="flex min-h-screen flex-col">
                <main className="flex-1 p-4 md:p-6">
                    {/* --- Header (giữ nguyên) --- */}
                    {/* ... (Header: Title, Global Buttons) ... */}
                    <div className="mb-4 flex flex-col items-start justify-between gap-4 md:flex-row md:items-center">
                        <h1 className="text-2xl font-bold md:text-3xl">Quản lý Model AI</h1>
                        <div className="flex gap-2">
                            <Button onClick={openGlobalTrainModal} disabled={isProcessing['global-train']}>
                                <RefreshCw className={`mr-2 h-4 w-4 ${isProcessing['global-train'] ? 'animate-spin' : ''}`} /> Huấn luyện Models
                            </Button>
                            <Button onClick={openGlobalPredictModal} variant="outline" disabled={isProcessing['global-predict']}>
                                <BrainCircuit className={`mr-2 h-4 w-4 ${isProcessing['global-predict'] ? 'animate-spin' : ''}`} /> Dự đoán Models
                            </Button>
                        </div>
                    </div>


                    {/* --- Filtering & Sorting Controls (giữ nguyên) --- */}
                    {/* ... (Filter/Sort Card) ... */}
                    <Card className="mb-4">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-lg">
                                <SlidersHorizontal className="h-5 w-5" /> Bộ lọc & Sắp xếp
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="flex flex-col gap-4 md:flex-row md:items-end">
                            <div className="flex-1">
                                <Label htmlFor="filter-station">Trạm Quan Trắc</Label>
                                <Select value={filterStationId} onValueChange={(value) => handleFilterChange('station', value)}>
                                    <SelectTrigger id="filter-station">
                                        <SelectValue placeholder="Tất cả các trạm" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="all">Tất cả các trạm</SelectItem>
                                        {stations.map(station => (
                                            <SelectItem key={station.id} value={station.id}>{station.name}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                            <div className="flex-1">
                                <Label htmlFor="filter-status">Trạng thái</Label>
                                <Select value={filterStatus} onValueChange={(value) => handleFilterChange('status', value)}>
                                    <SelectTrigger id="filter-status">
                                        <SelectValue placeholder="Tất cả trạng thái" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="all">Tất cả</SelectItem>
                                        {AVAILABLE_STATUSES.map(status => (
                                            <SelectItem key={status.value} value={status.value}>{status.label}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                            <div className="flex-1">
                                <Label htmlFor="sort-by">Sắp xếp theo</Label>
                                <Select value={sortBy} onValueChange={(value) => handleSortChange('field', value)}>
                                    <SelectTrigger id="sort-by">
                                        <SelectValue placeholder="Chọn trường sắp xếp" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {SORTABLE_FIELDS.map(field => (
                                            <SelectItem key={field.value} value={field.value}>{field.label}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                            <div className="flex items-end gap-2">
                                <Button
                                    variant="outline"
                                    size="icon"
                                    onClick={() => handleSortChange('direction', !sortDesc)}
                                    className="h-10 w-10"
                                >
                                    {sortDesc ? <SortDesc className="h-5 w-5" /> : <SortAsc className="h-5 w-5" />}
                                    <span className="sr-only">{sortDesc ? "Sắp xếp giảm dần" : "Sắp xếp tăng dần"}</span>
                                </Button>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Button variant="ghost" size="icon" onClick={clearFilters} className="h-10 w-10">
                                            <XCircle className="h-5 w-5 text-muted-foreground" />
                                            <span className="sr-only">Xóa bộ lọc</span>
                                        </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Xóa bộ lọc & sắp xếp</TooltipContent>
                                </Tooltip>
                            </div>
                        </CardContent>
                    </Card>


                    {/* --- Error Display (giữ nguyên) --- */}
                    {/* ... (Error Alert) ... */}
                    {error && (
                        <Alert variant="destructive" className="mb-4">
                            <AlertTriangle className="h-4 w-4" />
                            <AlertTitle>{error.title}</AlertTitle>
                            <AlertDescription>{error.message}</AlertDescription>
                        </Alert>
                    )}


                    {/* --- Main Content Area: List & Details (giữ nguyên layout 2 cột) --- */}
                    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                        {/* --- Model List Column (giữ nguyên) --- */}
                        {/* ... (Model List rendering) ... */}
                        <div className="lg:col-span-1 flex flex-col"> {/* Thêm flex flex-col để quản lý chiều cao */}
                            <h2 className="mb-3 text-lg font-semibold flex-shrink-0">Danh sách Models ({totalItems})</h2> {/* Header không cuộn */}
                            {isLoading ? (
                                <div className="flex-grow flex items-center justify-center"> {/* Cho loader vào giữa nếu cần */}
                                    <PageLoader message="Đang tải models..." />
                                </div>
                            ) : displayedModels.length > 0 ? (
                                // Container cho phần cuộn và phân trang
                                <div className="flex flex-col flex-grow min-h-0">
                                    {/* ---- BẮT ĐẦU KHU VỰC CUỘN ---- */}
                                    <ScrollArea className="pr-4 flex-grow max-h-[600px] lg:max-h-[calc(100vh-300px)]"> {/* Đặt max-height và padding phải */}
                                        {/* div này chỉ chứa danh sách các card */}
                                        <div className="space-y-3">
                                            {displayedModels.map(model => (
                                                <Card
                                                    key={model.id}
                                                    className={`cursor-pointer transition-all hover:shadow-md ${selectedModel?.id === model.id ? 'border-primary ring-1 ring-primary' : 'border-border'}`}
                                                    onClick={() => { setSelectedModel(model); setActiveTab('overview'); }}
                                                >
                                                    <CardHeader className="flex flex-row items-center justify-between space-y-0 p-3">
                                                        <CardTitle className="text-sm font-medium flex items-center gap-2 truncate pr-2">
                                                            <Cpu className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                                                            <span className="truncate" title={`${model.name} (${model.version})`}>{MODEL_DISPLAY_NAMES[model.name]} ({model.version})</span>
                                                        </CardTitle>
                                                        <div className="flex-shrink-0">{getStatusBadge(model)}</div>
                                                    </CardHeader>
                                                </Card>
                                            ))}
                                        </div>
                                    </ScrollArea>
                                    {/* ---- KẾT THÚC KHU VỰC CUỘN ---- */}

                                    {/* ---- Phân trang (Nằm ngoài khu vực cuộn) ---- */}
                                    {totalPages > 1 && (
                                        <div className="mt-4 flex justify-center flex-shrink-0"> {/* flex-shrink-0 để không bị co lại */}
                                            <Pagination
                                                currentPage={currentPage}
                                                totalPages={totalPages}
                                                onPageChange={handlePageChange}
                                                siblingCount={0} // Giữ cho phân trang gọn
                                            />
                                        </div>
                                    )}
                                </div>
                            ) : (
                                // Hiển thị khi không có model
                                <div className="mt-6 text-center text-muted-foreground border rounded-lg bg-card py-10 flex-grow flex items-center justify-center">
                                    {allFetchedModels.length > 0 ? "Không tìm thấy model nào phù hợp." : "Không có model nào."}
                                </div>
                            )}
                        </div>


                        {/* --- Model Details Column (giữ nguyên) --- */}
                        {/* ... (Model Details rendering with Tabs) ... */}
                        <div className="lg:col-span-2">
                            <h2 className="mb-3 text-lg font-semibold">Chi tiết Model</h2>
                            {selectedModel ? (
                                <Card>
                                    <CardHeader>
                                        <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:justify-between">
                                            <div>
                                                <CardTitle className="text-xl">{MODEL_DISPLAY_NAMES[selectedModel.name]} ({selectedStationInfo?.name})</CardTitle>
                                                <CardDescription>Phiên bản: {selectedModel.version} {getStatusBadge(selectedModel)}</CardDescription>
                                            </div>
                                            <div className="flex gap-2">
                                                <Tooltip>
                                                    <TooltipTrigger asChild>
                                                        <Button variant="outline" size="sm" onClick={() => openUpdateModal(selectedModel)} disabled={isProcessing[`update-${selectedModel.id}`]}>
                                                            <Edit className="h-4 w-4" /> <span className="ml-1 hidden sm:inline">Sửa</span>
                                                        </Button>
                                                    </TooltipTrigger>
                                                    <TooltipContent>Chỉnh sửa thông tin</TooltipContent>
                                                </Tooltip>
                                                <Dialog>
                                                    <DialogTrigger asChild>
                                                        <Button variant="destructive" size="sm" disabled={isProcessing[`delete-${selectedModel.id}`]}>
                                                            <Trash2 className="h-4 w-4" /> <span className="ml-1 hidden sm:inline">Xóa</span>
                                                        </Button>
                                                    </DialogTrigger>
                                                    <DialogContent>
                                                        <DialogHeader>
                                                            <DialogTitle>Xác nhận xóa Model?</DialogTitle>
                                                            <DialogDescription>
                                                                Hành động này sẽ xóa model <strong>{selectedModel.name} ({selectedModel.version})</strong>. Bạn có chắc chắn?
                                                            </DialogDescription>
                                                        </DialogHeader>
                                                        <DialogFooter>
                                                            <DialogClose asChild>
                                                                <Button variant="outline" disabled={isProcessing[`delete-${selectedModel.id}`]}>Hủy</Button>
                                                            </DialogClose>
                                                            <Button
                                                                variant="destructive"
                                                                onClick={() => handleDelete(selectedModel.id)}
                                                                disabled={isProcessing[`delete-${selectedModel.id}`]}
                                                            >
                                                                {isProcessing[`delete-${selectedModel.id}`] ? "Đang xóa..." : "Xác nhận xóa"}
                                                            </Button>
                                                        </DialogFooter>
                                                    </DialogContent>
                                                </Dialog>
                                            </div>
                                        </div>
                                    </CardHeader>
                                    <CardContent>
                                        <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
                                            <TabsList className="mb-4 grid w-full grid-cols-2">
                                                <TabsTrigger value="overview"> <Info className="mr-2 h-4 w-4" /> Tổng quan</TabsTrigger>
                                                <TabsTrigger value="station_info" disabled={!selectedModel.station_id}>
                                                    <Building className="mr-2 h-4 w-4" />Thông tin Trạm
                                                </TabsTrigger>
                                            </TabsList>

                                            <TabsContent value="overview">
                                                <div className="space-y-4 text-sm">
                                                    <div>
                                                        <Label className="text-xs font-semibold text-muted-foreground">Mô tả</Label>
                                                        <p className="mt-1">{selectedModel.description || <span className="italic text-muted-foreground">Không có mô tả.</span>}</p>
                                                    </div>
                                                    <Separator />
                                                    <div className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
                                                        <div>
                                                            <Label className="text-xs font-semibold text-muted-foreground">ID Model</Label>
                                                            <p className="mt-1 font-mono text-xs bg-muted px-2 py-0.5 rounded break-all">{selectedModel.id}</p>
                                                        </div>
                                                        {/* <div>
                                                            <Label className="text-xs font-semibold text-muted-foreground">Đường dẫn File</Label>
                                                            <p className="mt-1 text-xs break-all">{selectedModel.file_path}</p>
                                                        </div> */}
                                                        <div>
                                                            <Label className="text-xs font-semibold text-muted-foreground">Huấn luyện lúc</Label>
                                                            <p className="mt-1">{selectedModel.trained_at ? new Date(selectedModel.trained_at).toLocaleString('vi-VN') : 'Chưa rõ'}</p>
                                                        </div>
                                                        <div>
                                                            <Label className="text-xs font-semibold text-muted-foreground">Tạo lúc</Label>
                                                            <p className="mt-1">{new Date(selectedModel.created_at).toLocaleString('vi-VN')}</p>
                                                        </div>
                                                        <div>
                                                            <Label className="text-xs font-semibold text-muted-foreground">Cập nhật lúc</Label>
                                                            <p className="mt-1">{new Date(selectedModel.updated_at).toLocaleString('vi-VN')}</p>
                                                        </div>
                                                        {selectedModel.deleted_at && (
                                                            <div className="text-destructive">
                                                                <Label className="text-xs font-semibold">Xóa lúc</Label>
                                                                <p className="mt-1">{new Date(selectedModel.deleted_at).toLocaleString('vi-VN')}</p>
                                                            </div>
                                                        )}
                                                    </div>
                                                    <Separator />
                                                    <div>
                                                        <Label className="text-xs font-semibold text-muted-foreground">Danh sách tham số</Label>
                                                        {selectedModel.parameter_list && selectedModel.parameter_list.length > 0 ? (
                                                            <div className="mt-1 flex flex-wrap gap-1">
                                                                {selectedModel.parameter_list.map((param, index) => (
                                                                    <span key={index} className="text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded">
                                                                        {param}
                                                                    </span>
                                                                ))}
                                                            </div>
                                                        ) : (
                                                            <p className="mt-1 italic text-muted-foreground">Không có.</p>
                                                        )}
                                                    </div>
                                                </div>
                                            </TabsContent>

                                            <TabsContent value="station_info">
                                                {selectedStationInfo ? (
                                                    <div className="space-y-3 text-sm">
                                                        <div className="flex items-center gap-2">
                                                            <Building className="h-4 w-4 text-muted-foreground" />
                                                            <strong>Trạm:</strong> {selectedStationInfo.name}
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <Info className="h-4 w-4 text-muted-foreground" />
                                                            <strong>ID Trạm:</strong> <span className="font-mono text-xs bg-muted px-2 py-0.5 rounded">{selectedStationInfo.id}</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <MapPin className="h-4 w-4 text-muted-foreground" />
                                                            <strong>Vị trí:</strong> {selectedStationInfo.location}
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <Globe className="h-4 w-4 text-muted-foreground" />
                                                            <strong>Quốc gia:</strong> {selectedStationInfo.country}
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <List className="h-4 w-4 text-muted-foreground" />
                                                            <strong>Tọa độ:</strong> Vĩ độ: {selectedStationInfo.latitude?.toFixed(6)}, kinh độ: {selectedStationInfo.longitude?.toFixed(6)}
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <CalendarDays className="h-4 w-4 text-muted-foreground" />
                                                            <strong>Ngày tạo trạm:</strong> {new Date(selectedStationInfo.createdAt).toLocaleString('vi-VN')}
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <p className="text-muted-foreground italic">Model này không được liên kết với trạm nào hoặc không tìm thấy thông tin trạm.</p>
                                                )}
                                            </TabsContent>
                                        </Tabs>
                                    </CardContent>
                                </Card>
                            ) : (
                                <div className="flex h-60 items-center justify-center rounded-lg border border-dashed">
                                    <p className="text-muted-foreground">Chọn một model từ danh sách bên trái để xem chi tiết.</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* --- Modals --- */}
                    {/* Update Modal (giữ nguyên) */}
                    {/* ... (Update Modal Dialog) ... */}
                    <Dialog open={isUpdateModalOpen} onOpenChange={setIsUpdateModalOpen}>
                        <DialogContent className="sm:max-w-[480px]">
                            <DialogHeader>
                                <DialogTitle>Cập nhật Model: {currentModelForAction?.name ? (MODEL_DISPLAY_NAMES[currentModelForAction?.name]) : ""}</DialogTitle>
                                <DialogDescription>Chỉnh sửa thông tin chi tiết của model.</DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                                <div className="space-y-1.5">
                                    <Label htmlFor="update-name">Tên Model</Label>
                                    <Input id="update-name" value={updateFormData.name} onChange={(e) => handleFormInputChange('update', 'name', e.target.value)} />
                                </div>
                                <div className="space-y-1.5">
                                    <Label htmlFor="update-description">Mô tả</Label>
                                    <textarea
                                        id="update-description"
                                        value={updateFormData.description}
                                        onChange={(e) => handleFormInputChange('update', 'description', e.target.value)}
                                        className="h-48 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                        placeholder="Nhập mô tả cho model..."
                                    />
                                </div>
                                <div className="flex items-center space-x-2 pt-2">
                                    <Checkbox id="update-availability" checked={updateFormData.availability} onCheckedChange={(checked) => handleFormInputChange('update', 'availability', !!checked)} />
                                    <Label htmlFor="update-availability" className="font-normal">Sẵn sàng (Availability)</Label>
                                </div>
                            </div>
                            <DialogFooter>
                                <DialogClose asChild><Button variant="outline" disabled={isProcessing[`update-${currentModelForAction?.id}`]}>Hủy</Button></DialogClose>
                                <Button type="submit" onClick={handleUpdateSubmit} disabled={isProcessing[`update-${currentModelForAction?.id}`]}>
                                    {isProcessing[`update-${currentModelForAction?.id}`] ? "Đang lưu..." : "Lưu thay đổi"}
                                </Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>


                    {/* Train Modal (giữ nguyên) */}
                    {/* ... (Train Modal Dialog) ... */}
                    {/* Huấn luyện Models */}
<Dialog open={isTrainModalOpen} onOpenChange={setIsTrainModalOpen}>
  <DialogContent className="max-w-screen-md p-6">
    <DialogHeader>
      <DialogTitle>Huấn luyện Models</DialogTitle>
      <DialogDescription>Cấu hình tham số cho quá trình huấn luyện.</DialogDescription>
    </DialogHeader>
    <div className="grid gap-4 py-4">
      <div className="grid grid-cols-[150px_1fr] items-center gap-4">
        <Label htmlFor="train-ratio" className="text-right">Tỷ lệ Train/Test</Label>
        <Input
          id="train-ratio"
          type="number"
          step="0.05"
          min="0.1"
          max="0.9"
          value={trainFormData.train_test_ratio}
          onChange={(e) => handleFormInputChange('train', 'train_test_ratio', e.target.value)}
        />
      </div>

      <div className="grid grid-cols-[150px_1fr] items-center gap-4">
        <Label htmlFor="train-date-tag" className="text-right">Phiên bản</Label>
        <Input
          id="train-date-tag"
          value={trainFormData.date_tag}
          onChange={(e) => handleFormInputChange('train', 'date_tag', e.target.value)}
          placeholder="ddmmyy"
        />
      </div>

      <div className="grid grid-cols-[150px_1fr] items-start gap-4">
        <Label className="text-right pt-2">Chọn Trạm</Label>
        <div className="space-y-2 w-full">
          <ScrollArea className="h-72 w-full rounded-md border p-2">
            <div className="space-y-2">
              {stations.length > 0 ? (
                stations.map((station) => (
                  <div key={station.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={`train-station-${station.id}`}
                      checked={trainFormData.place_ids.includes(station.id)}
                      onCheckedChange={() => handleFormInputChange('train', 'place_ids', station.id)}
                    />
                    <Label htmlFor={`train-station-${station.id}`} className="font-normal">
                      {station.name}
                    </Label>
                  </div>
                ))
              ) : (
                <p className="text-xs text-muted-foreground">Đang tải trạm...</p>
              )}
            </div>
          </ScrollArea>
          <p className="text-xs text-muted-foreground">Để trống nếu huấn luyện tất cả trạm.</p>
        </div>
      </div>
    </div>
    <DialogFooter>
      <DialogClose asChild>
        <Button variant="outline" disabled={isProcessing['global-train']}>Hủy</Button>
      </DialogClose>
      <Button type="submit" onClick={handleTrainSubmit} disabled={isProcessing['global-train']}>
        {isProcessing['global-train'] ? "Đang huấn luyện..." : "Bắt đầu huấn luyện"}
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>

                    {/* Predict Modal (Toàn cục) - Sửa phần chọn Trạm */}
                    <Dialog open={isPredictModalOpen} onOpenChange={setIsPredictModalOpen}>
  <DialogContent className="max-w-screen-md p-6">
    <DialogHeader>
      <DialogTitle>Dự đoán Models</DialogTitle>
      <DialogDescription>Cấu hình tham số cho quá trình dự đoán.</DialogDescription>
    </DialogHeader>
    <div className="grid gap-4 py-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="predict-num-step">Num Step (1-14)</Label>
          <Input
            id="predict-num-step"
            type="number"
            min="1"
            max="14"
            value={predictFormData.num_step}
            onChange={(e) => handleFormInputChange('predict', 'num_step', e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="predict-freq-days">Freq Days (1-14)</Label>
          <Input
            id="predict-freq-days"
            type="number"
            min="1"
            max="14"
            value={predictFormData.freq_days}
            onChange={(e) => handleFormInputChange('predict', 'freq_days', e.target.value)}
          />
        </div>
      </div>

      <div className="grid grid-cols-[150px_1fr] items-start gap-4">
        <Label htmlFor="model-type-group" className="text-right pt-2">Model Type</Label>
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">Chọn loại model sẽ sử dụng để dự đoán (chọn một).</p>
          <RadioGroup
            id="model-type-group"
            value={predictFormData.model_types[0] || ''}
            onValueChange={(newValue) => handleFormInputChange('predict', 'model_types', newValue)}
            className="flex flex-wrap gap-x-4 gap-y-2"
          >
            {MODEL_TYPES_OPTIONS.map((type) => (
              <div key={type} className="flex items-center space-x-2">
                <RadioGroupItem value={type} id={`predict-type-${type}`} />
                <Label htmlFor={`predict-type-${type}`} className="font-normal">{MODEL_DISPLAY_NAMES[type]}</Label>
              </div>
            ))}
          </RadioGroup>
        </div>
      </div>

      <div className="grid grid-cols-[150px_1fr] items-start gap-4">
        <Label className="text-right pt-2">Chọn Trạm</Label>
        <div className="space-y-2 w-full">
          <ScrollArea className="h-72 w-full rounded-md border p-2">
            <div className="space-y-2">
              {stationsAvailableForPrediction.length > 0 ? (
                stationsAvailableForPrediction.map((station) => (
                  <div key={station.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={`predict-station-${station.id}`}
                      checked={predictFormData.place_ids.includes(station.id)}
                      onCheckedChange={() => handleFormInputChange('predict', 'place_ids', station.id)}
                    />
                    <Label htmlFor={`predict-station-${station.id}`} className="font-normal">
                      {station.name}
                    </Label>
                  </div>
                ))
              ) : (
                <p className="text-xs text-muted-foreground">Không có trạm nào có model AI liên kết để dự đoán.</p>
              )}
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
    <DialogFooter>
      <DialogClose asChild>
        <Button variant="outline" disabled={isProcessing['global-predict']}>Hủy</Button>
      </DialogClose>
      <Button type="submit" onClick={handlePredictSubmit} disabled={isProcessing['global-predict']}>
        {isProcessing['global-predict'] ? "Đang dự đoán..." : "Bắt đầu dự đoán"}
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>

                </main>
            </div>
        </TooltipProvider>
    );
}