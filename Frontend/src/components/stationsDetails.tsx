import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, Activity, Clock, MapPin, BarChart } from "lucide-react"
import { Station } from "@/types/station"
interface StationDetailsProps {
  selectedStation: Station | null
}

export default function StationDetails({ selectedStation }: StationDetailsProps) {
  if (!selectedStation) return null

  const getStatusColor = (status: Station["status"]) => {
    switch (status) {
      case "Rất tốt":
        return "bg-emerald-100 text-emerald-900 border-emerald-200"
      case "Tốt":
        return "bg-green-100 text-green-800 border-green-200"
      case "Bình thường":
        return "bg-amber-100 text-amber-800 border-amber-200"
      case "Không tốt":
        return "bg-orange-100 text-orange-800 border-orange-200"
      case "Nguy hiểm":
        return "bg-red-100 text-red-800 border-red-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  return (
    <Card className="w-full bg-white rounded-lg shadow-sm border border-gray-200">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-800">Trạm quan trắc: {selectedStation.name}</h2>
          <Badge variant="outline" className={`${getStatusColor(selectedStation.status)} px-4 py-1 text-base rounded-lg`}>
            {selectedStation.status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-50 p-4 rounded-md border border-gray-100">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-5 h-5 text-blue-600" />
              <h3 className="font-medium text-gray-700">Chỉ số chất lượng nước</h3>
            </div>
            <div className="flex items-center">
              <div className="text-3xl font-bold text-blue-700">{selectedStation.wqi}</div>
              <div className="ml-2 text-sm text-gray-500">WQI</div>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-md border border-gray-100">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-amber-600" />
              <h3 className="font-medium text-gray-700">Khuyến cáo</h3>
            </div>
            <p className="text-sm text-gray-700">{selectedStation.recommendation}</p>
          </div>
        </div>
      </CardContent>

      <CardFooter className="border-t border-gray-100 pt-3 text-xs text-gray-500 flex justify-between items-center">
        <div className="flex items-center">
          <Clock className="w-4 h-4 mr-1" />
          <span>Cập nhật gần nhất vào: {selectedStation.time}</span>
        </div>
        <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-100">
          <BarChart className="w-3 h-3 mr-1" />
          Dữ liệu đã được kiểm duyệt
        </Badge>
      </CardFooter>
    </Card>
  )
}

