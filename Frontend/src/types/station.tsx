export interface Station {
    name: string;
    lat: number;
    lng: number;
    wqi: number;
    status: string;
    recommendation: string;
    time: string;
    trend: number[];
    prediction: number[];
    feature: Indicator[];
  }

interface Indicator {
  name: string;
  trend: number[];
  prediction: number[];
}