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
  }