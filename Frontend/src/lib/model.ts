import { ModelAI } from "@/types/models";

import axios from "axios";

interface param {
    offset?: number,
    limit?: number
}
export async function getAllAIModels(options: param = {}): Promise<ModelAI[]> {
    try {
      const params = new URLSearchParams();
      if (options.limit !== undefined) {params.append("options.limit", options.limit.toString());}
      if (options.offset !== undefined) {params.append("options.offset", options.offset.toString());}
      const baseUrl = `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/models/`;
      const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;
      const res = await axios.get(url);
      return res.data.items;
    } catch (error) {
      console.error(error);
      throw new Error("Failed to get AI models");
    }
  }

export async function getListAIModelsByStationId(stationId:string ,options: param = {}): Promise<ModelAI[]> {
    try {
      const params = new URLSearchParams();
      if (options.limit !== undefined) {params.append("options.limit", options.limit.toString());}
      if (options.offset !== undefined) {params.append("options.offset", options.offset.toString());}
      const baseUrl = `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/models/station/${stationId}`;
      const url = params.toString() ? `${baseUrl}?${params.toString()}` : baseUrl;
      const res = await axios.get(url);
      return res.data.items;
    } catch (error) {
      console.error(error);
      throw new Error("Failed to get stations");
    }
  }

export async function getAIModelById(modelId:string): Promise<ModelAI> {
    try {
      const baseUrl = `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/models/${modelId}`;
      const res = await axios.get(baseUrl);
      return res.data;
    } catch (error) {
      console.error(error);
      throw new Error("Failed to get AI Model");
    }
  }

  export async function updateAIModel(update: Partial<ModelAI> , modelId: string): Promise<number> {
    try {
      const res = await axios.put( 
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/models/${modelId}`,
        update, 
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      return res.status;
    } catch (error) {
      console.error("Failed to update model:", error);
      throw new Error("Failed to update model");
    }
  }
  
  export async function deleteAIModel(modelId : string): Promise<number> {
    try {
      const url = `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/models/${modelId }`;
      const res = await axios.delete(url);
      return res.status;
    } catch (error) {
      console.error("Failed to delete model:", error);
      throw new Error("Failed to delete model");
    }
  }
  interface trainParam {
    train_test_ratio: number, // default is 0.7
    place_ids? : string[],   // empty for train all station
    date_tag: string          //default is today : A tag, often date-based (e.g., ddmmyy), to version the models.
  }
  export async function trainStationsModels(param: trainParam): Promise<ModelAI[]> {
    try {
        const urlEncodedData = new URLSearchParams(param as any).toString();
        console.log(urlEncodedData);
      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/models/train-station-models`,
        urlEncodedData,
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
        }
      );
      return res.data;
    } catch (error) {
      console.error("Failed to train model:", error);
      throw new Error("Failed to train model");
    }
  }
  interface predictParam {
    place_ids? : string[],   // empty for train all stations
    num_step: number; // default is 7 (range  is from 1 to 14)
    freq_days: number; // default is 7 (range  is from 1 to 14)
    model_types : string[]  //List of model types (e.g., 'xgb', 'rf') to use for prediction.
  }
  export async function predictStationsModels(param: predictParam): Promise<number> {
    try {
        
        const urlEncodedData = new URLSearchParams(param as any).toString();
        console.log(urlEncodedData);
      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/models/predict-station-models`,
        urlEncodedData,
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
        }
      );
      return res.status;
    } catch (error) {
      console.error("Failed to predict model:", error);
      throw new Error("Failed to predict model");
    }
  }
