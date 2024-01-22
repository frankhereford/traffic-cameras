import { useQuery } from "@tanstack/react-query"
import type { UseQueryResult } from "@tanstack/react-query"

// Define a type for your data, modify it according to your data structure
export interface SocrataData {
  camera_id: string
  location_name: string
  camera_status: string
  camera_mfg: string
  atd_location_id: string
  council_district: string
  jurisdiction_label: string
  location_type: string
  primary_st_block: string
  primary_st: string
  cross_st_block: string
  cross_st: string
  coa_intersection_id: string
  modified_date: string
  screenshot_address: string
  funding: string
  id: string
  location: {
    type: string
    coordinates: number[]
  }
  ":@computed_region_jcrc_4uuy"?: string
  ":@computed_region_m2th_e4b7"?: string
  ":@computed_region_rxpj_nzrk"?: string
  ":@computed_region_8spj_utxs"?: string
  ":@computed_region_q9nd_rr82"?: string
  ":@computed_region_e9j2_6w3z"?: string
  signal_eng_area?: string
}

const useGetSocrataData = (): UseQueryResult<SocrataData[], Error> => {
  const fetchSocrataData = async (): Promise<SocrataData[]> => {
    const url = "https://data.austintexas.gov/resource/b4k4-adkb.json"

    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = (await response.json()) as SocrataData[]

    // Filter out duplicates based on camera_id
    const uniqueCameraIdData = data.reduce(
      (acc: SocrataData[], current: SocrataData) => {
        const duplicate = acc.find(
          (item) => item.camera_id === current.camera_id,
        )
        return duplicate ? acc : [...acc, current]
      },
      [],
    )

    // Filter out duplicates based on location_name
    const uniqueData = uniqueCameraIdData.reduce(
      (acc: SocrataData[], current: SocrataData) => {
        const duplicate = acc.find(
          (item) => item.location_name === current.location_name,
        )
        return duplicate ? acc : [...acc, current]
      },
      [],
    )

    return uniqueData
  }

  return useQuery<SocrataData[], Error>({
    queryKey: ["socrataData"],
    queryFn: fetchSocrataData,
  })
}

export default useGetSocrataData
