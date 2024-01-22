import { useQuery, UseQueryResult } from "@tanstack/react-query"

// Define a type for your data, modify it according to your data structure
interface SocrataData {
  // example fields
  id: string
  name: string
  // add other fields as per your data structure
}

const useGetSocrataData = (): UseQueryResult<SocrataData[], Error> => {
  const fetchSocrataData = async (): Promise<SocrataData[]> => {
    const url = "https://data.austintexas.gov/resource/b4k4-adkb.json"

    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    return response.json() as Promise<SocrataData[]>
  }

  return useQuery<SocrataData[], Error>({
    queryKey: ["socrataData"],
    queryFn: fetchSocrataData,
  })
}

export default useGetSocrataData
