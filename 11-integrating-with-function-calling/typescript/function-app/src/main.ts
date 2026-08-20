import OpenAI from "openai";
import axios, { AxiosError } from "axios";
import * as dotenv from "dotenv";

dotenv.config();

// SECURITY: Validate required environment variables
const endpoint = process.env.AZURE_OPENAI_ENDPOINT;
const azureApiKey = process.env.AZURE_OPENAI_API_KEY;
const bingMapsBaseUrl = process.env.BING_MAPS_BASE_URL;
const bingApiKey = process.env.BING_API_KEY;

if (!endpoint || !azureApiKey) {
  throw new Error("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables are required");
}

if (!bingMapsBaseUrl || !bingApiKey) {
  throw new Error("BING_MAPS_BASE_URL and BING_API_KEY environment variables are required");
}

// SECURITY: Validate URL format
function isValidUrl(urlString: string): boolean {
  try {
    const url = new URL(urlString);
    return url.protocol === 'https:';
  } catch {
    return false;
  }
}

if (!isValidUrl(endpoint) || !isValidUrl(bingMapsBaseUrl)) {
  throw new Error("Invalid URL format for endpoints. Must be HTTPS URLs.");
}

async function findWeather(currentLocation: string, placeType: string): Promise<unknown> {
  // SECURITY: URL-encode ALL parameters to prevent injection
  const params = new URLSearchParams({
    query: currentLocation,
    type: placeType,
    key: bingApiKey!
  });
  const url = `${bingMapsBaseUrl}?${params.toString()}`;

  try {
    // SECURITY: Add timeout to prevent hanging requests
    const response = await axios.get(url, { timeout: 10000 });
    return response.data;
  } catch (error) {
    // SECURITY: Avoid propagating the full error, which may contain the API key,
    // but do propagate the failure so the caller can't mistake it for a result.
    if (error instanceof AxiosError) {
      throw new Error(`Weather API request failed: ${error.message} (Status: ${error.response?.status || 'N/A'})`);
    }
    throw new Error('Weather API request failed: an unexpected error occurred');
  }
}

// Responses API tools use a flat schema: { type, name, description, parameters }
const getCurrentWeatherTool = {
  type: "function" as const,
  name: "findWeather",
  description: "Get the current weather in a given location",
  parameters: {
    type: "object",
    properties: {
      location: {
        type: "string",
        description: "The city and state, e.g. San Francisco, CA"
      },
      unit: {
        type: "string",
        enum: ["C", "F"], // Celsius or Fahrenheit
      },
    },
    required: ["location"],
  },
};

async function main() {
  console.log("== Chat App with Functions (Responses API) ==");

  // The Responses API is served from the Azure OpenAI (Microsoft Foundry) v1 endpoint.
  const client = new OpenAI({
    apiKey: azureApiKey,
    baseURL: `${endpoint.replace(/\/$/, '')}/openai/v1/`,
  });
  const deploymentName = process.env.AZURE_OPENAI_DEPLOYMENT || "gpt-5-mini";

  const userParams = {
    location: "New York",
    unit: "C"
  };

  const result = await client.responses.create({
    model: deploymentName,
    input: [
      {
        role: "user",
        content: `What's the weather in ${userParams.location}, ${userParams.unit}?`,
      },
    ],
    tools: [getCurrentWeatherTool],
    store: false,
  });

  for (const item of result.output) {
    if (item.type === "function_call") {
      console.log(item);

      // SECURITY: Safely parse JSON with validation
      let parsedArgs: { location?: string; unit?: string };
      try {
        parsedArgs = JSON.parse(item.arguments || '{}');
      } catch (parseError) {
        const message = parseError instanceof Error ? parseError.message : String(parseError);
        throw new Error(`Failed to parse function arguments: ${message}`);
      }

      const { location, unit } = parsedArgs;
      if (!location) {
        throw new Error('The model did not provide the required location parameter');
      }

      const response = await findWeather(location, unit || 'C');
      console.log("Result from Bing Maps API..: ", response);
    }
  }
}

// Report failures and exit with a non-zero status instead of swallowing them.
main().catch((error) => {
  console.error("The sample encountered an error...:", error);
  process.exitCode = 1;
});