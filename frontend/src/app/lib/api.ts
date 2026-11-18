const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error("Backend health check failed");
    }

    return await response.json();
  } catch (error) {
    console.error("Backend connection error:", error);

    return null;
  }
}

export async function getModelInfo() {
  try {
    const response = await fetch(`${API_BASE_URL}/model/info`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error("Could not retrieve model information");
    }

    return await response.json();
  } catch (error) {
    console.error("Model information error:", error);

    return null;
  }
}

export async function predictTraffic(data: Record<string, number>) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",

    headers: {
      "Content-Type": "application/json",
    },

    body: JSON.stringify({ data }),
  });

  if (!response.ok) {
    throw new Error("Prediction request failed");
  }

  return await response.json();
}