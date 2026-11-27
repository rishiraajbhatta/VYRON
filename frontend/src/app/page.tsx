"use client";

import { useEffect, useState, type ReactNode } from "react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const navigation = [
  "Home",
  "Overview",
  "Live Analyzer",
  "Virtual Devices",
  "Threat Intelligence",
  "Analytics",
  "System",
];

type SampleType = "BENIGN" | "MALICIOUS";
type SimulationMode = "dataset_replay" | "balanced_demo";

type DatasetSample = {
  success: boolean;
  expected_class: SampleType;
  original_label: string;
  sample_source: string;
  total_features: number;
  data: Record<string, number>;
};

type Prediction = {
  id: number;
  prediction: SampleType;
  probability: number;
  threshold: number;
  time: string;
  expectedClass: SampleType | null;
  originalLabel: string | null;
};

type SimulationEvent = {
  id: number;
  timestamp: string;
  actual_class: SampleType;
  predicted_class: SampleType;
  original_label: string;
  malicious_probability: number;
  threshold: number;
  matches_dataset: boolean;
  detection_model: string;
  sample_source: string;
  simulation_mode?: SimulationMode;
  simulation_mode_label?: string;
};

type SimulationStatus = {
  running: boolean;
  total_events: number;
  interval_seconds: number;
  stored_events: number;
  detection_model: string;
  mode?: SimulationMode;
  mode_label?: string;
  available_modes?: SimulationMode[];
};

type VirtualDeviceStatus = "online" | "offline" | "simulating";

type VirtualDevice = {
  device_id: string;
  name: string;
  device_type: string;
  description: string;
  ip_address: string;
  status: VirtualDeviceStatus;
  location: string;
  created_at: string;
  last_activity: string | null;
};

type DeviceAnalysis = {
  model: string;
  prediction: SampleType;
  malicious_probability: number;
  threshold: number;
  actual_class: SampleType;
  original_label: string;
  matches_dataset: boolean;
  sample_source: string;
  analysis_type: string;
  note: string;
};

type VirtualDeviceEvent = {
  event_id: number;
  device_id: string;
  device_name: string;
  device_type: string;
  activity: string;
  scenario: string;
  timestamp: string;
  status: string;
  analysis: DeviceAnalysis | null;
};

type DeviceStatistics = {
  total_devices: number;
  online_devices: number;
  simulating_devices: number;
  offline_devices: number;
};

function VyronLogo({ large = false }: { large?: boolean }) {
  return (
    <svg
      viewBox="0 0 100 80"
      fill="none"
      className={large ? "h-28 w-32" : "h-12 w-14"}
    >
      <defs>
        <linearGradient id="vyronGradient" x1="12" y1="5" x2="88" y2="75">
          <stop stopColor="#DDD6FE" />
          <stop offset="0.45" stopColor="#A78BFA" />
          <stop offset="1" stopColor="#67E8F9" />
        </linearGradient>
        <linearGradient id="vyronOrbit" x1="0" y1="0" x2="100" y2="80">
          <stop stopColor="#C084FC" />
          <stop offset="1" stopColor="#22D3EE" />
        </linearGradient>
      </defs>

      <ellipse
        cx="50"
        cy="42"
        rx="45"
        ry="17"
        transform="rotate(-28 50 42)"
        stroke="url(#vyronOrbit)"
        strokeWidth="2"
      />

      <path
        d="M13 8L48 66L84 8H64L48 39L31 8H13Z"
        fill="url(#vyronGradient)"
      />

      <path
        d="M31 14L48 43L65 14"
        stroke="white"
        strokeWidth="1"
        opacity="0.8"
      />

      <path
        d="M12 57C25 46 42 31 62 20C76 12 87 10 91 13"
        stroke="#C4B5FD"
        strokeWidth="1.5"
        strokeLinecap="round"
      />

      <circle cx="13" cy="57" r="2.5" fill="#DDD6FE" />
      <circle cx="90" cy="13" r="2.5" fill="#67E8F9" />
    </svg>
  );
}

function NetworkGlobe() {
  const nodes = [
    [75, 110],
    [140, 80],
    [210, 125],
    [300, 85],
    [100, 205],
    [190, 200],
    [280, 190],
    [130, 290],
    [225, 315],
    [320, 270],
  ];

  return (
    <div className="relative flex min-h-[390px] items-center justify-center overflow-hidden md:min-h-[620px]">
      <div className="absolute h-[420px] w-[420px] rounded-full bg-violet-600/25 blur-[110px] md:h-[600px] md:w-[600px]" />

      <div className="absolute h-[300px] w-[300px] rounded-full bg-blue-500/25 blur-[100px] md:h-[450px] md:w-[450px]" />

      <div className="relative h-[310px] w-[310px] md:h-[470px] md:w-[470px]">
        <div className="absolute inset-0 rounded-full border border-blue-300/30 shadow-[0_0_100px_rgba(59,130,246,0.2)]" />

        <div className="absolute inset-[8%] overflow-hidden rounded-full border border-blue-200/40 bg-gradient-to-br from-[#20204b] via-[#101b3b] to-[#080c20] shadow-[inset_-35px_-20px_80px_rgba(59,130,246,0.3),inset_20px_0_60px_rgba(139,92,246,0.3)]">
          <div className="absolute -left-[15%] top-[5%] h-[70%] w-[65%] rounded-full bg-violet-500/30 blur-[55px]" />

          <div className="absolute -right-[20%] bottom-0 h-[70%] w-[70%] rounded-full bg-blue-400/30 blur-[55px]" />

          <svg
            viewBox="0 0 400 400"
            className="absolute inset-0 h-full w-full opacity-65"
          >
            <defs>
              <linearGradient
                id="gridGradient"
                x1="0"
                y1="0"
                x2="1"
                y2="1"
              >
                <stop stopColor="#C4B5FD" />
                <stop offset="1" stopColor="#67E8F9" />
              </linearGradient>
            </defs>

            {[70, 130, 200, 270, 330].map((y) => (
              <ellipse
                key={y}
                cx="200"
                cy="200"
                rx="185"
                ry={y / 2}
                stroke="url(#gridGradient)"
                strokeWidth="0.8"
                fill="none"
              />
            ))}

            {[55, 100, 150, 200, 250, 300, 345].map((x) => (
              <ellipse
                key={x}
                cx="200"
                cy="200"
                rx={Math.abs(x - 200) + 20}
                ry="190"
                stroke="url(#gridGradient)"
                strokeWidth="0.8"
                fill="none"
              />
            ))}
          </svg>

          <svg viewBox="0 0 400 400" className="absolute inset-0 h-full w-full">
            <g
              stroke="#A5B4FC"
              strokeWidth="1.2"
              opacity="0.8"
              fill="none"
            >
              <path d="M75 110L140 80L210 125L300 85" />
              <path d="M75 110L100 205L190 200L280 190" />
              <path d="M100 205L130 290L225 315L320 270" />
              <path d="M140 80L190 200L225 315" />
              <path d="M210 125L280 190L320 270" />
            </g>

            {nodes.map(([x, y], index) => (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="3.5"
                fill={index % 2 === 0 ? "#C4B5FD" : "#67E8F9"}
              />
            ))}
          </svg>
        </div>

        <div className="absolute inset-[-9%] animate-[spin_25s_linear_infinite] rounded-full border border-violet-300/50 [transform:rotateX(65deg)]" />

        <div className="absolute inset-[-14%] animate-[spin_35s_linear_infinite_reverse] rounded-full border border-blue-300/40 [transform:rotateX(65deg)_rotateZ(55deg)]" />

        <div className="absolute inset-[-4%] rounded-full border border-cyan-200/30 [transform:rotateY(65deg)]" />

        <div className="absolute inset-0 flex items-center justify-center">
          <div className="drop-shadow-[0_0_35px_rgba(167,139,250,0.9)]">
            <VyronLogo large />
          </div>
        </div>

        <div className="absolute right-[5%] top-[25%] h-2 w-2 rounded-full bg-cyan-200 shadow-[0_0_20px_5px_rgba(34,211,238,0.8)]" />

        <div className="absolute bottom-[20%] left-[2%] h-2 w-2 rounded-full bg-violet-300 shadow-[0_0_20px_5px_rgba(167,139,250,0.8)]" />
      </div>

      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 whitespace-nowrap text-[9px] tracking-[0.35em] text-indigo-200/80">
        NETWORK INTELLIGENCE · VYRON CORE
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="mb-6 flex items-center gap-3">
      <span className="h-[2px] w-10 bg-gradient-to-r from-violet-400 to-cyan-300" />

      <span className="text-[10px] font-medium tracking-[0.3em] text-violet-200">
        {children}
      </span>
    </div>
  );
}

export default function Home() {
  const [activePage, setActivePage] = useState("Home");

  const [backendStatus, setBackendStatus] = useState<
    "checking" | "online" | "offline"
  >("checking");

  const [modelFeatures, setModelFeatures] = useState<number | null>(null);
  const [modelThreshold, setModelThreshold] = useState<number | null>(null);

  // Manual analyzer state

  const [trafficData, setTrafficData] = useState<Record<string, number>>({});
  const [expectedClass, setExpectedClass] = useState<SampleType | null>(null);
  const [originalLabel, setOriginalLabel] = useState<string | null>(null);
  const [sampleSource, setSampleSource] = useState<string | null>(null);

  const [loadingSample, setLoadingSample] = useState(false);
  const [loadedSample, setLoadedSample] = useState<SampleType | null>(null);

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");

  // Continuous simulation state

  const [simulationRunning, setSimulationRunning] = useState(false);
  const [simulationLoading, setSimulationLoading] = useState(false);
  const [simulationMode, setSimulationMode] =
    useState<SimulationMode>("dataset_replay");

  const [simulationStatus, setSimulationStatus] =
    useState<SimulationStatus | null>(null);

  const [simulationEvents, setSimulationEvents] = useState<
    SimulationEvent[]
  >([]);

  const [simulationError, setSimulationError] = useState("");

  // Virtual IoT device state
  const [virtualDevices, setVirtualDevices] = useState<VirtualDevice[]>([]);
  const [deviceStatistics, setDeviceStatistics] = useState<DeviceStatistics>({
    total_devices: 0,
    online_devices: 0,
    simulating_devices: 0,
    offline_devices: 0,
  });
  const [deviceEvents, setDeviceEvents] = useState<VirtualDeviceEvent[]>([]);
  const [deviceScenario, setDeviceScenario] = useState<Record<string, string>>({});
  const [deviceBusyId, setDeviceBusyId] = useState<string | null>(null);
  const [deviceError, setDeviceError] = useState("");
  const [deviceMessage, setDeviceMessage] = useState("");

  // ==================================================
  // BACKEND HEALTH CHECK
  // ==================================================

  useEffect(() => {
    let mounted = true;

    const checkStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`, {
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error("Health check failed");
        }

        const health = await response.json();

        if (!mounted) return;

        if (health?.status === "healthy" && health?.model_loaded) {
          setBackendStatus("online");
          setModelFeatures(health.features);
          setModelThreshold(health.threshold);
        } else {
          setBackendStatus("offline");
        }
      } catch {
        if (!mounted) return;

        setBackendStatus("offline");
        setModelFeatures(null);
        setModelThreshold(null);
      }
    };

    checkStatus();

    const interval = setInterval(checkStatus, 10000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // ==================================================
  // SIMULATION STATUS + LIVE EVENT POLLING
  // ==================================================

  useEffect(() => {
    if (backendStatus !== "online") return;

    let mounted = true;
    let lastEventId = 0;

    const fetchSimulationData = async () => {
      try {
        const [statusResponse, eventsResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/simulation/status`, {
            cache: "no-store",
          }),

          fetch(
            `${API_BASE_URL}/simulation/events?after_id=${lastEventId}&limit=100`,
            {
              cache: "no-store",
            }
          ),
        ]);

        if (!statusResponse.ok || !eventsResponse.ok) {
          throw new Error("Unable to retrieve simulation data.");
        }

        const status: SimulationStatus = await statusResponse.json();
        const eventResult = await eventsResponse.json();

        if (!mounted) return;

        setSimulationStatus(status);
        setSimulationRunning(status.running);

        // Keep the selector in sync if a simulation is already running.
        if (
          status.running &&
          (status.mode === "dataset_replay" || status.mode === "balanced_demo")
        ) {
          setSimulationMode(status.mode);
        }

        const newEvents: SimulationEvent[] = eventResult.events || [];

        if (newEvents.length > 0) {
          lastEventId = Math.max(
            lastEventId,
            ...newEvents.map((event) => event.id)
          );

          setSimulationEvents((previous) =>
            [...newEvents, ...previous].slice(0, 100)
          );
        }
      } catch (err) {
        if (!mounted) return;

        setSimulationError(
          err instanceof Error
            ? err.message
            : "Unable to retrieve live events."
        );
      }
    };

    fetchSimulationData();

    const interval = setInterval(fetchSimulationData, 1200);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [backendStatus]);

  // ==================================================
  // VIRTUAL DEVICE DATA POLLING
  // ==================================================

  useEffect(() => {
    if (backendStatus !== "online") return;

    let mounted = true;

    const fetchDeviceData = async () => {
      try {
        const [devicesResponse, statusResponse, eventsResponse] =
          await Promise.all([
            fetch(`${API_BASE_URL}/devices/`, { cache: "no-store" }),
            fetch(`${API_BASE_URL}/devices/status`, { cache: "no-store" }),
            fetch(`${API_BASE_URL}/devices/events?limit=50`, { cache: "no-store" }),
          ]);

        if (!devicesResponse.ok || !statusResponse.ok || !eventsResponse.ok) {
          throw new Error("Unable to retrieve virtual device information.");
        }

        const [devicesResult, statusResult, eventsResult] = await Promise.all([
          devicesResponse.json(),
          statusResponse.json(),
          eventsResponse.json(),
        ]);

        if (!mounted) return;

        setVirtualDevices(devicesResult.devices ?? []);
        setDeviceStatistics({
          total_devices: statusResult.total_devices ?? 0,
          online_devices: statusResult.online_devices ?? 0,
          simulating_devices: statusResult.simulating_devices ?? 0,
          offline_devices: statusResult.offline_devices ?? 0,
        });
        setDeviceEvents(eventsResult.events ?? []);
      } catch (err) {
        if (!mounted) return;
        setDeviceError(
          err instanceof Error
            ? err.message
            : "Unable to load virtual device data."
        );
      }
    };

    fetchDeviceData();
    const interval = setInterval(fetchDeviceData, 5000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [backendStatus]);

  // ==================================================
  // NAVIGATION
  // ==================================================

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
    });
  };

  const handleNavigation = (item: string) => {
    setActivePage(item);

    const sections: Record<string, string> = {
      Home: "home",
      Overview: "overview",
      "Live Analyzer": "analyzer",
      "Virtual Devices": "devices",
      "Threat Intelligence": "threats",
      Analytics: "analytics",
      System: "system",
    };

    scrollTo(sections[item] || "home");
  };

  // ==================================================
  // LOAD REAL DATASET SAMPLES
  // ==================================================

  const loadSample = async (type: "benign" | "malicious") => {
    setLoadingSample(true);
    setError("");
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE_URL}/samples/${type}`, {
        cache: "no-store",
      });

      const result: DatasetSample = await response.json();

      if (!response.ok || !result.success) {
        throw new Error("Unable to load dataset sample.");
      }

      setTrafficData(result.data);
      setExpectedClass(result.expected_class);
      setOriginalLabel(result.original_label);
      setSampleSource(result.sample_source);
      setLoadedSample(result.expected_class);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load the dataset sample."
      );
    } finally {
      setLoadingSample(false);
    }
  };

  // ==================================================
  // MANUAL FEATURE EDITING
  // ==================================================

  const handleFeatureChange = (key: string, value: string) => {
    setTrafficData((previous) => ({
      ...previous,
      [key]: Number(value),
    }));

    setExpectedClass(null);
    setOriginalLabel(null);
    setSampleSource(null);
    setLoadedSample(null);
    setPrediction(null);
  };

  // ==================================================
  // MANUAL XGBOOST ANALYSIS
  // ==================================================

  const runAnalysis = async () => {
    if (Object.keys(trafficData).length !== 38) {
      setError("Please load a complete 38-feature dataset sample first.");
      return;
    }

    setAnalyzing(true);
    setError("");
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          data: trafficData,
        }),
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(
          result?.detail || result?.message || "Prediction request failed."
        );
      }

      const predictionResult: Prediction = {
        id: Date.now(),
        prediction: result.result.prediction,
        probability: result.result.malicious_probability,
        threshold: result.result.threshold,
        time: new Date().toLocaleTimeString(),
        expectedClass,
        originalLabel,
      };

      setPrediction(predictionResult);

      setHistory((previous) =>
        [predictionResult, ...previous].slice(0, 10)
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to communicate with the detection engine."
      );
    } finally {
      setAnalyzing(false);
    }
  };

  // ==================================================
  // START SIMULATION
  // ==================================================

  const startSimulation = async () => {
    setSimulationLoading(true);
    setSimulationError("");

    try {
      const response = await fetch(`${API_BASE_URL}/simulation/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          interval_seconds: 1,
          mode: simulationMode,
        }),
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(
          result?.detail || "Unable to start traffic simulation."
        );
      }

      setSimulationRunning(true);
      setSimulationStatus((previous) => ({
        running: true,
        total_events: previous?.total_events ?? 0,
        interval_seconds: 1,
        stored_events: previous?.stored_events ?? 0,
        detection_model: previous?.detection_model ?? "XGBoost",
        mode: simulationMode,
        mode_label:
          simulationMode === "balanced_demo" ? "Balanced Demo" : "Dataset Replay",
      }));
    } catch (err) {
      setSimulationError(
        err instanceof Error
          ? err.message
          : "Unable to start simulation."
      );
    } finally {
      setSimulationLoading(false);
    }
  };

  // ==================================================
  // STOP SIMULATION
  // ==================================================

  const stopSimulation = async () => {
    setSimulationLoading(true);
    setSimulationError("");

    try {
      const response = await fetch(`${API_BASE_URL}/simulation/stop`, {
        method: "POST",
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(
          result?.detail || "Unable to stop traffic simulation."
        );
      }

      setSimulationRunning(false);
    } catch (err) {
      setSimulationError(
        err instanceof Error
          ? err.message
          : "Unable to stop simulation."
      );
    } finally {
      setSimulationLoading(false);
    }
  };

  // ==================================================
  // VIRTUAL DEVICE ACTIONS
  // ==================================================

  const refreshDeviceData = async () => {
    const [devicesResponse, statusResponse, eventsResponse] = await Promise.all([
      fetch(`${API_BASE_URL}/devices/`, { cache: "no-store" }),
      fetch(`${API_BASE_URL}/devices/status`, { cache: "no-store" }),
      fetch(`${API_BASE_URL}/devices/events?limit=50`, { cache: "no-store" }),
    ]);

    if (!devicesResponse.ok || !statusResponse.ok || !eventsResponse.ok) {
      throw new Error("Unable to refresh virtual device data.");
    }

    const [devicesResult, statusResult, eventsResult] = await Promise.all([
      devicesResponse.json(),
      statusResponse.json(),
      eventsResponse.json(),
    ]);

    setVirtualDevices(devicesResult.devices ?? []);
    setDeviceStatistics({
      total_devices: statusResult.total_devices ?? 0,
      online_devices: statusResult.online_devices ?? 0,
      simulating_devices: statusResult.simulating_devices ?? 0,
      offline_devices: statusResult.offline_devices ?? 0,
    });
    setDeviceEvents(eventsResult.events ?? []);
  };

  const controlVirtualDevice = async (
    deviceId: string,
    action: "start" | "stop"
  ) => {
    setDeviceBusyId(deviceId);
    setDeviceError("");
    setDeviceMessage("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/devices/${deviceId}/${action}`,
        { method: "POST" }
      );
      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.detail || result.message || `Unable to ${action} device.`);
      }

      setDeviceMessage(result.message || `Device ${action}ed successfully.`);
      await refreshDeviceData();
    } catch (err) {
      setDeviceError(
        err instanceof Error ? err.message : "Device operation failed."
      );
    } finally {
      setDeviceBusyId(null);
    }
  };

  const generateDeviceActivity = async (deviceId: string) => {
    setDeviceBusyId(deviceId);
    setDeviceError("");
    setDeviceMessage("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/devices/${deviceId}/simulate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scenario: deviceScenario[deviceId] || "normal",
          }),
        }
      );
      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.detail || result.message || "Unable to generate device activity.");
      }

      setDeviceMessage(result.message || "Virtual device activity generated.");
      await refreshDeviceData();
    } catch (err) {
      setDeviceError(
        err instanceof Error ? err.message : "Unable to generate device activity."
      );
    } finally {
      setDeviceBusyId(null);
    }
  };

  const analyzeVirtualDevice = async (deviceId: string) => {
    setDeviceBusyId(deviceId);
    setDeviceError("");
    setDeviceMessage("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/devices/${deviceId}/analyze`,
        { method: "POST" }
      );
      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.detail || result.message || "XGBoost analysis failed.");
      }

      setDeviceMessage("XGBoost analysis completed successfully.");
      await refreshDeviceData();
    } catch (err) {
      setDeviceError(
        err instanceof Error ? err.message : "Unable to analyze device event."
      );
    } finally {
      setDeviceBusyId(null);
    }
  };

  // ==================================================
  // LIVE SIMULATION STATISTICS
  // ==================================================

  const benignCount = simulationEvents.filter(
    (event) => event.predicted_class === "BENIGN"
  ).length;

  const maliciousCount = simulationEvents.filter(
    (event) => event.predicted_class === "MALICIOUS"
  ).length;

  const verifiedCount = simulationEvents.filter(
    (event) => event.matches_dataset
  ).length;

  const verificationRate =
    simulationEvents.length > 0
      ? (verifiedCount / simulationEvents.length) * 100
      : 0;

  const latestEvent = simulationEvents[0];

  const statusText =
    backendStatus === "online"
      ? "SYSTEM ONLINE"
      : backendStatus === "offline"
        ? "SYSTEM OFFLINE"
        : "CHECKING SYSTEM";

  const probabilityPercent = prediction
    ? Math.min(100, Math.max(0, prediction.probability * 100))
    : 0;

  const predictionMatches =
    prediction?.expectedClass !== null &&
    prediction?.expectedClass !== undefined &&
    prediction?.prediction === prediction.expectedClass;

  return (
    <main className="min-h-screen overflow-hidden bg-[#080a18] text-white selection:bg-violet-500/40">
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute -left-[20%] -top-[20%] h-[700px] w-[700px] rounded-full bg-violet-800/25 blur-[180px]" />

        <div className="absolute -right-[20%] top-[20%] h-[700px] w-[700px] rounded-full bg-blue-800/25 blur-[180px]" />

        <div className="absolute bottom-0 left-[35%] h-[500px] w-[500px] rounded-full bg-indigo-700/15 blur-[170px]" />
      </div>

      {/* NAVIGATION */}

      <header className="sticky top-0 z-50 border-b border-white/[0.10] bg-[#0a0b1b]/90 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-[1700px] items-center justify-between gap-5 px-5 py-4 md:px-10 lg:px-14">
          <button
            onClick={() => handleNavigation("Home")}
            className="group flex shrink-0 items-center gap-2"
          >
            <VyronLogo />

            <div className="text-left">
              <h1 className="text-[23px] font-semibold tracking-[0.25em]">
                VYRON<span className="text-violet-400">.</span>
              </h1>

              <p className="text-[8px] tracking-[0.32em] text-slate-400">
                NETWORK INTELLIGENCE
              </p>
            </div>
          </button>

          <nav className="hidden items-center gap-6 xl:flex">
            {navigation.map((item) => (
              <button
                key={item}
                onClick={() => handleNavigation(item)}
                className={`relative py-3 text-[10px] font-medium tracking-[0.08em] transition ${
                  activePage === item
                    ? "text-white"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {item.toUpperCase()}

                {activePage === item && (
                  <span className="absolute bottom-0 left-0 h-[2px] w-full bg-gradient-to-r from-violet-400 to-cyan-300" />
                )}
              </button>
            ))}
          </nav>

          <div className="hidden items-center gap-4 md:flex">
            <div
              className={`flex items-center gap-2 rounded-full border px-4 py-2.5 ${
                backendStatus === "online"
                  ? "border-emerald-400/30 bg-emerald-400/[0.09]"
                  : backendStatus === "offline"
                    ? "border-red-400/30 bg-red-400/[0.09]"
                    : "border-yellow-400/30 bg-yellow-400/[0.09]"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  backendStatus === "online"
                    ? "animate-pulse bg-emerald-400"
                    : backendStatus === "offline"
                      ? "bg-red-400"
                      : "animate-pulse bg-yellow-400"
                }`}
              />

              <span className="text-[9px] font-medium tracking-widest">
                {statusText}
              </span>
            </div>

            <button
              onClick={() => handleNavigation("Live Analyzer")}
              className="rounded-full bg-gradient-to-r from-violet-600 to-blue-500 px-6 py-3 text-[10px] font-semibold tracking-[0.12em] shadow-[0_0_30px_rgba(124,58,237,0.3)] transition hover:scale-105"
            >
              OPEN ANALYZER →
            </button>
          </div>
        </div>

        <div className="flex gap-6 overflow-x-auto border-t border-white/[0.07] px-5 py-3 xl:hidden">
          {navigation.map((item) => (
            <button
              key={item}
              onClick={() => handleNavigation(item)}
              className={`whitespace-nowrap text-[9px] tracking-wider ${
                activePage === item
                  ? "text-violet-300"
                  : "text-slate-400"
              }`}
            >
              {item.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      {/* HERO */}

      <section
        id="home"
        className="relative z-10 mx-auto grid max-w-[1700px] items-center gap-2 px-6 pb-12 pt-12 lg:grid-cols-[0.95fr_1.05fr] lg:px-16 lg:pb-20 lg:pt-14"
      >
        <div className="relative z-10">
          <SectionLabel>ADAPTIVE IOT SECURITY</SectionLabel>

          <h2 className="max-w-[780px] text-[52px] font-semibold leading-[1.04] tracking-[-0.055em] sm:text-6xl md:text-7xl xl:text-[88px]">
            See beyond
            <br />
            the{" "}
            <span className="bg-gradient-to-r from-violet-300 via-purple-300 to-cyan-300 bg-clip-text text-transparent">
              threat.
            </span>
            <br />
            <span className="text-[0.66em] tracking-[-0.04em]">
              Secure what matters.
            </span>
          </h2>

          <p className="mt-8 max-w-[520px] text-sm leading-7 text-slate-300 md:text-base">
            An intelligent machine learning platform designed to identify
            suspicious network behaviour and detect potential intrusions
            across IoT environments.
          </p>

          <div className="mt-9 flex flex-wrap gap-4">
            <button
              onClick={() => handleNavigation("Live Analyzer")}
              className="rounded-full bg-gradient-to-r from-violet-600 to-blue-500 px-7 py-4 text-[10px] font-semibold tracking-[0.16em] shadow-[0_0_35px_rgba(124,58,237,0.3)] transition hover:scale-105"
            >
              START ANALYSIS →
            </button>

            <button
              onClick={() => handleNavigation("Analytics")}
              className="rounded-full border border-white/25 bg-white/[0.04] px-7 py-4 text-[10px] font-semibold tracking-[0.16em] transition hover:border-violet-400 hover:bg-violet-500/10"
            >
              EXPLORE RESULTS
            </button>
          </div>

          <div className="mt-14 flex flex-wrap gap-8 border-t border-white/[0.13] pt-7 md:gap-12">
            <div>
              <p className="text-3xl font-semibold">38</p>
              <p className="mt-2 text-[9px] tracking-[0.2em] text-slate-400">
                ML FEATURES
              </p>
            </div>

            <div className="border-l border-violet-400/40 pl-8 md:pl-12">
              <p className="text-3xl font-semibold">03</p>
              <p className="mt-2 text-[9px] tracking-[0.2em] text-slate-400">
                MODELS EVALUATED
              </p>
            </div>

            <div className="border-l border-violet-400/40 pl-8 md:pl-12">
              <p className="text-3xl font-semibold">99.67%</p>
              <p className="mt-2 text-[9px] tracking-[0.2em] text-slate-400">
                TEST ACCURACY
              </p>
            </div>
          </div>
        </div>

        <NetworkGlobe />
      </section>

      {/* OVERVIEW */}

      <section
        id="overview"
        className="relative z-10 border-y border-white/[0.10] bg-[#17152b]/55 backdrop-blur-xl"
      >
        <div className="mx-auto grid max-w-[1600px] grid-cols-2 divide-x divide-white/[0.12] md:grid-cols-4">
          {[
            ["712,311", "DATASET RECORDS"],
            ["695,734", "MALICIOUS RECORDS"],
            ["99.67%", "TEST ACCURACY"],
            ["94.39%", "BALANCED ACCURACY"],
          ].map(([value, label]) => (
            <div key={label} className="px-6 py-8 md:px-12 md:py-10">
              <p className="text-2xl font-semibold tracking-tight md:text-4xl">
                {value}
              </p>

              <p className="mt-3 text-[9px] tracking-[0.2em] text-slate-400">
                {label}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* VIRTUAL IOT DEVICES */}

      <section
        id="devices"
        className="relative z-10 border-y border-white/[0.08] bg-[#0d1022]/80"
      >
        <div className="mx-auto max-w-[1600px] px-6 py-24 lg:px-16">
          <SectionLabel>EDGE DEVICE LABORATORY</SectionLabel>

          <div className="mb-12 flex flex-wrap items-end justify-between gap-6">
            <div>
              <h2 className="text-4xl font-semibold tracking-tight md:text-6xl">
                Virtual IoT <span className="text-slate-400">Devices.</span>
              </h2>
              <p className="mt-5 max-w-2xl text-sm leading-7 text-slate-300">
                Operate a simulated IoT environment, generate device activity,
                and associate each event with an XGBoost inference from the
                CICIoT2023 held-out test dataset.
              </p>
            </div>

            <button
              onClick={() => {
                setDeviceError("");
                refreshDeviceData().catch((err) =>
                  setDeviceError(err instanceof Error ? err.message : "Refresh failed.")
                );
              }}
              disabled={backendStatus !== "online"}
              className="rounded-xl border border-violet-300/25 bg-violet-400/[0.07] px-5 py-3 text-[9px] font-semibold tracking-[0.16em] text-violet-200 transition hover:border-violet-300/50 hover:bg-violet-400/[0.12] disabled:opacity-40"
            >
              ↻ REFRESH DEVICES
            </button>
          </div>

          <div className="mb-8 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[
              { label: "REGISTERED DEVICES", value: deviceStatistics.total_devices, tone: "text-white" },
              { label: "SIMULATING", value: deviceStatistics.simulating_devices, tone: "text-cyan-300" },
              { label: "ONLINE", value: deviceStatistics.online_devices, tone: "text-emerald-300" },
              { label: "OFFLINE", value: deviceStatistics.offline_devices, tone: "text-slate-400" },
            ].map((item) => (
              <div key={item.label} className="rounded-2xl border border-white/10 bg-white/[0.035] p-5">
                <p className="text-[9px] tracking-[0.16em] text-slate-500">{item.label}</p>
                <p className={`mt-3 text-3xl font-semibold ${item.tone}`}>{item.value}</p>
              </div>
            ))}
          </div>

          {deviceError && (
            <div className="mb-5 rounded-xl border border-red-400/25 bg-red-400/[0.07] p-4 text-xs text-red-200">
              {deviceError}
            </div>
          )}
          {deviceMessage && (
            <div className="mb-5 rounded-xl border border-emerald-400/25 bg-emerald-400/[0.07] p-4 text-xs text-emerald-200">
              {deviceMessage}
            </div>
          )}

          {backendStatus !== "online" ? (
            <div className="rounded-2xl border border-amber-400/20 bg-amber-400/[0.05] p-8 text-sm text-amber-200">
              Connect the VYRON backend to load and control virtual devices.
            </div>
          ) : virtualDevices.length === 0 ? (
            <div className="rounded-2xl border border-white/10 bg-[#111327]/80 p-8 text-sm text-slate-400">
              Loading registered virtual devices...
            </div>
          ) : (
            <div className="grid gap-5 xl:grid-cols-2">
              {virtualDevices.map((device) => {
                const latestDeviceEvent = deviceEvents.find(
                  (event) => event.device_id === device.device_id
                );
                const isBusy = deviceBusyId === device.device_id;
                const statusTone =
                  device.status === "simulating"
                    ? "border-cyan-400/25 bg-cyan-400/10 text-cyan-300"
                    : device.status === "online"
                      ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300"
                      : "border-slate-400/20 bg-slate-400/[0.07] text-slate-400";
                const deviceIcon =
                  device.device_type === "cctv_camera" ? "◉"
                    : device.device_type === "door_lock" ? "⌑"
                      : device.device_type === "temperature_sensor" ? "⌁"
                        : device.device_type === "smart_light" ? "✧" : "⌘";

                return (
                  <article
                    key={device.device_id}
                    className="group relative overflow-hidden rounded-2xl border border-white/[0.11] bg-gradient-to-br from-[#171a34] via-[#111428] to-[#0c1020] p-6 transition duration-300 hover:-translate-y-1 hover:border-violet-300/30 hover:shadow-[0_20px_70px_rgba(76,29,149,0.14)] md:p-7"
                  >
                    <div className="pointer-events-none absolute -right-16 -top-20 h-48 w-48 rounded-full bg-violet-500/[0.08] blur-3xl transition group-hover:bg-cyan-400/[0.10]" />
                    <div className="relative">
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div className="flex items-center gap-4">
                          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-violet-300/20 bg-violet-400/[0.08] text-2xl text-violet-200">
                            {deviceIcon}
                          </div>
                          <div>
                            <p className="text-[9px] tracking-[0.16em] text-slate-500">{device.device_id}</p>
                            <h3 className="mt-1 text-lg font-semibold">{device.name}</h3>
                            <p className="mt-1 text-[10px] capitalize text-slate-400">
                              {device.device_type.replaceAll("_", " ")}
                            </p>
                          </div>
                        </div>
                        <span className={`rounded-full border px-3 py-1.5 text-[8px] font-semibold tracking-[0.14em] ${statusTone}`}>
                          <span className="mr-2 inline-block h-1.5 w-1.5 rounded-full bg-current" />
                          {device.status.toUpperCase()}
                        </span>
                      </div>

                      <p className="mt-5 min-h-10 text-xs leading-5 text-slate-400">{device.description}</p>

                      <div className="mt-5 grid grid-cols-2 gap-3 rounded-xl border border-white/[0.07] bg-black/15 p-4">
                        <div>
                          <p className="text-[8px] tracking-[0.14em] text-slate-500">IP ADDRESS</p>
                          <p className="mt-2 font-mono text-xs text-cyan-200">{device.ip_address}</p>
                        </div>
                        <div>
                          <p className="text-[8px] tracking-[0.14em] text-slate-500">LOCATION</p>
                          <p className="mt-2 text-xs text-slate-300">{device.location}</p>
                        </div>
                      </div>

                      <div className="mt-5 flex flex-wrap gap-2">
                        <button
                          onClick={() => controlVirtualDevice(device.device_id, "start")}
                          disabled={isBusy || device.status !== "offline" || backendStatus !== "online"}
                          className="rounded-lg bg-gradient-to-r from-emerald-600 to-teal-500 px-4 py-3 text-[8px] font-semibold tracking-[0.13em] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          {isBusy ? "WORKING..." : "▶ START DEVICE"}
                        </button>
                        <button
                          onClick={() => controlVirtualDevice(device.device_id, "stop")}
                          disabled={isBusy || device.status === "offline"}
                          className="rounded-lg border border-red-400/25 bg-red-400/[0.06] px-4 py-3 text-[8px] font-semibold tracking-[0.13em] text-red-300 transition hover:bg-red-400/[0.12] disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          ■ STOP
                        </button>
                      </div>

                      <div className="mt-6 border-t border-white/[0.08] pt-5">
                        <p className="text-[9px] font-semibold tracking-[0.16em] text-violet-200">ACTIVITY GENERATOR</p>
                        <div className="mt-3 flex flex-col gap-3 sm:flex-row">
                          <select
                            aria-label={`Activity scenario for ${device.name}`}
                            value={deviceScenario[device.device_id] || "normal"}
                            onChange={(event) =>
                              setDeviceScenario((previous) => ({
                                ...previous,
                                [device.device_id]: event.target.value,
                              }))
                            }
                            className="min-w-0 flex-1 rounded-lg border border-white/10 bg-[#090d1d] px-3 py-3 text-xs text-white outline-none focus:border-violet-400"
                          >
                            <option value="normal">Normal activity</option>
                            <option value="suspicious">Suspicious activity</option>
                          </select>
                          <button
                            onClick={() => generateDeviceActivity(device.device_id)}
                            disabled={isBusy || device.status === "offline" || backendStatus !== "online"}
                            className="rounded-lg border border-cyan-300/25 bg-cyan-300/[0.08] px-4 py-3 text-[8px] font-semibold tracking-[0.12em] text-cyan-200 transition hover:bg-cyan-300/[0.14] disabled:cursor-not-allowed disabled:opacity-35"
                          >
                            GENERATE ACTIVITY
                          </button>
                        </div>
                        <button
                          onClick={() => analyzeVirtualDevice(device.device_id)}
                          disabled={isBusy || device.status === "offline" || !latestDeviceEvent || backendStatus !== "online"}
                          className="mt-3 w-full rounded-lg bg-gradient-to-r from-violet-600 to-blue-500 px-4 py-3 text-[9px] font-semibold tracking-[0.15em] shadow-[0_0_25px_rgba(124,58,237,0.13)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          {isBusy ? "PROCESSING..." : "RUN XGBOOST ANALYSIS →"}
                        </button>
                      </div>

                      {latestDeviceEvent && (
                        <div className="mt-5 rounded-xl border border-white/[0.08] bg-black/20 p-4">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <span className="text-[8px] tracking-[0.15em] text-slate-500">LATEST EVENT · #{latestDeviceEvent.event_id}</span>
                            <span className={`text-[8px] font-semibold tracking-wider ${latestDeviceEvent.analysis?.prediction === "MALICIOUS" ? "text-red-300" : latestDeviceEvent.analysis?.prediction === "BENIGN" ? "text-emerald-300" : "text-slate-400"}`}>
                              {latestDeviceEvent.analysis?.prediction || "AWAITING ANALYSIS"}
                            </span>
                          </div>
                          <p className="mt-3 text-xs text-slate-200">{latestDeviceEvent.activity}</p>
                          {latestDeviceEvent.analysis && (
                            <div className="mt-4 grid grid-cols-2 gap-3">
                              <div>
                                <p className="text-[8px] text-slate-500">MALICIOUS PROBABILITY</p>
                                <p className="mt-1 font-mono text-sm text-red-200">
                                  {(latestDeviceEvent.analysis.malicious_probability * 100).toFixed(3)}%
                                </p>
                              </div>
                              <div>
                                <p className="text-[8px] text-slate-500">DATASET LABEL</p>
                                <p className="mt-1 break-words text-xs text-slate-200">{latestDeviceEvent.analysis.original_label}</p>
                              </div>
                              <div className="col-span-2">
                                <p className={`text-[9px] font-semibold ${latestDeviceEvent.analysis.matches_dataset ? "text-emerald-300" : "text-amber-300"}`}>
                                  {latestDeviceEvent.analysis.matches_dataset ? "✓ PREDICTION MATCHES DATASET LABEL" : "⚠ PREDICTION DIFFERS FROM DATASET LABEL"}
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}

          <div className="mt-8 rounded-2xl border border-amber-300/20 bg-amber-300/[0.045] p-5 md:p-6">
            <p className="text-[9px] font-semibold tracking-[0.17em] text-amber-200">DEMONSTRATION SCOPE & ML METHODOLOGY</p>
            <p className="mt-3 text-xs leading-6 text-slate-300">
              Device identities, IP addresses, and generated activities are simulated. When you run analysis,
              VYRON selects a separate genuine record from the CICIoT2023 held-out test dataset and sends its
              38 features to XGBoost. The prediction is attached to the device event for demonstration, but it
              is not a direct classification of the simulated activity or a real device/network packet.
            </p>
          </div>

          <div className="mt-10 overflow-hidden rounded-2xl border border-white/10 bg-[#111327]/90">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 p-6">
              <div>
                <p className="text-[9px] tracking-[0.18em] text-cyan-300">DEVICE ACTIVITY LOG</p>
                <h3 className="mt-2 text-xl font-semibold">Recent Virtual Device Events</h3>
              </div>
              <span className="text-[9px] tracking-widest text-slate-500">{deviceEvents.length} EVENTS</span>
            </div>
            {deviceEvents.length === 0 ? (
              <div className="p-10 text-center text-xs text-slate-500">
                No virtual device activity has been generated yet.
              </div>
            ) : (
              <div className="max-h-[440px] overflow-auto">
                <table className="w-full min-w-[760px] text-left">
                  <thead className="sticky top-0 bg-[#17182e]">
                    <tr className="border-b border-white/10 text-[8px] tracking-widest text-slate-500">
                      <th className="px-5 py-4 font-normal">EVENT</th>
                      <th className="px-5 py-4 font-normal">DEVICE</th>
                      <th className="px-5 py-4 font-normal">ACTIVITY</th>
                      <th className="px-5 py-4 font-normal">SCENARIO</th>
                      <th className="px-5 py-4 font-normal">ML RESULT</th>
                      <th className="px-5 py-4 font-normal">TIME</th>
                    </tr>
                  </thead>
                  <tbody>
                    {deviceEvents.map((event) => (
                      <tr key={event.event_id} className="border-b border-white/[0.05] hover:bg-white/[0.025]">
                        <td className="px-5 py-4 font-mono text-xs text-slate-400">#{event.event_id}</td>
                        <td className="px-5 py-4">
                          <p className="text-xs text-slate-200">{event.device_name}</p>
                          <p className="mt-1 font-mono text-[9px] text-slate-500">{event.device_id}</p>
                        </td>
                        <td className="max-w-[260px] px-5 py-4 text-xs text-slate-300">{event.activity}</td>
                        <td className="px-5 py-4 text-[9px] uppercase tracking-wider text-slate-400">{event.scenario}</td>
                        <td className="px-5 py-4 text-xs">
                          {event.analysis ? (
                            <span className={event.analysis.prediction === "MALICIOUS" ? "text-red-300" : "text-emerald-300"}>
                              {event.analysis.prediction} · {(event.analysis.malicious_probability * 100).toFixed(2)}%
                            </span>
                          ) : <span className="text-slate-500">Not analyzed</span>}
                        </td>
                        <td className="px-5 py-4 text-[10px] text-slate-400">{new Date(event.timestamp).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="border-t border-white/10 px-6 py-4 text-[9px] text-slate-500">
              Event records are generated by the VYRON virtual-device simulator; they are not captured network packets.
            </div>
          </div>
        </div>
      </section>

      {/* LIVE SIMULATION DASHBOARD */}

      <section
        id="simulation"
        className="relative z-10 mx-auto max-w-[1600px] px-6 py-24 lg:px-16"
      >
        <SectionLabel>CONTINUOUS NETWORK MONITORING</SectionLabel>

        <div className="mb-10 flex flex-wrap items-end justify-between gap-6">
          <div>
            <h2 className="text-4xl font-semibold tracking-tight md:text-6xl">
              Network{" "}
              <span className="text-slate-400">Surveillance.</span>
            </h2>

            <p className="mt-5 max-w-xl text-sm leading-7 text-slate-300">
              Continuously replay real CICIoT2023 traffic records through
              the XGBoost intrusion detection engine.
            </p>
          </div>

          <div
            className={`flex items-center gap-2 rounded-full border px-4 py-3 text-[10px] tracking-widest ${
              simulationRunning
                ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                : "border-slate-400/20 bg-white/[0.04] text-slate-400"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                simulationRunning
                  ? "animate-pulse bg-emerald-400"
                  : "bg-slate-500"
              }`}
            />

            {simulationRunning ? "SIMULATION RUNNING" : "SIMULATION STOPPED"}
          </div>
        </div>

        {/* CONTROL PANEL */}

        <div className="rounded-2xl border border-white/10 bg-[#111327]/90 p-6 md:p-8">
          <div className="flex flex-wrap items-end justify-between gap-6">
            <div className="min-w-[260px] flex-1">
              <p className="text-[10px] tracking-[0.2em] text-violet-300">
                SIMULATION CONTROL CENTER
              </p>

              <h3 className="mt-3 text-2xl font-semibold">
                Traffic Replay Engine
              </h3>

              <p className="mt-2 text-xs text-slate-400">
                CICIoT2023 · XGBoost · 38 Features · 1-second interval
              </p>

              <div className="mt-6 max-w-xl">
                <label
                  htmlFor="simulation-mode"
                  className="mb-2 block text-[9px] font-semibold tracking-[0.18em] text-slate-300"
                >
                  SELECT SIMULATION MODE
                </label>

                <select
                  id="simulation-mode"
                  value={simulationMode}
                  onChange={(event) =>
                    setSimulationMode(event.target.value as SimulationMode)
                  }
                  disabled={simulationRunning || simulationLoading}
                  className="w-full rounded-xl border border-violet-400/25 bg-[#090d1d] px-4 py-3 text-sm text-white outline-none transition focus:border-violet-400 disabled:cursor-not-allowed disabled:opacity-50 sm:max-w-sm"
                >
                  <option value="dataset_replay">Dataset Replay</option>
                  <option value="balanced_demo">Balanced Demo</option>
                </select>

                <p className="mt-3 max-w-lg text-[11px] leading-5 text-slate-400">
                  {simulationMode === "dataset_replay"
                    ? "Replays records using the original class distribution of the held-out CICIoT2023 test dataset. The stream may contain mostly malicious records."
                    : "Alternates between genuine benign and malicious records from the held-out test dataset for a clearer demonstration. Model predictions are not forced, so mismatches can occur."}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                onClick={startSimulation}
                disabled={
                  simulationLoading ||
                  simulationRunning ||
                  backendStatus !== "online"
                }
                className="rounded-xl bg-gradient-to-r from-emerald-600 to-teal-500 px-7 py-4 text-[10px] font-semibold tracking-widest transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {simulationLoading
                  ? "PLEASE WAIT..."
                  : "▶ START SIMULATION"}
              </button>

              <button
                onClick={stopSimulation}
                disabled={simulationLoading || !simulationRunning}
                className="rounded-xl border border-red-400/30 bg-red-400/[0.08] px-7 py-4 text-[10px] font-semibold tracking-widest text-red-300 transition hover:bg-red-400/20 disabled:cursor-not-allowed disabled:opacity-40"
              >
                ■ STOP SIMULATION
              </button>
            </div>
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3 rounded-xl border border-violet-400/15 bg-violet-400/[0.04] px-4 py-3">
            <span className="text-[9px] tracking-[0.18em] text-slate-400">
              {simulationRunning ? "ACTIVE MODE" : "SELECTED MODE"}
            </span>
            <span className="text-xs font-semibold text-violet-200">
              {simulationRunning
                ? simulationStatus?.mode_label ??
                  (simulationMode === "balanced_demo"
                    ? "Balanced Demo"
                    : "Dataset Replay")
                : simulationMode === "balanced_demo"
                  ? "Balanced Demo"
                  : "Dataset Replay"}
            </span>
          </div>

          {simulationError && (
            <div className="mt-6 rounded-lg border border-red-400/30 bg-red-400/[0.08] p-4 text-xs text-red-300">
              {simulationError}
            </div>
          )}

          {/* STATISTICS */}

          <div className="mt-9 grid grid-cols-2 gap-4 lg:grid-cols-4">
            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
              <p className="text-[9px] tracking-widest text-slate-400">
                TOTAL PROCESSED
              </p>

              <p className="mt-4 text-3xl font-semibold">
                {simulationStatus?.total_events ?? 0}
              </p>

              <p className="mt-2 text-[9px] text-slate-500">
                Lifetime simulation events
              </p>
            </div>

            <div className="rounded-xl border border-emerald-400/20 bg-emerald-400/[0.05] p-5">
              <p className="text-[9px] tracking-widest text-emerald-300">
                BENIGN DETECTED
              </p>

              <p className="mt-4 text-3xl font-semibold text-emerald-300">
                {benignCount}
              </p>

              <p className="mt-2 text-[9px] text-slate-500">
                Currently displayed events
              </p>
            </div>

            <div className="rounded-xl border border-red-400/20 bg-red-400/[0.05] p-5">
              <p className="text-[9px] tracking-widest text-red-300">
                MALICIOUS DETECTED
              </p>

              <p className="mt-4 text-3xl font-semibold text-red-300">
                {maliciousCount}
              </p>

              <p className="mt-2 text-[9px] text-slate-500">
                Currently displayed events
              </p>
            </div>

            <div className="rounded-xl border border-violet-400/20 bg-violet-400/[0.05] p-5">
              <p className="text-[9px] tracking-widest text-violet-300">
                LABEL VERIFICATION
              </p>

              <p className="mt-4 text-3xl font-semibold text-violet-300">
                {verificationRate.toFixed(1)}%
              </p>

              <p className="mt-2 text-[9px] text-slate-500">
                Displayed event agreement
              </p>
            </div>
          </div>

          {/* LATEST DETECTION */}

          <div className="mt-6 rounded-xl border border-white/10 bg-[#090d1d] p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-[9px] tracking-widest text-slate-400">
                  LATEST DETECTION EVENT
                </p>

                <p className="mt-3 text-lg font-semibold">
                  {latestEvent
                    ? latestEvent.original_label
                    : "Awaiting incoming traffic..."}
                </p>
              </div>

              {latestEvent && (
                <div
                  className={`rounded-lg border px-4 py-3 text-[10px] font-semibold tracking-widest ${
                    latestEvent.predicted_class === "MALICIOUS"
                      ? "border-red-400/30 bg-red-400/10 text-red-300"
                      : "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                  }`}
                >
                  {latestEvent.predicted_class}
                </div>
              )}
            </div>

            {latestEvent && (
              <div className="mt-5 grid grid-cols-2 gap-4 md:grid-cols-4">
                <div>
                  <p className="text-[9px] text-slate-500">
                    EVENT ID
                  </p>
                  <p className="mt-2 font-mono text-xs">
                    #{latestEvent.id}
                  </p>
                </div>

                <div>
                  <p className="text-[9px] text-slate-500">
                    MALICIOUS PROBABILITY
                  </p>
                  <p className="mt-2 font-mono text-xs">
                    {(latestEvent.malicious_probability * 100).toFixed(3)}%
                  </p>
                </div>

                <div>
                  <p className="text-[9px] text-slate-500">
                    DATASET VERIFICATION
                  </p>
                  <p className="mt-2 text-xs text-emerald-300">
                    {latestEvent.matches_dataset ? "✓ MATCH" : "✗ MISMATCH"}
                  </p>
                </div>

                <div>
                  <p className="text-[9px] text-slate-500">
                    EVENT TIME
                  </p>
                  <p className="mt-2 text-xs">
                    {new Date(latestEvent.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* LIVE EVENT FEED */}

        <div className="mt-8 overflow-hidden rounded-2xl border border-white/10 bg-[#111327]/90">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 p-6">
            <div>
              <p className="text-[10px] tracking-[0.2em] text-cyan-300">
                LIVE DETECTION STREAM
              </p>

              <h3 className="mt-2 text-xl font-semibold">
                Network Event Feed
              </h3>
            </div>

            <div className="flex items-center gap-2 text-[9px] tracking-widest text-slate-400">
              <span
                className={`h-2 w-2 rounded-full ${
                  simulationRunning
                    ? "animate-pulse bg-emerald-400"
                    : "bg-slate-500"
                }`}
              />

              {simulationRunning ? "RECEIVING EVENTS" : "STREAM PAUSED"}
            </div>
          </div>

          {simulationEvents.length === 0 ? (
            <div className="flex min-h-[220px] flex-col items-center justify-center text-center">
              <div className="text-4xl text-violet-300">◈</div>

              <p className="mt-5 text-sm font-semibold">
                No traffic events yet
              </p>

              <p className="mt-2 text-xs text-slate-500">
                Start the simulation to populate the detection stream.
              </p>
            </div>
          ) : (
            <div className="max-h-[540px] overflow-auto">
              <table className="w-full min-w-[850px] text-left">
                <thead className="sticky top-0 bg-[#17182e]">
                  <tr className="border-b border-white/10 text-[9px] tracking-widest text-slate-500">
                    <th className="px-5 py-4 font-normal">EVENT</th>
                    <th className="px-5 py-4 font-normal">TIME</th>
                    <th className="px-5 py-4 font-normal">ATTACK CATEGORY</th>
                    <th className="px-5 py-4 font-normal">CLASSIFICATION</th>
                    <th className="px-5 py-4 font-normal">PROBABILITY</th>
                    <th className="px-5 py-4 font-normal">VERIFICATION</th>
                  </tr>
                </thead>

                <tbody>
                  {simulationEvents.map((event) => (
                    <tr
                      key={event.id}
                      className="border-b border-white/[0.05] transition hover:bg-white/[0.025]"
                    >
                      <td className="px-5 py-4 font-mono text-xs text-slate-400">
                        #{event.id}
                      </td>

                      <td className="px-5 py-4 text-xs text-slate-400">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </td>

                      <td className="max-w-[230px] truncate px-5 py-4 text-xs text-slate-300">
                        {event.original_label}
                      </td>

                      <td className="px-5 py-4">
                        <span
                          className={`rounded-md px-3 py-1.5 text-[9px] font-semibold ${
                            event.predicted_class === "MALICIOUS"
                              ? "bg-red-400/10 text-red-300"
                              : "bg-emerald-400/10 text-emerald-300"
                          }`}
                        >
                          {event.predicted_class}
                        </span>
                      </td>

                      <td className="px-5 py-4 font-mono text-xs">
                        {(
                          event.malicious_probability * 100
                        ).toFixed(3)}
                        %
                      </td>

                      <td
                        className={`px-5 py-4 text-xs ${
                          event.matches_dataset
                            ? "text-emerald-300"
                            : "text-amber-300"
                        }`}
                      >
                        {event.matches_dataset ? "✓ MATCH" : "✗ MISMATCH"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="border-t border-white/10 px-6 py-4 text-[9px] text-slate-500">
            Displaying up to 100 most recently received events. Event data
            is replayed from the CICIoT2023 held-out test dataset.
          </div>
        </div>
      </section>

      {/* MANUAL LIVE ANALYZER */}

      <section
        id="analyzer"
        className="relative z-10 mx-auto max-w-[1600px] px-6 py-24 lg:px-16"
      >
        <SectionLabel>MANUAL INTRUSION DETECTION</SectionLabel>

        <div className="mb-12 flex flex-wrap items-end justify-between gap-6">
          <div>
            <h2 className="text-4xl font-semibold tracking-tight md:text-6xl">
              Traffic{" "}
              <span className="text-slate-400">Analyzer.</span>
            </h2>

            <p className="mt-5 max-w-xl text-sm leading-7 text-slate-300">
              Load actual CICIoT2023 test records and submit their network
              features to the trained XGBoost detection engine.
            </p>
          </div>

          <div
            className={`rounded-full border px-4 py-2 text-[9px] tracking-[0.2em] ${
              backendStatus === "online"
                ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                : "border-red-400/30 bg-red-400/10 text-red-300"
            }`}
          >
            ● {backendStatus === "online" ? "ENGINE CONNECTED" : "ENGINE DISCONNECTED"}
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          {/* INPUT PANEL */}

          <div className="rounded-2xl border border-white/[0.12] bg-[#111327]/90 p-6 md:p-8">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-[10px] tracking-[0.2em] text-violet-300">
                  NETWORK FEATURE INPUT
                </p>

                <p className="mt-2 text-xs text-slate-400">
                  {Object.keys(trafficData).length} / 38 model features
                </p>
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => loadSample("benign")}
                  disabled={loadingSample || backendStatus !== "online"}
                  className="rounded-lg border border-emerald-400/30 px-3 py-2 text-[9px] text-emerald-300 transition hover:bg-emerald-400/10 disabled:opacity-40"
                >
                  {loadingSample ? "LOADING..." : "LOAD BENIGN"}
                </button>

                <button
                  onClick={() => loadSample("malicious")}
                  disabled={loadingSample || backendStatus !== "online"}
                  className="rounded-lg border border-red-400/30 px-3 py-2 text-[9px] text-red-300 transition hover:bg-red-400/10 disabled:opacity-40"
                >
                  {loadingSample ? "LOADING..." : "LOAD MALICIOUS"}
                </button>
              </div>
            </div>

            {loadedSample && (
              <div className="mt-5 rounded-lg border border-violet-400/20 bg-violet-400/[0.06] p-4">
                <p className="text-[10px] tracking-widest text-violet-300">
                  DATASET SAMPLE LOADED
                </p>

                <div className="mt-2 flex flex-wrap justify-between gap-2 text-xs">
                  <span>
                    Expected class:{" "}
                    <strong
                      className={
                        expectedClass === "BENIGN"
                          ? "text-emerald-300"
                          : "text-red-300"
                      }
                    >
                      {expectedClass}
                    </strong>
                  </span>

                  <span className="text-slate-400">
                    {originalLabel}
                  </span>
                </div>

                <p className="mt-2 text-[9px] text-slate-500">
                  Source: {sampleSource}
                </p>
              </div>
            )}

            <div className="mt-7 grid max-h-[440px] grid-cols-1 gap-3 overflow-y-auto pr-2 sm:grid-cols-2">
              {Object.entries(trafficData).map(([key, value]) => (
                <label key={key} className="block">
                  <span className="mb-2 block text-[10px] text-slate-400">
                    {key}
                  </span>

                  <input
                    type="number"
                    step="any"
                    value={value}
                    onChange={(event) =>
                      handleFeatureChange(key, event.target.value)
                    }
                    className="w-full rounded-lg border border-white/10 bg-[#090d1d] px-3 py-2.5 font-mono text-xs text-white outline-none transition focus:border-violet-400"
                  />
                </label>
              ))}
            </div>

            <div className="mt-7 border-t border-white/10 pt-6">
              <button
                onClick={runAnalysis}
                disabled={
                  analyzing ||
                  loadingSample ||
                  backendStatus !== "online" ||
                  Object.keys(trafficData).length !== 38
                }
                className="w-full rounded-xl bg-gradient-to-r from-violet-600 to-blue-500 px-6 py-4 text-[11px] font-semibold tracking-[0.18em] shadow-[0_0_30px_rgba(124,58,237,0.2)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {analyzing
                  ? "ANALYZING NETWORK TRAFFIC..."
                  : "RUN XGBOOST ANALYSIS →"}
              </button>

              <p className="mt-4 text-center text-[9px] leading-5 text-slate-500">
                Samples are actual records from the held-out CICIoT2023
                test dataset. This demonstration does not capture live
                packets from a network interface.
              </p>
            </div>
          </div>

          {/* RESULT PANEL */}

          <div className="rounded-2xl border border-white/[0.12] bg-[#111327]/90 p-6 md:p-8">
            <p className="text-[10px] tracking-[0.2em] text-cyan-300">
              DETECTION RESULT
            </p>

            {!prediction && !error && (
              <div className="flex min-h-[380px] flex-col items-center justify-center text-center">
                <div className="flex h-24 w-24 items-center justify-center rounded-full border border-violet-400/30 bg-violet-400/10 text-4xl text-violet-300">
                  ◈
                </div>

                <h3 className="mt-7 text-xl font-semibold">
                  Awaiting Analysis
                </h3>

                <p className="mt-3 max-w-xs text-xs leading-6 text-slate-400">
                  Load a real dataset sample and submit its features to
                  receive an XGBoost prediction.
                </p>
              </div>
            )}

            {error && (
              <div className="mt-8 rounded-xl border border-red-400/30 bg-red-400/[0.08] p-6">
                <p className="text-sm font-semibold text-red-300">
                  ANALYSIS FAILED
                </p>

                <p className="mt-3 break-words text-xs leading-6 text-red-200/80">
                  {error}
                </p>
              </div>
            )}

            {prediction && (
              <div className="mt-8">
                <div
                  className={`rounded-xl border p-6 ${
                    prediction.prediction === "MALICIOUS"
                      ? "border-red-400/30 bg-red-400/[0.08]"
                      : "border-emerald-400/30 bg-emerald-400/[0.08]"
                  }`}
                >
                  <p className="text-[9px] tracking-[0.2em] text-slate-400">
                    CLASSIFICATION
                  </p>

                  <h3
                    className={`mt-4 text-3xl font-semibold ${
                      prediction.prediction === "MALICIOUS"
                        ? "text-red-300"
                        : "text-emerald-300"
                    }`}
                  >
                    {prediction.prediction}
                  </h3>

                  <p className="mt-3 text-xs text-slate-300">
                    {prediction.prediction === "MALICIOUS"
                      ? "Potential malicious network activity detected."
                      : "Traffic classified as benign by the model."}
                  </p>
                </div>

                <div
                  className={`mt-5 rounded-xl border p-5 ${
                    prediction.expectedClass === null
                      ? "border-white/10 bg-white/[0.03]"
                      : predictionMatches
                        ? "border-emerald-400/30 bg-emerald-400/[0.07]"
                        : "border-amber-400/30 bg-amber-400/[0.07]"
                  }`}
                >
                  <p className="text-[9px] tracking-[0.2em] text-slate-400">
                    DATASET VERIFICATION
                  </p>

                  {prediction.expectedClass === null ? (
                    <p className="mt-3 text-xs text-slate-300">
                      No original dataset label available.
                    </p>
                  ) : (
                    <>
                      <p
                        className={`mt-3 text-sm font-semibold ${
                          predictionMatches
                            ? "text-emerald-300"
                            : "text-amber-300"
                        }`}
                      >
                        {predictionMatches
                          ? "✓ PREDICTION MATCHES DATASET LABEL"
                          : "⚠ PREDICTION DOES NOT MATCH DATASET LABEL"}
                      </p>

                      <div className="mt-4 space-y-2 text-xs">
                        <div className="flex justify-between gap-3">
                          <span className="text-slate-400">
                            Expected class
                          </span>
                          <span>{prediction.expectedClass}</span>
                        </div>

                        <div className="flex justify-between gap-3">
                          <span className="text-slate-400">
                            Model prediction
                          </span>
                          <span>{prediction.prediction}</span>
                        </div>

                        <div className="flex justify-between gap-3">
                          <span className="text-slate-400">
                            Original dataset label
                          </span>
                          <span>{prediction.originalLabel}</span>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                <div className="mt-8">
                  <div className="flex justify-between gap-3">
                    <span className="text-xs text-slate-400">
                      Malicious probability
                    </span>

                    <span className="font-mono text-sm font-semibold">
                      {prediction.probability.toFixed(6)}
                    </span>
                  </div>

                  <div className="mt-4 h-3 overflow-hidden rounded-full bg-white/10">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        prediction.prediction === "MALICIOUS"
                          ? "bg-gradient-to-r from-red-500 to-orange-400"
                          : "bg-gradient-to-r from-emerald-500 to-cyan-400"
                      }`}
                      style={{
                        width: `${probabilityPercent}%`,
                      }}
                    />
                  </div>

                  <p className="mt-2 text-right text-[10px] text-slate-400">
                    {probabilityPercent.toFixed(2)}%
                  </p>
                </div>

                <div className="mt-8 space-y-4">
                  <div className="flex justify-between border-b border-white/10 pb-4">
                    <span className="text-xs text-slate-400">
                      Detection threshold
                    </span>

                    <span className="font-mono text-xs">
                      {prediction.threshold}
                    </span>
                  </div>

                  <div className="flex justify-between border-b border-white/10 pb-4">
                    <span className="text-xs text-slate-400">
                      Prediction time
                    </span>

                    <span className="text-xs">{prediction.time}</span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-xs text-slate-400">
                      Detection model
                    </span>

                    <span className="text-xs text-violet-300">
                      XGBoost
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* MANUAL PREDICTION HISTORY */}

        <div className="mt-8 rounded-2xl border border-white/[0.12] bg-[#111327]/90 p-6 md:p-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-[10px] tracking-[0.2em] text-violet-300">
                ANALYSIS LOG
              </p>

              <h3 className="mt-3 text-xl font-semibold">
                Recent Predictions
              </h3>
            </div>

            <button
              onClick={() => setHistory([])}
              className="text-[9px] tracking-widest text-slate-400 transition hover:text-white"
            >
              CLEAR HISTORY
            </button>
          </div>

          {history.length === 0 ? (
            <p className="mt-8 text-center text-xs text-slate-500">
              No manual predictions recorded in this session.
            </p>
          ) : (
            <div className="mt-7 overflow-x-auto">
              <table className="w-full min-w-[650px] text-left">
                <thead>
                  <tr className="border-b border-white/10 text-[9px] tracking-widest text-slate-500">
                    <th className="py-4 font-normal">TIME</th>
                    <th className="py-4 font-normal">EXPECTED</th>
                    <th className="py-4 font-normal">PREDICTION</th>
                    <th className="py-4 font-normal">PROBABILITY</th>
                    <th className="py-4 font-normal">VERIFICATION</th>
                  </tr>
                </thead>

                <tbody>
                  {history.map((item) => (
                    <tr
                      key={item.id}
                      className="border-b border-white/[0.06]"
                    >
                      <td className="py-4 text-xs text-slate-400">
                        {item.time}
                      </td>

                      <td className="py-4 text-xs text-slate-300">
                        {item.expectedClass ?? "—"}
                      </td>

                      <td
                        className={`py-4 text-xs font-semibold ${
                          item.prediction === "MALICIOUS"
                            ? "text-red-300"
                            : "text-emerald-300"
                        }`}
                      >
                        {item.prediction}
                      </td>

                      <td className="py-4 font-mono text-xs">
                        {item.probability.toFixed(6)}
                      </td>

                      <td className="py-4 text-xs">
                        {item.expectedClass === null
                          ? "Not available"
                          : item.prediction === item.expectedClass
                            ? "✓ MATCH"
                            : "✗ MISMATCH"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {/* ANALYTICS */}

      <section
        id="analytics"
        className="relative z-10 mx-auto max-w-[1600px] px-6 py-24 lg:px-16"
      >
        <SectionLabel>INTELLIGENCE OVERVIEW</SectionLabel>

        <div className="mb-14 flex flex-wrap items-end justify-between gap-6">
          <div>
            <h2 className="text-4xl font-semibold tracking-tight md:text-6xl">
              Intelligence,{" "}
              <span className="text-slate-400">engineered.</span>
            </h2>

            <p className="mt-5 max-w-xl text-sm leading-7 text-slate-300">
              Machine learning models trained and evaluated to distinguish
              malicious network behaviour from benign traffic.
            </p>
          </div>

          <span className="rounded-full border border-violet-300/20 bg-violet-400/[0.06] px-4 py-2 text-[9px] tracking-[0.2em] text-violet-200">
            EXPERIMENTAL RESULTS
          </span>
        </div>

        <div className="grid gap-5 lg:grid-cols-3">
          <div className="relative overflow-hidden rounded-2xl border border-violet-400/25 bg-gradient-to-br from-[#29204b] via-[#19172f] to-[#101323] p-8 lg:col-span-2">
            <p className="relative text-[10px] tracking-[0.25em] text-violet-200">
              UNSEEN-ATTACK EVALUATION
            </p>

            <div className="relative mt-12 grid gap-10 sm:grid-cols-2">
              <div>
                <p className="text-6xl font-semibold tracking-tight">
                  99.67<span className="text-3xl text-violet-300">%</span>
                </p>

                <p className="mt-3 text-xs text-slate-300">
                  Overall test accuracy
                </p>
              </div>

              <div>
                <p className="text-6xl font-semibold tracking-tight">
                  94.39<span className="text-3xl text-cyan-300">%</span>
                </p>

                <p className="mt-3 text-xs text-slate-300">
                  Balanced accuracy
                </p>
              </div>
            </div>

            <div className="relative mt-12 h-px bg-white/15">
              <div className="absolute h-px w-[85%] bg-gradient-to-r from-violet-400 to-cyan-300" />
            </div>

            <div className="relative mt-4 flex flex-wrap justify-between gap-3 text-[9px] tracking-widest text-slate-300">
              <span>XGBOOST CLASSIFIER</span>
              <span>THRESHOLD · 0.07</span>
            </div>
          </div>

          <div
            id="system"
            className="rounded-2xl border border-white/[0.12] bg-[#15172b]/90 p-8"
          >
            <p className="text-[10px] tracking-[0.25em] text-cyan-300">
              DETECTION ENGINE
            </p>

            <h3 className="mt-8 text-4xl font-semibold">XGBoost</h3>

            <p className="mt-3 text-xs leading-6 text-slate-300">
              Gradient-boosted decision trees selected following comparative
              model evaluation.
            </p>

            <div className="mt-9 space-y-5">
              <div className="flex justify-between border-b border-white/[0.10] pb-4">
                <span className="text-xs text-slate-300">
                  Input features
                </span>

                <span className="text-xs">
                  {modelFeatures ?? "38"}
                </span>
              </div>

              <div className="flex justify-between border-b border-white/[0.10] pb-4">
                <span className="text-xs text-slate-300">
                  Detection threshold
                </span>

                <span className="text-xs">
                  {modelThreshold !== null ? modelThreshold : "0.07"}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-xs text-slate-300">
                  Backend status
                </span>

                <span
                  className={`text-xs ${
                    backendStatus === "online"
                      ? "text-emerald-300"
                      : backendStatus === "offline"
                        ? "text-red-300"
                        : "text-yellow-300"
                  }`}
                >
                  {backendStatus === "online"
                    ? "Connected"
                    : backendStatus === "offline"
                      ? "Disconnected"
                      : "Checking"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* THREAT INTELLIGENCE */}

      <section
        id="threats"
        className="relative z-10 border-t border-white/[0.10] bg-[#101122]/75"
      >
        <div className="mx-auto max-w-[1600px] px-6 py-24 lg:px-16">
          <SectionLabel>THREAT INTELLIGENCE</SectionLabel>

          <div className="mb-12">
            <h2 className="text-4xl font-semibold tracking-tight md:text-6xl">
              Signals worth{" "}
              <span className="text-slate-400">watching.</span>
            </h2>

            <p className="mt-5 text-sm text-slate-300">
              Real-time attack categories identified by the detection engine.
            </p>
          </div>

          {simulationEvents.length === 0 ? (
            <div className="rounded-2xl border border-white/10 bg-[#111327]/80 p-8">
              <p className="text-sm text-slate-400">
                No threat intelligence events available yet. Start the
                simulation to populate this section.
              </p>
            </div>
          ) : (
            <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
              {simulationEvents
                .filter((event) => event.predicted_class === "MALICIOUS")
                .slice(0, 9)
                .map((event) => (
                  <div
                    key={event.id}
                    className="rounded-xl border border-red-400/20 bg-red-400/[0.05] p-6"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-[9px] tracking-widest text-red-300">
                        MALICIOUS EVENT
                      </span>

                      <span className="text-[9px] text-slate-500">
                        #{event.id}
                      </span>
                    </div>

                    <h3 className="mt-5 break-words text-lg font-semibold">
                      {event.original_label}
                    </h3>

                    <div className="mt-5 flex justify-between text-xs">
                      <span className="text-slate-400">
                        Malicious probability
                      </span>

                      <span className="font-mono text-red-300">
                        {(event.malicious_probability * 100).toFixed(2)}%
                      </span>
                    </div>

                    <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-red-500 to-orange-400"
                        style={{
                          width: `${event.malicious_probability * 100}%`,
                        }}
                      />
                    </div>

                    <p className="mt-4 text-[9px] text-slate-500">
                      {new Date(event.timestamp).toLocaleString()}
                    </p>
                  </div>
                ))}
            </div>
          )}
        </div>
      </section>

      {/* FOOTER */}

      <footer className="relative z-10 border-t border-white/[0.12] bg-[#090b19]">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-6 px-6 py-9 lg:px-16">
          <div className="flex items-center gap-3">
            <VyronLogo />

            <div>
              <p className="text-sm font-semibold tracking-[0.2em]">
                VYRON
              </p>

              <p className="text-[8px] tracking-[0.2em] text-slate-400">
                NETWORK INTELLIGENCE
              </p>
            </div>
          </div>

          <p className="text-[9px] tracking-[0.15em] text-slate-400">
            ADAPTIVE ML-BASED IOT INTRUSION DETECTION FRAMEWORK
          </p>
        </div>
      </footer>
    </main>
  );
}