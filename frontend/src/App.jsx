import { useEffect, useMemo, useState } from "react";
import UploadCard from "./components/UploadCard.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";
import HistoryPanel from "./components/HistoryPanel.jsx";
import EntryDrawer from "./components/EntryDrawer.jsx";
import { formatDateTime } from "./utils/format.js";

const API_HEADERS = { "Content-Type": "application/json" };

const toPreviewItem = (file) => ({
  id: `${file.name}-${file.size}-${file.lastModified}-${Math.random()}`,
  file,
  preview: URL.createObjectURL(file),
});

export default function App() {
  const [queue, setQueue] = useState([]);
  const [results, setResults] = useState([]);
  const [history, setHistory] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [status, setStatus] = useState({ uploading: false, progress: 0 });
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [activeEntry, setActiveEntry] = useState(null);
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);

  useEffect(() => {
    loadModels();
    loadHistory();
  }, []);

  useEffect(() => {
    return () => {
      queue.forEach((item) => URL.revokeObjectURL(item.preview));
    };
  }, [queue]);

  const totalQueued = queue.length;
  const queuedSize = queue.reduce((sum, item) => sum + item.file.size, 0);

  const latestResults = useMemo(() => results.slice(0, 8), [results]);
  const averageTime = useMemo(() => {
    if (!results.length) return null;
    const total = results.reduce(
      (sum, item) => sum + (item.processing_time || 0),
      0
    );
    return (total / results.length).toFixed(2);
  }, [results]);
  const lastRunAt = results[0]?.timestamp;

  const loadModels = async () => {
    try {
      const response = await fetch("/models");
      const data = await response.json();
      setModels(data.models || []);
      setSelectedModel(data.default || "");
    } catch (err) {
      setError("モデル情報の取得に失敗しました。");
    }
  };

  const loadHistory = async () => {
    try {
      const response = await fetch("/history");
      const data = await response.json();
      if (data.success) {
        setHistory(data.history || []);
      }
    } catch (err) {
      setError("履歴の取得に失敗しました。");
    }
  };

  const handleFiles = (files) => {
    const images = Array.from(files).filter((file) =>
      file.type.startsWith("image/")
    );
    if (!images.length) return;

    setQueue((prev) => [...prev, ...images.map(toPreviewItem)]);
  };

  const removeFromQueue = (id) => {
    setQueue((prev) => {
      const target = prev.find((item) => item.id === id);
      if (target) {
        URL.revokeObjectURL(target.preview);
      }
      return prev.filter((item) => item.id !== id);
    });
  };

  const clearQueue = () => {
    queue.forEach((item) => URL.revokeObjectURL(item.preview));
    setQueue([]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    if (event.dataTransfer?.files) {
      handleFiles(event.dataTransfer.files);
    }
  };

  const diagnoseQueue = async () => {
    if (!queue.length || status.uploading) return;

    setError("");
    setStatus({ uploading: true, progress: 0 });

    const total = queue.length;
    const newResults = [];

    for (let i = 0; i < total; i += 1) {
      const item = queue[i];
      const formData = new FormData();
      formData.append("file", item.file);

      const query = selectedModel
        ? `?model_name=${encodeURIComponent(selectedModel)}`
        : "";

      try {
        const response = await fetch(`/diagnose${query}`, {
          method: "POST",
          body: formData,
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
          setError(data.detail || "診断に失敗しました。");
        } else {
          newResults.unshift(data);
        }
      } catch (err) {
        setError("診断中にネットワークエラーが発生しました。");
      }

      const progressValue = Math.round(((i + 1) / total) * 100);
      setStatus({ uploading: true, progress: progressValue });
    }

    setResults((prev) => [...newResults, ...prev]);
    clearQueue();
    setStatus({ uploading: false, progress: 0 });
    await loadHistory();
  };

  const openHistoryEntry = async (entryId) => {
    try {
      const response = await fetch(`/history/${entryId}`);
      const data = await response.json();
      if (data.success) {
        setActiveEntry(data.entry);
      }
    } catch (err) {
      setError("履歴詳細の取得に失敗しました。");
    }
  };

  const sendChatMessage = async () => {
    if (!activeEntry || !chatInput.trim()) return;

    setChatSending(true);
    try {
      const response = await fetch(`/chat/${activeEntry.id}`, {
        method: "POST",
        headers: API_HEADERS,
        body: JSON.stringify({ message: chatInput.trim() }),
      });
      const data = await response.json();
      if (data.success) {
        setActiveEntry((prev) => {
          if (!prev) return prev;
          const updatedChat = [
            ...(prev.chat || []),
            {
              role: "user",
              message: chatInput.trim(),
              timestamp: new Date().toISOString(),
            },
            {
              role: "assistant",
              message: data.reply,
              timestamp: new Date().toISOString(),
            },
          ];
          return { ...prev, chat: updatedChat };
        });
        setChatInput("");
      }
    } catch (err) {
      setError("チャット送信に失敗しました。");
    } finally {
      setChatSending(false);
    }
  };

  const clearHistory = async () => {
    if (!window.confirm("診断履歴をすべて削除します。よろしいですか？")) {
      return;
    }

    try {
      const response = await fetch("/clear-history", { method: "POST" });
      const data = await response.json();
      if (data.success) {
        setHistory([]);
        setActiveEntry(null);
      }
    } catch (err) {
      setError("履歴の削除に失敗しました。");
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-copy">
          <span className="eyebrow">PlantLLM</span>
          <h1>植物の調子を、数秒で見立てる。</h1>
          <p>
            画像をアップロードすると、植物の種類と健康状態を推定し、
            すぐに対策のヒントを返します。研究用の履歴とチャットも一括管理。
          </p>
          <div className="hero-meta">
            <div>
              <strong>最新Flash</strong>
              <span>Gemini 2.0 Flash を優先</span>
            </div>
            <div>
              <strong>履歴管理</strong>
              <span>SQLiteで自動保存</span>
            </div>
            <div>
              <strong>平均処理時間</strong>
              <span>{averageTime ? `${averageTime}s` : "計測中"}</span>
            </div>
          </div>
          {lastRunAt ? (
            <div className="hero-last-run">
              最終診断: {formatDateTime(lastRunAt)}
            </div>
          ) : null}
        </div>

        <UploadCard
          dragActive={dragActive}
          onDrop={handleDrop}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onFiles={handleFiles}
          models={models}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          queue={queue}
          queuedSize={queuedSize}
          totalQueued={totalQueued}
          onRemoveQueue={removeFromQueue}
          onClearQueue={clearQueue}
          onDiagnose={diagnoseQueue}
          status={status}
          error={error}
        />
      </header>

      <main className="content-grid">
        <ResultsPanel
          results={latestResults}
          totalCount={results.length}
          onOpenEntry={openHistoryEntry}
        />
        <HistoryPanel
          history={history}
          onOpenEntry={openHistoryEntry}
          onClear={clearHistory}
        />
      </main>

      <EntryDrawer
        entry={activeEntry}
        onClose={() => setActiveEntry(null)}
        chatInput={chatInput}
        onChatInputChange={setChatInput}
        onSendChat={sendChatMessage}
        chatSending={chatSending}
      />
    </div>
  );
}
