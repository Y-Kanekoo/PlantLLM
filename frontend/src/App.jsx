import { useEffect, useMemo, useState } from "react";

const API_HEADERS = { "Content-Type": "application/json" };

const formatDateTime = (value) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("ja-JP");
};

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

  const latestResults = useMemo(() => results.slice(0, 8), [results]);

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
          </div>
        </div>

        <div className="hero-card">
          <div className="card-header">
            <h2>画像を診断する</h2>
            <p>ドラッグ&ドロップ、複数枚まとめてOK。</p>
          </div>
          <div
            className={`drop-zone ${dragActive ? "active" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            <div>
              <strong>画像をここにドロップ</strong>
              <span>またはファイルを選択</span>
            </div>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(event) => handleFiles(event.target.files)}
            />
          </div>

          <div className="form-row">
            <label htmlFor="model-select">使用モデル</label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
            >
              {models.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.description}
                </option>
              ))}
            </select>
          </div>

          <div className="queue-grid">
            {queue.map((item) => (
              <div className="queue-item" key={item.id}>
                <img src={item.preview} alt="preview" />
                <button
                  type="button"
                  className="ghost"
                  onClick={() => removeFromQueue(item.id)}
                >
                  削除
                </button>
              </div>
            ))}
          </div>

          <button
            type="button"
            className="primary"
            onClick={diagnoseQueue}
            disabled={!queue.length || status.uploading}
          >
            {status.uploading
              ? `診断中 ${status.progress}%`
              : `診断開始 (${totalQueued}枚)`}
          </button>

          {status.uploading ? (
            <div className="progress">
              <span style={{ width: `${status.progress}%` }} />
            </div>
          ) : null}

          {error ? <div className="error-card">{error}</div> : null}
        </div>
      </header>

      <main className="content-grid">
        <section className="panel">
          <div className="panel-header">
            <h3>最新の診断結果</h3>
            <span>{results.length} 件</span>
          </div>

          <div className="results-list">
            {latestResults.length === 0 ? (
              <p className="muted">まだ結果がありません。</p>
            ) : (
              latestResults.map((result, index) => (
                <article
                  className="result-card"
                  key={`${result.entry_id}-${index}`}
                  style={{ animationDelay: `${index * 0.05}s` }}
                >
                  <div className="result-image">
                    <img src={result.image?.url} alt="診断画像" />
                    {result.from_cache ? (
                      <span className="pill">cache</span>
                    ) : null}
                  </div>
                  <div className="result-body">
                    <div className="result-meta">
                      <span>{formatDateTime(result.timestamp)}</span>
                      <span>{result.model?.description}</span>
                    </div>
                    <pre>{result.diagnosis}</pre>
                    <div className="result-footer">
                      <span>処理時間 {result.processing_time}s</span>
                      <button
                        type="button"
                        className="ghost"
                        onClick={() => openHistoryEntry(result.entry_id)}
                      >
                        詳細を見る
                      </button>
                    </div>
                  </div>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h3>診断履歴</h3>
            <button type="button" className="ghost" onClick={clearHistory}>
              履歴を削除
            </button>
          </div>
          <div className="history-list">
            {history.length === 0 ? (
              <p className="muted">履歴がまだありません。</p>
            ) : (
              history.map((entry, index) => (
                <button
                  type="button"
                  className="history-item"
                  key={entry.id}
                  style={{ animationDelay: `${index * 0.03}s` }}
                  onClick={() => openHistoryEntry(entry.id)}
                >
                  <img src={entry.image?.url} alt="診断" />
                  <div>
                    <strong>{formatDateTime(entry.timestamp)}</strong>
                    <p>{entry.diagnosis}</p>
                    <span>
                      {entry.model?.description} · {entry.processing_time}s
                    </span>
                  </div>
                </button>
              ))
            )}
          </div>
        </section>
      </main>

      {activeEntry ? (
        <div className="drawer" role="dialog" aria-modal="true">
          <div className="drawer-content">
            <button
              type="button"
              className="drawer-close"
              onClick={() => setActiveEntry(null)}
            >
              ×
            </button>
            <div className="drawer-grid">
              <div>
                <img
                  className="drawer-image"
                  src={activeEntry.image?.url}
                  alt="診断画像"
                />
                <div className="drawer-meta">
                  <strong>{activeEntry.model?.description}</strong>
                  <span>{formatDateTime(activeEntry.timestamp)}</span>
                  <span>処理時間 {activeEntry.processing_time}s</span>
                </div>
                <pre className="drawer-diagnosis">{activeEntry.diagnosis}</pre>
              </div>
              <div className="drawer-chat">
                <h4>追加で質問する</h4>
                <div className="chat-window">
                  {(activeEntry.chat || []).length === 0 ? (
                    <p className="muted">まだチャット履歴がありません。</p>
                  ) : (
                    (activeEntry.chat || []).map((chat, index) => (
                      <div
                        key={`${chat.timestamp}-${index}`}
                        className={`chat-bubble ${chat.role}`}
                      >
                        <span>{chat.message}</span>
                        <time>{formatDateTime(chat.timestamp)}</time>
                      </div>
                    ))
                  )}
                </div>
                <div className="chat-input">
                  <textarea
                    rows="3"
                    placeholder="気になる点や症状を入力..."
                    value={chatInput}
                    onChange={(event) => setChatInput(event.target.value)}
                  />
                  <button
                    type="button"
                    className="primary"
                    disabled={chatSending || !chatInput.trim()}
                    onClick={sendChatMessage}
                  >
                    {chatSending ? "送信中..." : "送信"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
