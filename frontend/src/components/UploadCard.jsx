import { formatBytes } from "../utils/format.js";

export default function UploadCard({
  dragActive,
  onDrop,
  onDragOver,
  onDragLeave,
  onFiles,
  models,
  selectedModel,
  onModelChange,
  queue,
  queuedSize,
  totalQueued,
  onRemoveQueue,
  onClearQueue,
  onDiagnose,
  status,
  error,
}) {
  return (
    <div className="hero-card">
      <div className="card-header">
        <h2>画像を診断する</h2>
        <p>ドラッグ&ドロップ、複数枚まとめてOK。</p>
      </div>

      <div
        className={`drop-zone ${dragActive ? "active" : ""}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <div>
          <strong>画像をここにドロップ</strong>
          <span>またはファイルを選択</span>
        </div>
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={(event) => onFiles(event.target.files)}
        />
      </div>

      <div className="form-row">
        <label htmlFor="model-select">使用モデル</label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(event) => onModelChange(event.target.value)}
          disabled={!models.length}
        >
          {!models.length ? (
            <option value="">モデルを取得中...</option>
          ) : null}
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.description}
            </option>
          ))}
        </select>
      </div>

      <div className="queue-meta">
        <span>{totalQueued}枚の画像を準備中</span>
        <span>合計 {formatBytes(queuedSize)}</span>
        <button type="button" className="ghost" onClick={onClearQueue}>
          クリア
        </button>
      </div>

      <div className="queue-grid">
        {queue.map((item) => (
          <div className="queue-item" key={item.id}>
            <img src={item.preview} alt="preview" />
            <button
              type="button"
              className="ghost"
              onClick={() => onRemoveQueue(item.id)}
            >
              削除
            </button>
          </div>
        ))}
      </div>

      <button
        type="button"
        className="primary"
        onClick={onDiagnose}
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
      <div className="helper-text">
        ※ 画像は10MB以下、JPEG/PNG/GIFに対応しています。
      </div>
    </div>
  );
}
