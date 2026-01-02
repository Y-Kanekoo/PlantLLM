import { formatDateTime } from "../utils/format.js";

export default function EntryDrawer({
  entry,
  onClose,
  chatInput,
  onChatInputChange,
  onSendChat,
  chatSending,
}) {
  if (!entry) return null;

  return (
    <div className="drawer" role="dialog" aria-modal="true">
      <div className="drawer-content">
        <button type="button" className="drawer-close" onClick={onClose}>
          ×
        </button>
        <div className="drawer-grid">
          <div>
            <img className="drawer-image" src={entry.image?.url} alt="診断画像" />
            <div className="drawer-meta">
              <strong>{entry.model?.description}</strong>
              <span>{formatDateTime(entry.timestamp)}</span>
              <span>処理時間 {entry.processing_time}s</span>
            </div>
            <pre className="drawer-diagnosis">{entry.diagnosis}</pre>
          </div>
          <div className="drawer-chat">
            <h4>追加で質問する</h4>
            <div className="chat-window">
              {(entry.chat || []).length === 0 ? (
                <p className="muted">まだチャット履歴がありません。</p>
              ) : (
                (entry.chat || []).map((chat, index) => (
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
                onChange={(event) => onChatInputChange(event.target.value)}
              />
              <button
                type="button"
                className="primary"
                disabled={chatSending || !chatInput.trim()}
                onClick={onSendChat}
              >
                {chatSending ? "送信中..." : "送信"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
