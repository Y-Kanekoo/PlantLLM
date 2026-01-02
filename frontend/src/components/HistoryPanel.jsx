import { formatDateTime } from "../utils/format.js";

export default function HistoryPanel({ history, onOpenEntry, onClear }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>診断履歴</h3>
        <button type="button" className="ghost" onClick={onClear}>
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
              onClick={() => onOpenEntry(entry.id)}
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
  );
}
