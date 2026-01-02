import { formatDateTime } from "../utils/format.js";

export default function ResultsPanel({ results, totalCount, onOpenEntry }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>最新の診断結果</h3>
        <span>{totalCount} 件</span>
      </div>

      <div className="results-list">
        {results.length === 0 ? (
          <p className="muted">まだ結果がありません。</p>
        ) : (
          results.map((result, index) => (
            <article
              className="result-card"
              key={`${result.entry_id}-${index}`}
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <div className="result-image">
                <img src={result.image?.url} alt="診断画像" />
                {result.from_cache ? <span className="pill">cache</span> : null}
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
                    onClick={() => onOpenEntry(result.entry_id)}
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
  );
}
