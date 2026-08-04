import { useState } from 'react';

function App() {
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed');
      }
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">AI-assisted medical screening</p>
          <h1>Brain Stroke AI</h1>
          <p className="hero-copy">Upload a brain scan to receive a fast stroke-type assessment and a clear summary of the model’s confidence.</p>
        </div>
      </header>

      <main className="content-stack">
        <section className="panel hero-panel">
          <h2>Home Overview</h2>
          <p>This application helps identify whether a scan suggests a stroke and provides a simple, professional summary for ischemic or hemorrhagic patterns.</p>
        </section>

        <section className="grid">
          <div className="panel">
            <h2>Image Upload</h2>
            <label className="upload-box">
              <input type="file" accept="image/*" onChange={handleUpload} />
              <span>Choose image</span>
            </label>
            {imagePreview && (
              <div className="preview-card">
                <img src={imagePreview} alt="Uploaded scan" />
              </div>
            )}
          </div>

          <div className="panel">
            <h2>Prediction Analysis</h2>
            {loading && <p>Analyzing scan…</p>}
            {error && <p className="error">{error}</p>}
            {result && (
              <div className="result-card">
                <div className="hero-score">
                  <span className="label">Stroke Type</span>
                  <div className="value">{result.stroke_type}</div>
                </div>
                <div className="score-grid">
                  <div className="score-box">
                    <span className="label">Confidence Score</span>
                    <strong>{result.confidence_score?.toFixed(2)}%</strong>
                  </div>
                  <div className="score-box accent">
                    <span className="label">Severity</span>
                    <strong>{result.severity || 'Unknown'}</strong>
                  </div>
                </div>
                {result.probabilities?.length > 0 && (
                  <div className="score-box" style={{ marginTop: '12px' }}>
                    <span className="label">Probability Summary</span>
                    <strong>{result.probabilities[0].class}: {result.probabilities[0].probability.toFixed(2)}%</strong>
                  </div>
                )}
                {result.report_url && (
                  <a className="report-link" href={result.report_url} target="_blank" rel="noreferrer">
                    Download clinical report
                  </a>
                )}
              </div>
            )}
            {!loading && !result && !error && <p>Upload an image to begin the analysis.</p>}
          </div>
        </section>

        <section className="panel overview-panel">
          <h2>Quick Stroke Overview</h2>
          <div className="overview-grid">
            <article>
              <h3>Ischemic Stroke</h3>
              <p>Occurs when a blood vessel is blocked, reducing oxygen flow to part of the brain.</p>
            </article>
            <article>
              <h3>Hemorrhagic Stroke</h3>
              <p>Occurs when a blood vessel ruptures and bleeding occurs inside or around the brain.</p>
            </article>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
