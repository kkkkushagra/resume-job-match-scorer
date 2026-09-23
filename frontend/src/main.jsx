import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AlertCircle, ArrowUpRight, Check, FileText, Gauge, LoaderCircle, Plus, Sparkles, Upload, X } from 'lucide-react';
import './styles.css';

const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '');

function FilePill({ file, onRemove }) {
  return (
    <div className="file-pill">
      <FileText size={17} />
      <span>{file.name}</span>
      <button type="button" aria-label={`Remove ${file.name}`} onClick={onRemove}><X size={14} /></button>
    </div>
  );
}

const matchClass = {
  'Exact match': 'exact',
  'Strong semantic match': 'strong',
  'Related but not confirmed': 'related',
  'Partial match': 'partial',
  'Not found': 'missing',
};

function ScoreCard({ result, index }) {
  const score = Math.round(result.score);
  const analysis = result.analysis;
  return (
    <article className="result-card" style={{ '--delay': `${index * 90}ms` }}>
      <div className="result-topline">
        <div className="resume-name"><span className="file-mark"><FileText size={17} /></span><strong>{result.filename}</strong></div>
        <div className="score-number"><span>{score}</span><small>/ 100</small></div>
      </div>
      <div className="score-track"><div className="score-fill" style={{ width: `${Math.min(score, 100)}%` }} /></div>
      <div className="result-grid">
        <div>
          <div className="label success"><Check size={15} /> Matched skills <b>{result.matched.length}</b></div>
          <div className="tag-list">{result.matched.length ? result.matched.map(skill => <span className="tag tag-green" key={skill}>{skill}</span>) : <span className="empty">No shared skills found</span>}</div>
        </div>
        <div>
          <div className="label warning"><AlertCircle size={15} /> Missing skills <b>{result.missing.length}</b></div>
          <div className="tag-list">{result.missing.length ? result.missing.map(skill => <span className="tag tag-amber" key={skill}>{skill}</span>) : <span className="empty">Full skill coverage</span>}</div>
        </div>
      </div>
      {analysis && <details className="evidence-panel" open>
        <summary><span>Requirement evidence</span><small>{analysis.requirements.length} requirements analyzed</small></summary>
        <div className="evidence-summary"><span className="evidence-count strong">{analysis.strong_matches.length} strong</span><span className="evidence-count partial">{analysis.partial_matches.length} partial</span><span className="evidence-count related">{analysis.related_matches.length} related</span><span className="evidence-count missing">{analysis.missing.length} missing</span></div>
        <div className="requirement-list">{analysis.requirements.map(item => <div className="requirement" key={item.requirement}>
          <div className="requirement-top"><strong>{item.requirement}</strong><span className={`match-badge ${matchClass[item.match] || 'missing'}`}>{item.match}</span></div>
          <div className="requirement-meta">{item.category} · {item.evidence_strength} evidence{item.mandatory ? ' · mandatory' : ''}</div>
          <p>{item.reason}</p>
          {item.evidence && <blockquote>{item.evidence}</blockquote>}
        </div>)}</div>
      </details>}
    </article>
  );
}

function App() {
  const [jdText, setJdText] = useState('');
  const [jdFile, setJdFile] = useState(null);
  const [resumes, setResumes] = useState([]);
  const [method, setMethod] = useState('semantic');
  const [mode, setMode] = useState('single');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const addResumes = (event) => {
    const selected = Array.from(event.target.files || []);
    setResumes(current => mode === 'single' ? selected.slice(0, 1) : [...current, ...selected]);
    event.target.value = '';
  };

  const handleModeChange = (nextMode) => {
    setMode(nextMode);
    if (nextMode === 'single') setResumes(current => current.slice(0, 1));
  };

  const scoreResumes = async (event) => {
    event.preventDefault();
    if ((!jdText.trim() && !jdFile) || !resumes.length) {
      setError('Add a job description and at least one resume to continue.');
      return;
    }
    setLoading(true);
    setError('');
    setResults([]);
    const formData = new FormData();
    if (jdText.trim()) formData.append('jd_text', jdText);
    if (jdFile) formData.append('jd_file', jdFile);
    resumes.forEach(file => formData.append('resumes', file));
    formData.append('method', method);
    try {
      const response = await fetch(`${API_URL}/api/score`, { method: 'POST', body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Unable to score resumes.');
      setResults(payload.results);
    } catch (requestError) {
      setError(requestError.message || 'Something went wrong while scoring.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar"><div className="brand"><span className="brand-icon"><Sparkles size={17} /></span><span>matchline</span></div><span className="status-dot">LOCAL WORKSPACE</span></header>
      <section className="hero">
        <div className="eyebrow"><Gauge size={15} /> Resume intelligence, made legible</div>
        <h1>Find the signal<br /><em>inside the stack.</em></h1>
        <p>Compare resumes against a role in seconds. Matchline surfaces the evidence behind every score, so a shortlist starts with context.</p>
      </section>

      <form className="workspace" onSubmit={scoreResumes}>
        <section className="panel jd-panel">
          <div className="section-heading"><span className="step">01</span><div><h2>Role brief</h2><p>Paste the job description or bring in a file.</p></div></div>
          <textarea value={jdText} onChange={event => setJdText(event.target.value)} placeholder="Paste the role description here..." />
          <div className="upload-line">
            <label className="file-button"><Upload size={16} /> {jdFile ? 'Replace file' : 'Upload brief'}<input type="file" accept=".pdf,.docx,.txt" onChange={event => setJdFile(event.target.files?.[0] || null)} /></label>
            {jdFile && <FilePill file={jdFile} onRemove={() => setJdFile(null)} />}
            <span className="hint">PDF, DOCX, or TXT</span>
          </div>
        </section>

        <section className="panel resume-panel">
          <div className="section-heading"><span className="step">02</span><div><h2>Candidate files</h2><p>Build a focused shortlist or rank the whole set.</p></div></div>
          <div className="mode-switch" role="group" aria-label="Candidate mode"><button type="button" className={mode === 'single' ? 'active' : ''} onClick={() => handleModeChange('single')}>Single candidate</button><button type="button" className={mode === 'batch' ? 'active' : ''} onClick={() => handleModeChange('batch')}>Rank a batch</button></div>
          <label className="drop-zone"><input type="file" accept=".pdf,.docx,.txt" multiple={mode === 'batch'} onChange={addResumes} /><span className="upload-circle"><Plus size={21} /></span><strong>{resumes.length ? 'Add another resume' : 'Choose resume files'}</strong><small>PDF, DOCX, or TXT · {mode === 'single' ? 'one file' : 'multiple files'}</small></label>
          {resumes.length > 0 && <div className="file-list">{resumes.map((file, index) => <FilePill key={`${file.name}-${index}`} file={file} onRemove={() => setResumes(current => current.filter((_, itemIndex) => itemIndex !== index))} />)}</div>}
        </section>

        <section className="action-row">
          <div className="method-control"><span>Scoring model</span><select value={method} onChange={event => setMethod(event.target.value)}><option value="semantic">Semantic embeddings</option><option value="tfidf">TF-IDF, lightweight</option></select></div>
          <button className="score-button" type="submit" disabled={loading}>{loading ? <><LoaderCircle className="spin" size={18} /> Scoring...</> : <>Score {resumes.length || ''} resume{resumes.length === 1 ? '' : 's'} <ArrowUpRight size={18} /></>}</button>
        </section>
        {error && <div className="error-message"><AlertCircle size={17} /> {error}</div>}
      </form>

      {results.length > 0 && <section className="results"><div className="results-heading"><div><div className="eyebrow">03 / Analysis complete</div><h2>Your shortlist, decoded.</h2></div><span className="result-count">{results.length} {results.length === 1 ? 'result' : 'results'}</span></div><div className="result-list">{results.map((result, index) => <ScoreCard result={result} index={index} key={result.filename} />)}</div></section>}
      <footer><span>Matchline</span><span>Decision support, not a hiring decision.</span></footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
