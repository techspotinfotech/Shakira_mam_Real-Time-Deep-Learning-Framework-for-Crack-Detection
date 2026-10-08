import { useEffect, useState } from 'react';

const emptyForm = {
  title: '',
  description: '',
  status: 'Planned',
  priority: 'Medium',
};

const emptyAnalysisForm = {
  title: '',
  domain: 'AI/ML',
  description: '',
};

function App() {
  const [projects, setProjects] = useState([]);
  const [stats, setStats] = useState({ total_projects: 0, planned: 0, in_progress: 0, completed: 0, high_priority: 0 });
  const [form, setForm] = useState(emptyForm);
  const [analysisForm, setAnalysisForm] = useState(emptyAnalysisForm);
  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [loading, setLoading] = useState(false);

  const loadProjects = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/projects');
      const data = await response.json();
      setProjects(data);
    } catch (error) {
      console.error('Failed to load projects', error);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats', error);
    }
  };

  const refreshData = async () => {
    await Promise.all([loadProjects(), loadStats()]);
  };

  useEffect(() => {
    refreshData();
  }, []);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  };

  const handleAnalysisChange = (event) => {
    const { name, value } = event.target;
    setAnalysisForm((current) => ({ ...current, [name]: value }));
  };

  const handleAnalyze = async (event) => {
    event.preventDefault();
    setAnalyzing(true);

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analysisForm),
      });

      const data = await response.json();
      setAnalysis(data);
    } catch (error) {
      console.error('AI analysis failed', error);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);

    try {
      await fetch('http://localhost:8000/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      setForm(emptyForm);
      await refreshData();
    } catch (error) {
      console.error('Failed to create project', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (projectId) => {
    await fetch(`http://localhost:8000/api/projects/${projectId}`, {
      method: 'DELETE',
    });
    await refreshData();
  };

  return (
    <div className="app-shell">
      <div className="card">
        <div className="header-row">
          <div>
            <p className="eyebrow">API Backend Service</p>
            <h1>Project Office Dashboard</h1>
          </div>
          <span className="pill">FastAPI + React</span>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <span>Total projects</span>
            <strong>{stats.total_projects}</strong>
          </div>
          <div className="stat-card">
            <span>Planned</span>
            <strong>{stats.planned}</strong>
          </div>
          <div className="stat-card">
            <span>In progress</span>
            <strong>{stats.in_progress}</strong>
          </div>
          <div className="stat-card">
            <span>Completed</span>
            <strong>{stats.completed}</strong>
          </div>
          <div className="stat-card highlight">
            <span>High priority</span>
            <strong>{stats.high_priority}</strong>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="project-form">
          <div className="form-grid">
            <label>
              Project title
              <input name="title" value={form.title} onChange={handleChange} placeholder="Enter project name" required />
            </label>

            <label>
              Status
              <select name="status" value={form.status} onChange={handleChange}>
                <option value="Planned">Planned</option>
                <option value="In Progress">In Progress</option>
                <option value="Completed">Completed</option>
              </select>
            </label>

            <label className="full-width">
              Description
              <textarea
                name="description"
                value={form.description}
                onChange={handleChange}
                rows={5}
                placeholder="Describe the backend service, workflow, or feature"
                required
              />
            </label>

            <label>
              Priority
              <select name="priority" value={form.priority} onChange={handleChange}>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
              </select>
            </label>
          </div>

          <div className="button-row">
            <button type="submit" disabled={loading}>{loading ? 'Saving...' : 'Add project'}</button>
          </div>
        </form>

        <form onSubmit={handleAnalyze} className="project-form analysis-form">
          <div className="header-row mb-20">
            <div>
              <p className="eyebrow">AI Model</p>
              <h2>Project Feasibility Analyzer</h2>
            </div>
          </div>

          <div className="form-grid">
            <label>
              Project title
              <input name="title" value={analysisForm.title} onChange={handleAnalysisChange} placeholder="Enter idea title" required />
            </label>

            <label>
              Domain
              <select name="domain" value={analysisForm.domain} onChange={handleAnalysisChange}>
                <option value="AI/ML">AI/ML</option>
                <option value="Web Development">Web Development</option>
                <option value="Data Analytics">Data Analytics</option>
                <option value="Automation">Automation</option>
                <option value="Healthcare">Healthcare</option>
                <option value="Finance">Finance</option>
                <option value="Education">Education</option>
              </select>
            </label>

            <label className="full-width">
              Description
              <textarea
                name="description"
                value={analysisForm.description}
                onChange={handleAnalysisChange}
                rows={5}
                placeholder="Describe the project idea, goals, and expected impact"
                required
              />
            </label>
          </div>

          <div className="button-row">
            <button type="submit" disabled={analyzing}>{analyzing ? 'Analyzing...' : 'Analyze with AI'}</button>
          </div>
        </form>

        {analysis && (
          <div className="analysis-panel">
            <div className="metric-row">
              <div className="metric">
                <span>Feasibility score</span>
                <strong>{analysis.feasibility_score}%</strong>
              </div>
              <div className="metric">
                <span>Risk level</span>
                <strong>{analysis.risk_level}</strong>
              </div>
            </div>

            <p className="recommendation">{analysis.recommendation}</p>
            <div className="details-grid">
              <div>
                <span className="label">Project</span>
                <strong>{analysis.project_title}</strong>
              </div>
              <div>
                <span className="label">Domain</span>
                <strong>{analysis.domain}</strong>
              </div>
            </div>

            <div className="focus-areas">
              {analysis.matched_focus_areas.map((area) => (
                <span key={area} className="focus-tag">{area}</span>
              ))}
            </div>
          </div>
        )}

        <div className="project-list">
          {projects.length === 0 ? (
            <p className="empty-state">No projects yet. Add the first one above.</p>
          ) : (
            projects.map((project) => (
              <div key={project.id} className="project-card">
                <div className="project-head">
                  <div>
                    <h3>{project.title}</h3>
                    <span className={`badge ${project.priority.toLowerCase()}`}>{project.priority}</span>
                  </div>
                  <button className="delete-btn" type="button" onClick={() => handleDelete(project.id)}>Delete</button>
                </div>

                <p>{project.description}</p>
                <div className="meta-row">
                  <span className="status-tag">{project.status}</span>
                  <span>{new Date(project.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
