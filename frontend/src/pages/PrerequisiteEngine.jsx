import { useState, useEffect, useCallback } from 'react'
import {
  GitFork, CheckCircle2, AlertTriangle, Clock, ArrowRight,
  BookOpen, ChevronLeft, LoaderCircle, Sparkles, Zap, ShieldCheck
} from 'lucide-react'
import '../prereq.css'
import { getPrerequisiteGraph, BOOK_TITLES } from '../api'

const BOOKS = ['CLRS', 'DS', 'C', 'CPP']

export default function PrerequisiteEngine({ onBackToWorkspace, onNavigateToQuiz }) {
  const [book, setBook] = useState('CLRS')
  const [graphData, setGraphData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('sequence') // 'sequence' | 'explorer'
  const [selectedNode, setSelectedNode] = useState(null)
  const [filterSlot, setFilterSlot] = useState('ALL') // 'ALL' | 'T1' | 'T2' | 'T3'

  const loadGraph = useCallback(async (b) => {
    setLoading(true)
    setError(null)
    try {
      const data = await getPrerequisiteGraph(b)
      setGraphData(data)
      if (data?.nodes?.length > 0) {
        setSelectedNode(data.nodes[0])
      }
    } catch (e) {
      setError(e.message || 'Failed to load prerequisite graph.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadGraph(book)
  }, [book, loadGraph])

  const summary = graphData?.summary
  const nodes = graphData?.nodes || []
  const sequence = graphData?.learning_sequence || []

  // Filter nodes for explorer
  const filteredNodes = nodes.filter(n => {
    if (filterSlot === 'ALL') return true
    return n.exam_slot === filterSlot
  })

  return (
    <div className="prereq-page">
      {/* Back button */}
      <div>
        <button className="back-nav-btn" onClick={onBackToWorkspace}>
          <ChevronLeft size={15} /> Back to Workspace
        </button>
      </div>

      {/* Header */}
      <div className="prereq-header">
        <div className="prereq-title">
          <div className="prereq-title-icon"><GitFork size={20} /></div>
          <div>
            Prerequisite Engine
            <div className="prereq-subtitle">Topic dependencies, student mastery readiness &amp; personalized learning sequence</div>
          </div>
        </div>

        {/* Book selector */}
        <div className="prereq-book-select">
          {BOOKS.map(b => (
            <button
              key={b}
              className={`prereq-book-btn${book === b ? ' active' : ''}`}
              onClick={() => setBook(b)}
            >
              {b}
            </button>
          ))}
        </div>
      </div>

      {/* Readiness Metrics Cards */}
      {summary && (
        <div className="prereq-metrics-banner">
          <div className="prereq-metric-card readiness">
            <span className="prereq-metric-label">Course Readiness</span>
            <span className="prereq-metric-val">{summary.readiness_percentage}%</span>
            <span className="prereq-metric-desc">{summary.completed} of {summary.total_chapters} chapters mastered</span>
          </div>

          <div className="prereq-metric-card completed">
            <span className="prereq-metric-label">Mastered Topics</span>
            <span className="prereq-metric-val" style={{ color: '#34d399' }}>{summary.completed}</span>
            <span className="prereq-metric-desc">Prerequisites fully satisfied</span>
          </div>

          <div className="prereq-metric-card weak">
            <span className="prereq-metric-label">Needs Practice</span>
            <span className="prereq-metric-val" style={{ color: '#fbbf24' }}>{summary.weak}</span>
            <span className="prereq-metric-desc">Weak foundational knowledge detected</span>
          </div>

          <div className="prereq-metric-card pending">
            <span className="prereq-metric-label">Pending Syllabus</span>
            <span className="prereq-metric-val" style={{ color: '#94a3b8' }}>{summary.pending}</span>
            <span className="prereq-metric-desc">Unassessed course chapters</span>
          </div>
        </div>
      )}

      {/* Error alert */}
      {error && <div style={{ padding: '12px 16px', background: '#351b22', border: '1px solid #632a36', borderRadius: '10px', color: '#ff9ea9', fontSize: '0.82rem' }}>⚠ {error}</div>}

      {/* Navigation Tabs */}
      <div className="prereq-tab-row">
        <button
          className={`prereq-tab${activeTab === 'sequence' ? ' active' : ''}`}
          onClick={() => setActiveTab('sequence')}
        >
          <Sparkles size={15} /> Recommended Learning Sequence ({sequence.length})
        </button>
        <button
          className={`prereq-tab${activeTab === 'explorer' ? ' active' : ''}`}
          onClick={() => setActiveTab('explorer')}
        >
          <GitFork size={15} /> Dependency Graph Explorer ({nodes.length})
        </button>
      </div>

      {/* View 1: Recommended Learning Sequence */}
      {activeTab === 'sequence' && (
        <div>
          {loading ? (
            <div className="prereq-loading">
              <LoaderCircle size={22} className="prereq-spin" /> Analyzing prerequisite dependencies &amp; optimizing learning path…
            </div>
          ) : (
            <div className="prereq-sequence-container">
              {sequence.map((step) => {
                const isWeak = step.status === 'weak'
                const isCompleted = step.status === 'completed'
                return (
                  <div
                    key={step.step}
                    className={`sequence-step-card ${step.status}`}
                  >
                    <div className="sequence-step-left">
                      <div className="sequence-step-num">{step.step}</div>
                      <div className="sequence-step-info">
                        <h4>
                          Ch. {step.chapter}: {step.chapter_title || `Chapter ${step.chapter}`}
                          <span className={`ad-tier-tag ${step.exam_slot.toLowerCase()}`} style={{ marginLeft: '8px' }}>
                            {step.exam_slot}
                          </span>
                        </h4>
                        <p>
                          <span>{step.recommendation}</span>
                          {step.prerequisites.length > 0 && (
                            <span style={{ color: '#64748b' }}>
                              · Requires: {step.prerequisites.map(p => `Ch. ${p.chapter}`).join(', ')}
                            </span>
                          )}
                          {step.unlocks_count > 0 && (
                            <span style={{ color: '#38bdf8' }}>
                              · Unlocks {step.unlocks_count} downstream topic(s)
                            </span>
                          )}
                        </p>
                      </div>
                    </div>

                    <div className="sequence-step-right">
                      <span className={`status-pill ${step.status}`}>
                        <span className="dot" />
                        {step.status === 'completed' ? 'Mastered' : step.status === 'weak' ? 'Weak Foundation' : 'Pending'}
                      </span>

                      {onNavigateToQuiz && (
                        <button
                          className="practice-node-btn"
                          onClick={() => onNavigateToQuiz(book, step.chapter)}
                          title="Practice this chapter in Adaptive Quiz"
                        >
                          <Zap size={12} /> Practice
                        </button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* View 2: Dependency Graph Explorer */}
      {activeTab === 'explorer' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {/* Exam Tier Filter */}
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase' }}>Filter Tier:</span>
            {['ALL', 'T1', 'T2', 'T3'].map(slot => (
              <button
                key={slot}
                onClick={() => setFilterSlot(slot)}
                style={{
                  padding: '5px 12px',
                  borderRadius: '6px',
                  border: filterSlot === slot ? '1px solid var(--primary)' : '1px solid var(--card-border)',
                  background: filterSlot === slot ? 'var(--badge-bg)' : 'var(--card-bg)',
                  color: filterSlot === slot ? 'var(--primary)' : 'var(--text-muted)',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                {slot === 'ALL' ? 'All Tiers' : slot}
              </button>
            ))}
          </div>

          {loading ? (
            <div className="prereq-loading">
              <LoaderCircle size={22} className="prereq-spin" /> Loading topic nodes…
            </div>
          ) : (
            <div className="prereq-grid">
              {filteredNodes.map(node => (
                <div
                  key={node.chapter}
                  className={`prereq-node-card${selectedNode?.chapter === node.chapter ? ' selected' : ''}`}
                  onClick={() => setSelectedNode(node)}
                >
                  <div className="prereq-node-top">
                    <div>
                      <div className="prereq-node-title">
                        Ch. {node.chapter}: {node.chapter_title}
                      </div>
                      <div className="prereq-node-meta">
                        <span className={`ad-tier-tag ${node.exam_slot.toLowerCase()}`}>{node.exam_slot}</span>
                        <span>· {node.chunks} chunks</span>
                        <span>· θ {node.theta}</span>
                      </div>
                    </div>

                    <span className={`status-pill ${node.status}`}>
                      <span className="dot" />
                      {node.status}
                    </span>
                  </div>

                  {/* Prerequisites (Requires) */}
                  <div className="prereq-dep-group">
                    <span className="prereq-dep-title">Requires Prerequisite(s):</span>
                    <div className="prereq-dep-items">
                      {node.prerequisites.length === 0 ? (
                        <span className="dep-none">None (Course Entry Point)</span>
                      ) : (
                        node.prerequisites.map(p => (
                          <span key={p.chapter} className={`dep-chip ${p.exam_slot.toLowerCase()}`}>
                            Ch. {p.chapter} ({p.exam_slot})
                          </span>
                        ))
                      )}
                    </div>
                  </div>

                  {/* Unlocks (Downstream) */}
                  <div className="prereq-dep-group">
                    <span className="prereq-dep-title">Unlocks Downstream Topic(s):</span>
                    <div className="prereq-dep-items">
                      {node.unlocks.length === 0 ? (
                        <span className="dep-none">End of Track</span>
                      ) : (
                        node.unlocks.map(u => (
                          <span key={u.chapter} className={`dep-chip ${u.exam_slot.toLowerCase()}`}>
                            Ch. {u.chapter}
                          </span>
                        ))
                      )}
                    </div>
                  </div>

                  {/* Action */}
                  {onNavigateToQuiz && (
                    <div style={{ marginTop: 'auto', paddingTop: '6px' }}>
                      <button
                        className="practice-node-btn"
                        onClick={(e) => {
                          e.stopPropagation()
                          onNavigateToQuiz(book, node.chapter)
                        }}
                      >
                        <Zap size={12} /> Practice in Adaptive Quiz
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
