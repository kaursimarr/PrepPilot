import { useState, useEffect, useCallback } from 'react'
import {
  Target, Award, CheckCircle2, AlertTriangle, Clock, Play,
  ChevronLeft, ArrowRight, BookOpen, Layers3, RefreshCw, LoaderCircle,
  FileText, ShieldAlert, Sparkles, Zap
} from 'lucide-react'
import '../examPrep.css'
import { getPrerequisiteGraph, getSyllabus, BOOK_TITLES } from '../api'

const BOOKS = ['CLRS', 'DS', 'C', 'CPP']

const EXAM_SLOTS = [
  { key: 'T1', label: 'Test 1 (T1) Midterm', scopeDesc: 'Foundations Tier — Initial 33.3% Syllabus' },
  { key: 'T2', label: 'Test 2 (T2) Midterm', scopeDesc: 'Intermediate Tier — Core 33.3% Syllabus' },
  { key: 'T3', label: 'Test 3 (T3) Comprehensive', scopeDesc: 'Full End-Term Exam — 100% Comprehensive Syllabus' },
]

export default function PrepareForExam({ onBackToWorkspace, onNavigateToPractice, onNavigateToMock }) {
  const [book, setBook] = useState('CLRS')
  const [slot, setSlot] = useState('T1')
  const [graphData, setGraphData] = useState(null)
  const [syllabusData, setSyllabusData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const loadData = useCallback(async (b) => {
    setLoading(true)
    setError(null)
    try {
      const [g, s] = await Promise.all([
        getPrerequisiteGraph(b, 'default').catch(() => null),
        getSyllabus(b).catch(() => null),
      ])
      setGraphData(g)
      setSyllabusData(s)
    } catch (e) {
      setError(e.message || 'Failed to load exam preparation analytics.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData(book)
  }, [book, loadData])

  // Extract chapters relevant to the selected exam slot
  const allNodes = graphData?.nodes || []
  const examChapters = allNodes.filter(n => {
    if (slot === 'T3') return true // T3 is comprehensive: includes all chapters
    return n.exam_slot === slot
  })

  // Compute Done vs Left metrics
  const total = examChapters.length
  const completed = examChapters.filter(n => n.status === 'completed')
  const weak = examChapters.filter(n => n.status === 'weak')
  const unattempted = examChapters.filter(n => n.attempts === 0)
  const readinessPct = total > 0 ? Math.round((completed.length / total) * 100) : 0

  return (
    <div className="prep-page">
      {/* Top Back Navigation */}
      <div>
        <button className="back-nav-btn" onClick={onBackToWorkspace}>
          <ChevronLeft size={15} /> Back to Workspace
        </button>
      </div>

      {/* Header */}
      <div className="prep-header">
        <div className="prep-title-wrap">
          <div className="prep-icon-box"><Target size={22} /></div>
          <div>
            <h1 className="prep-title">Prepare for an Exam</h1>
            <p className="prep-subtitle">
              Syllabus diagnostic tracking done concepts, weak areas, and uncompleted topics for T1, T2 &amp; T3.
            </p>
          </div>
        </div>

        {/* Textbook Selector */}
        <div className="prep-book-selector">
          {BOOKS.map(b => (
            <button
              key={b}
              className={`prep-book-btn ${book === b ? 'active' : ''}`}
              onClick={() => setBook(b)}
            >
              {b}
            </button>
          ))}
        </div>
      </div>

      {/* Exam Slot Selector Tabs (T1, T2, T3) */}
      <div className="prep-slot-tabs">
        {EXAM_SLOTS.map(s => (
          <button
            key={s.key}
            className={`prep-slot-tab ${slot === s.key ? 'active' : ''} ${s.key.toLowerCase()}`}
            onClick={() => setSlot(s.key)}
          >
            <div className="slot-tab-header">
              <span className="slot-tab-title">{s.label}</span>
              <span className={`slot-badge ${s.key.toLowerCase()}`}>{s.key}</span>
            </div>
            <span className="slot-tab-desc">{s.scopeDesc}</span>
          </button>
        ))}
      </div>

      {loading ? (
        <div className="prep-loading-box">
          <LoaderCircle size={28} className="spin" />
          <span>Analyzing exam syllabus and student mastery...</span>
        </div>
      ) : error ? (
        <div className="prep-error-box">⚠ {error}</div>
      ) : (
        <>
          {/* Readiness & Leftness Diagnostic Cards */}
          <div className="prep-summary-grid">
            {/* Card 1: Exam Readiness Score */}
            <div className="prep-card readiness-card">
              <div className="card-top-label">EXAM READINESS SCORE</div>
              <div className="readiness-score-row">
                <div className="score-number">{readinessPct}%</div>
                <div className="score-details">
                  <div className="score-tier">
                    {readinessPct >= 80 ? 'Exam Ready' : readinessPct >= 40 ? 'Moderate Preparation' : 'Preparation Needed'}
                  </div>
                  <div className="score-count">{completed.length} of {total} chapters mastered</div>
                </div>
              </div>
              <div className="prep-prog-track">
                <div className="prep-prog-fill" style={{ width: `${readinessPct}%` }} />
              </div>
            </div>

            {/* Card 2: The Leftness Breakdown */}
            <div className="prep-card leftness-card">
              <div className="card-top-label">WHAT IS LEFT (DIAGNOSTIC BREAKDOWN)</div>
              <div className="leftness-items-list">
                <div className="leftness-item">
                  <Clock size={16} className="text-dim" />
                  <div>
                    <b>{unattempted.length} chapters unattempted</b>
                    <span>No quiz or practice attempts recorded yet</span>
                  </div>
                </div>
                <div className="leftness-item">
                  <AlertTriangle size={16} className="text-amber" />
                  <div>
                    <b>{weak.length} chapters need revision</b>
                    <span>Accuracy &lt; 50% or low theta ability</span>
                  </div>
                </div>
                <div className="leftness-item">
                  <CheckCircle2 size={16} className="text-green" />
                  <div>
                    <b>{completed.length} chapters mastered</b>
                    <span>Sufficient practice completed with high accuracy</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Chapters Checklist (Done vs Left) */}
          <div className="prep-chapters-panel">
            <div className="chapters-panel-header">
              <div>
                <h2 className="panel-heading">{BOOK_TITLES[book] || book} — {slot} Chapter Diagnostic</h2>
                <p className="panel-sub">
                  Detailed status of each chapter in this exam scope showing what is done and what remains left.
                </p>
              </div>
              <button
                className="primary-btn prep-mock-btn"
                onClick={() => onNavigateToMock && onNavigateToMock(book, slot)}
              >
                <FileText size={15} /> Take {slot} Practice Mock Exam (25 Qs)
              </button>
            </div>

            <div className="prep-chapters-list">
              {examChapters.map(ch => {
                const isDone = ch.status === 'completed'
                const isWeak = ch.status === 'weak'
                const isLeft = ch.attempts === 0

                return (
                  <div
                    key={ch.chapter}
                    className={`prep-chapter-row ${isDone ? 'done' : isWeak ? 'weak' : 'left'}`}
                  >
                    {/* Left: Status Icon & Info */}
                    <div className="prep-ch-main">
                      <div className={`status-pill ${isDone ? 'done' : isWeak ? 'weak' : 'left'}`}>
                        {isDone ? (
                          <><CheckCircle2 size={14} /> Mastered (Done)</>
                        ) : isWeak ? (
                          <><AlertTriangle size={14} /> Needs Revision</>
                        ) : (
                          <><Clock size={14} /> Left to Study</>
                        )}
                      </div>
                      <div className="prep-ch-text">
                        <div className="ch-num-title">
                          <span className="ch-num">CH {ch.chapter}</span>
                          <span className="ch-title">{ch.chapter_title || `Chapter ${ch.chapter}`}</span>
                          <span className="ch-slot-badge">{ch.exam_slot || slot}</span>
                        </div>
                        <div className="ch-meta">
                          <span>{ch.chunks || 0} textbook passages</span>
                          <span>·</span>
                          <span>{ch.attempts} practice attempts ({ch.accuracy || 0}% accuracy)</span>
                          {ch.prerequisites && ch.prerequisites.length > 0 && (
                            <>
                              <span>·</span>
                              <span className="ch-prereq-note">
                                Requires Ch. {ch.prerequisites.map(p => p.chapter).join(', ')}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Right: Leftness note & Action */}
                    <div className="prep-ch-action">
                      <div className="leftness-status-text">
                        {isDone ? (
                          <span className="text-green">✓ Completed &amp; Verified</span>
                        ) : isWeak ? (
                          <span className="text-amber">⚠ Re-study weak concepts</span>
                        ) : (
                          <span className="text-dim">Pending: Complete initial quiz</span>
                        )}
                      </div>
                      <button
                        className="prep-practice-btn"
                        onClick={() => onNavigateToPractice && onNavigateToPractice(book, ch.chapter)}
                      >
                        <Play size={13} /> Practice
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
