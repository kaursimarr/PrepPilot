import { useState, useEffect, useCallback } from 'react'
import {
  GraduationCap, ChevronLeft, LoaderCircle, CheckCircle2,
  AlertTriangle, Clock, Play, Search, BookOpen
} from 'lucide-react'
import '../syllabus.css'
import {
  getPrerequisiteGraph,
  BOOK_TITLES,
} from '../api'

const BOOK_TABS = [
  { key: 'CLRS', label: 'Algorithms (CLRS)' },
  { key: 'DS', label: 'Data Structures' },
  { key: 'C', label: 'Let Us C' },
  { key: 'CPP', label: 'C++ Reference' },
]

const FILTER_TABS = [
  { key: 'ALL', label: 'All Chapters' },
  { key: 'T1', label: 'Test 1 (T1) Midterm' },
  { key: 'T2', label: 'Test 2 (T2) Midterm' },
  { key: 'T3', label: 'Test 3 (T3) Comprehensive' },
]

export default function SyllabusIntelligence({ onBackToWorkspace, onNavigateToPractice }) {
  const [book, setBook] = useState('CLRS')
  const [activeTab, setActiveTab] = useState('ALL')
  const [searchQuery, setSearchQuery] = useState('')
  const [graphData, setGraphData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const loadData = useCallback(async (b) => {
    setLoading(true)
    setError(null)
    try {
      const g = await getPrerequisiteGraph(b, 'default')
      setGraphData(g)
    } catch (e) {
      setError(e.message || 'Failed to load syllabus.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData(book)
  }, [book, loadData])

  const nodes = graphData?.nodes || []

  // Filter chapters by search query and slot
  const filteredNodes = nodes.filter(n => {
    if (activeTab !== 'ALL' && n.exam_slot !== activeTab) return false
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      const title = (n.chapter_title || '').toLowerCase()
      const ch = String(n.chapter)
      return title.includes(q) || ch === q
    }
    return true
  })

  return (
    <div className="syl-page">
      {/* Back button */}
      <div>
        <button className="back-nav-btn" onClick={onBackToWorkspace}>
          <ChevronLeft size={15} /> Back to Workspace
        </button>
      </div>

      {/* Header */}
      <div className="syl-header">
        <div className="syl-title">
          <div className="syl-title-icon"><GraduationCap size={20} /></div>
          <div>
            Syllabus &amp; Course Roadmap
            <div className="syl-subtitle">
              {BOOK_TITLES[book] || book} — Clean vertical directory of chapters, prerequisites, and exam tiers.
            </div>
          </div>
        </div>

        {/* Book selector */}
        <div className="syl-book-select">
          {BOOK_TABS.map(b => (
            <button
              key={b.key}
              className={`syl-book-btn ${book === b.key ? 'active' : ''}`}
              onClick={() => {
                setBook(b.key)
                setSearchQuery('')
              }}
            >
              {b.label}
            </button>
          ))}
        </div>
      </div>

      {/* Quick Summary Strip */}
      <div className="syl-summary-strip">
        <span className="syl-stat-item"><b>{nodes.length}</b> Chapters</span>
        <span className="syl-stat-dot">·</span>
        <span className="syl-stat-item"><b>{nodes.filter(n => n.exam_slot === 'T1').length}</b> in T1 Midterm</span>
        <span className="syl-stat-dot">·</span>
        <span className="syl-stat-item"><b>{nodes.filter(n => n.exam_slot === 'T2').length}</b> in T2 Midterm</span>
        <span className="syl-stat-dot">·</span>
        <span className="syl-stat-item"><b>{nodes.filter(n => n.exam_slot === 'T3').length}</b> in T3 Comprehensive</span>
        <span className="syl-stat-dot">·</span>
        <span className="syl-stat-item text-success"><b>{nodes.filter(n => n.status === 'completed').length}</b> Mastered</span>
      </div>

      {/* Controls Bar: Search & Exam Slot Filter */}
      <div className="syl-controls-bar">
        <div className="syl-search-box">
          <Search size={14} className="syl-search-icon" />
          <input
            type="text"
            placeholder="Search chapters by title or chapter number..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="syl-slot-pills">
          {FILTER_TABS.map(tab => (
            <button
              key={tab.key}
              className={`syl-slot-pill ${activeTab === tab.key ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="syl-loading-panel">
          <LoaderCircle size={24} className="spin" />
          <span>Loading syllabus chapters...</span>
        </div>
      ) : error ? (
        <div className="syl-error-panel">⚠ {error}</div>
      ) : (
        /* Vertical Chapter Cards List */
        <div className="syl-vertical-container">
          <div className="syl-vertical-list">
            {filteredNodes.length === 0 ? (
              <div className="syl-empty-note">No chapters match your search filter.</div>
            ) : (
              filteredNodes.map(ch => {
                const isCompleted = ch.status === 'completed'
                const isWeak = ch.status === 'weak'
                const prereqs = ch.prerequisites || []
                const unlocks = ch.unlocks || []

                return (
                  <div key={ch.chapter} className="syl-chapter-vertical-row">
                    {/* Left: Index badge */}
                    <div className="syl-ch-index-badge">
                      {String(ch.chapter).padStart(2, '0')}
                    </div>

                    {/* Middle: Chapter info, prerequisites and unlocks */}
                    <div className="syl-row-left">
                      <div className="syl-ch-header">
                        <span className="syl-ch-number">
                          Chapter {ch.chapter}
                        </span>
                        <span className={`syl-ch-slot-badge ${ch.exam_slot?.toLowerCase() || 't1'}`}>
                          {ch.exam_slot === 'T1' ? 'T1 Midterm' : ch.exam_slot === 'T2' ? 'T2 Midterm' : 'T3 Final'}
                        </span>
                        {prereqs.length > 0 ? (
                          <span className="syl-prereq-badge">
                            Requires Ch. {prereqs.map(p => p.chapter).join(', ')}
                          </span>
                        ) : (
                          <span className="syl-prereq-badge foundational">
                            Foundational Topic
                          </span>
                        )}
                        {unlocks.length > 0 && (
                          <span className="syl-unlock-badge">
                            Unlocks Ch. {unlocks.map(u => u.chapter).slice(0, 3).join(', ')}
                          </span>
                        )}
                      </div>
                      <h3 className="syl-ch-title">{ch.chapter_title || `Chapter ${ch.chapter}`}</h3>
                    </div>

                    {/* Right: Status badge & Action */}
                    <div className="syl-row-right">
                      <div className="syl-status-badge">
                        {isCompleted ? (
                          <span className="badge-done"><CheckCircle2 size={13} /> Mastered</span>
                        ) : isWeak ? (
                          <span className="badge-weak"><AlertTriangle size={13} /> Needs Review</span>
                        ) : (
                          <span className="badge-pending"><Clock size={13} /> Unattempted</span>
                        )}
                      </div>

                      <button
                        className="syl-row-practice-btn"
                        onClick={() => onNavigateToPractice && onNavigateToPractice(book, ch.chapter)}
                      >
                        <Play size={13} /> Practice
                      </button>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      )}
    </div>
  )
}
