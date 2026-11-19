import { useState, useEffect, useCallback } from 'react'
import { GraduationCap, RotateCcw, ChevronLeft, LoaderCircle, Sparkles, BookOpen } from 'lucide-react'
import '../syllabus.css'
import {
  getSyllabus,
  getSyllabusView,
  assignChapterToExam,
  resetSyllabusPartition,
  BOOK_TITLES,
} from '../api'

const BOOKS = ['CLRS', 'DS', 'C', 'CPP']

const TABS = [
  { key: 'T1',    label: 'T1 — Test 1 (33%)',           cls: 't1' },
  { key: 'T2',    label: 'T2 — Test 2 (33%)',           cls: 't2' },
  { key: 'T1+T2', label: 'T1 + T2 Combined (66%)',      cls: 't1t2' },
  { key: 'T3',    label: 'T3 — End Term (Whole Syllabus)', cls: 't3' },
  { key: 'ALL',   label: 'All Chapters',                cls: 'all' },
]

function slotCls(slot) {
  if (!slot) return 'un'
  return slot.toLowerCase().replace('+', '')
}

function ChapterCard({ ch, onChangeSlot }) {
  const cls = slotCls(ch.exam_slot)
  return (
    <div className={`chapter-card ${cls}`}>
      <div className="chapter-card-top">
        <span className="chapter-num">CH {ch.chapter}</span>
        <span className="chapter-title">{ch.chapter_title || `Chapter ${ch.chapter}`}</span>
        <span className={`chapter-badge ${cls}`}>{ch.exam_slot || '—'}</span>
      </div>
      <div className="chapter-card-footer">
        <span className="chapter-chunks">{ch.chunks} chunks</span>
        <select
          className="syl-assign-select"
          value={ch.exam_slot || ''}
          onChange={e => onChangeSlot(ch.chapter, e.target.value)}
        >
          <option value="">Unassigned</option>
          <option value="T1">T1 (Test 1)</option>
          <option value="T2">T2 (Test 2)</option>
          <option value="T3">T3 (Test 3)</option>
        </select>
      </div>
    </div>
  )
}

export default function SyllabusIntelligence({ onBackToWorkspace }) {
  const [book, setBook]           = useState('CLRS')
  const [activeTab, setActiveTab] = useState('ALL')
  const [summary, setSummary]     = useState(null)
  const [viewData, setViewData]   = useState(null)
  const [loading, setLoading]     = useState(false)
  const [resetting, setResetting] = useState(false)
  const [error, setError]         = useState(null)

  const loadSummary = useCallback(async (b) => {
    setLoading(true)
    setError(null)
    try {
      const s = await getSyllabus(b)
      setSummary(s)
    } catch (e) {
      setError(e.message || 'Failed to load syllabus.')
    } finally {
      setLoading(false)
    }
  }, [])

  const loadView = useCallback(async (b, tab) => {
    setLoading(true)
    setError(null)
    try {
      const d = await getSyllabusView(b, tab)
      setViewData(d)
    } catch (e) {
      setError(e.message || 'Failed to load view.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    setSummary(null)
    setViewData(null)
    loadSummary(book)
    if (activeTab !== 'ALL') loadView(book, activeTab)
  }, [book]) // eslint-disable-line

  useEffect(() => {
    setViewData(null)
    if (activeTab !== 'ALL') loadView(book, activeTab)
  }, [activeTab]) // eslint-disable-line

  const handleResetPartition = async () => {
    setResetting(true)
    setError(null)
    try {
      await resetSyllabusPartition(book)
      await loadSummary(book)
      if (activeTab !== 'ALL') await loadView(book, activeTab)
    } catch (e) {
      setError(e.message || 'Reset failed.')
    } finally {
      setResetting(false)
    }
  }

  const handleChangeSlot = async (chapter, newSlot) => {
    if (!newSlot) return
    setError(null)
    try {
      await assignChapterToExam(book, String(chapter), newSlot)
      await loadSummary(book)
      if (activeTab !== 'ALL') await loadView(book, activeTab)
    } catch (e) {
      setError(e.message || 'Failed to assign slot.')
    }
  }

  /* Filter chapters based on active tab */
  const displayChapters = (() => {
    if (activeTab === 'ALL') {
      return summary
        ? [
            ...(summary.T1?.chapters || []),
            ...(summary.T2?.chapters || []),
            ...(summary.T3?.new_chapters || summary.T3?.chapters || []),
            ...(summary.unassigned?.chapters || []),
          ]
        : []
    }
    return viewData?.chapters || []
  })()

  const isCombined = activeTab === 'T1+T2'
  const isT3All = activeTab === 'T3'

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
            Syllabus Intelligence
            <div className="syl-subtitle">Exam-segmented syllabus: T1 (33.3%) · T2 (33.3%) · T3 (Whole Syllabus)</div>
          </div>
        </div>

        <div className="syl-header-right">
          {/* Book selector */}
          <div className="syl-book-select">
            {BOOKS.map(b => (
              <button
                key={b}
                className={`syl-book-btn${book === b ? ' active' : ''}`}
                onClick={() => setBook(b)}
              >
                {b}
              </button>
            ))}
          </div>

          {/* Reset button */}
          <button
            className="syl-reset-btn"
            onClick={handleResetPartition}
            disabled={resetting}
            title="Reset to 33.33% equal division"
          >
            {resetting ? <LoaderCircle size={13} className="syl-spin" /> : <RotateCcw size={13} />}
            Reset 33%
          </button>
        </div>
      </div>

      {/* Division Overview Cards */}
      {summary && (
        <div className="syl-division-banner">
          <div className="syl-division-card t1">
            <div className="syl-div-header">
              <span className="syl-div-name">Test 1 (T1)</span>
              <span className="syl-div-pct">33.3%</span>
            </div>
            <div className="syl-div-count">{summary.T1?.count ?? 0} chapters</div>
            <div className="syl-div-desc">Initial foundation syllabus tested in Exam 1</div>
          </div>

          <div className="syl-division-card t2">
            <div className="syl-div-header">
              <span className="syl-div-name">Test 2 (T2)</span>
              <span className="syl-div-pct">33.3%</span>
            </div>
            <div className="syl-div-count">{summary.T2?.count ?? 0} chapters</div>
            <div className="syl-div-desc">Core intermediate concepts tested in Exam 2</div>
          </div>

          <div className="syl-division-card t1t2">
            <div className="syl-div-header">
              <span className="syl-div-name">T1 + T2 Combined</span>
              <span className="syl-div-pct">66.7%</span>
            </div>
            <div className="syl-div-count">{summary.T1_plus_T2?.count ?? 0} chapters</div>
            <div className="syl-div-desc">Cumulative syllabus through mid-term scope</div>
          </div>

          <div className="syl-division-card t3">
            <div className="syl-div-header">
              <span className="syl-div-name">Test 3 / End Term (T3)</span>
              <span className="syl-div-pct">100% Full</span>
            </div>
            <div className="syl-div-count">{summary.total_chapters ?? 0} chapters</div>
            <div className="syl-div-desc">Comprehensive exam: T1 + T2 + Remaining chapters</div>
          </div>
        </div>
      )}

      {/* Error banner */}
      {error && <div className="syl-error">⚠ {error}</div>}

      {/* Exam Tabs */}
      <div className="exam-tab-row">
        {TABS.map(t => (
          <button
            key={t.key}
            className={`exam-tab${activeTab === t.key ? ` active ${t.cls}` : ''}`}
            onClick={() => setActiveTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* T3 notice */}
      {isT3All && (
        <div className="syl-scope-notice">
          <BookOpen size={16} />
          <span><b>T3 Comprehensive Scope:</b> Includes all chapters across the entire course syllabus (T1 Foundations + T2 Intermediate + T3 Remaining).</span>
        </div>
      )}

      {/* Chapter Grid */}
      {loading ? (
        <div className="syl-loading">
          <LoaderCircle size={20} className="syl-spin" /> Loading syllabus chapters…
        </div>
      ) : (
        <div className="syl-grid">
          {displayChapters.length === 0 && (
            <div className="syl-empty">
              No chapters found for this selection.
            </div>
          )}

          {/* Grouped view for T3 (Whole syllabus) */}
          {isT3All ? (
            <>
              {(summary?.T1?.chapters || []).length > 0 && (
                <>
                  <div className="syl-section-label t1">
                    <span>▸ T1 Foundation Scope (33.3%)</span>
                    <span>{summary.T1.count} chapters</span>
                  </div>
                  {(summary?.T1?.chapters || []).map(ch => (
                    <ChapterCard key={ch.chapter} ch={ch} onChangeSlot={handleChangeSlot} />
                  ))}
                </>
              )}

              {(summary?.T2?.chapters || []).length > 0 && (
                <>
                  <div className="syl-section-label t2">
                    <span>▸ T2 Intermediate Scope (33.3%)</span>
                    <span>{summary.T2.count} chapters</span>
                  </div>
                  {(summary?.T2?.chapters || []).map(ch => (
                    <ChapterCard key={ch.chapter} ch={ch} onChangeSlot={handleChangeSlot} />
                  ))}
                </>
              )}

              {((summary?.T3?.new_chapters || []).length > 0) && (
                <>
                  <div className="syl-section-label t3">
                    <span>▸ T3 Advanced / Remaining Scope (33.4%)</span>
                    <span>{summary.T3.new_count || summary.T3.new_chapters.length} chapters</span>
                  </div>
                  {(summary?.T3?.new_chapters || []).map(ch => (
                    <ChapterCard key={ch.chapter} ch={ch} onChangeSlot={handleChangeSlot} />
                  ))}
                </>
              )}
            </>
          ) : isCombined ? (
            /* T1+T2 combined view */
            <>
              {(summary?.T1?.chapters || []).length > 0 && (
                <>
                  <div className="syl-section-label t1">
                    <span>▸ T1 Scope</span>
                    <span>{summary.T1.count} chapters</span>
                  </div>
                  {(summary?.T1?.chapters || []).map(ch => (
                    <ChapterCard key={ch.chapter} ch={ch} onChangeSlot={handleChangeSlot} />
                  ))}
                </>
              )}
              {(summary?.T2?.chapters || []).length > 0 && (
                <>
                  <div className="syl-section-label t2">
                    <span>▸ T2 Scope</span>
                    <span>{summary.T2.count} chapters</span>
                  </div>
                  {(summary?.T2?.chapters || []).map(ch => (
                    <ChapterCard key={ch.chapter} ch={ch} onChangeSlot={handleChangeSlot} />
                  ))}
                </>
              )}
            </>
          ) : (
            displayChapters.map(ch => (
              <ChapterCard key={ch.chapter} ch={ch} onChangeSlot={handleChangeSlot} />
            ))
          )}
        </div>
      )}
    </div>
  )
}
