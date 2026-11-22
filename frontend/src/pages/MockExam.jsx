import { useState, useEffect, useRef, useCallback } from 'react'
import {
  FileText, Clock3, Award, CheckCircle2, XCircle, AlertTriangle,
  ChevronLeft, ChevronRight, Play, RotateCcw, Flag, BookOpen,
  Sparkles, Check, HelpCircle, LoaderCircle
} from 'lucide-react'
import '../mockExam.css'
import {
  createMockExam, getMockExam, submitMockExam, getMockExamHistory,
  BOOK_TITLES
} from '../api'

const BOOKS = ['CLRS', 'DS', 'C', 'CPP']

export default function MockExam({ onBackToWorkspace, initialBook = 'CLRS', initialSlot = 'T1' }) {
  // Navigation / Mode state
  const [viewState, setViewState] = useState('launcher') // 'launcher' | 'exam' | 'scorecard'
  const [book, setBook] = useState(initialBook || 'CLRS')

  useEffect(() => {
    if (initialBook) setBook(initialBook)
  }, [initialBook])

  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [starting, setStarting] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  // Active exam state
  const [activeExam, setActiveExam] = useState(null)
  const [currentIdx, setCurrentIdx] = useState(0)
  const [answers, setAnswers] = useState({}) // { questionId: 'A' }
  const [markedForReview, setMarkedForReview] = useState(new Set())
  const [secondsRemaining, setSecondsRemaining] = useState(1800)
  const [examResult, setExamResult] = useState(null)

  const timerRef = useRef(null)
  const startTimeRef = useRef(Date.now())

  // Load history
  const loadHistory = useCallback(async (b) => {
    try {
      const list = await getMockExamHistory('default', b)
      setHistory(list || [])
    } catch (e) {
      console.warn('Failed to load history:', e)
    }
  }, [])

  useEffect(() => {
    loadHistory(book)
  }, [book, loadHistory])

  // Timer countdown
  useEffect(() => {
    if (viewState === 'exam' && secondsRemaining > 0) {
      timerRef.current = setInterval(() => {
        setSecondsRemaining(prev => {
          if (prev <= 1) {
            clearInterval(timerRef.current)
            handleAutoSubmit()
            return 0
          }
          return prev - 1
        })
      }, 1000)
    }
    return () => clearInterval(timerRef.current)
  }, [viewState]) // eslint-disable-line

  // Start Exam
  const handleLaunchExam = async (examType, questionCount, durationMin) => {
    setStarting(true)
    setError(null)
    try {
      const res = await createMockExam('default', book, examType, questionCount, durationMin)
      setActiveExam(res)
      setAnswers({})
      setMarkedForReview(new Set())
      setCurrentIdx(0)
      setSecondsRemaining(durationMin * 60)
      startTimeRef.current = Date.now()
      setViewState('exam')
    } catch (e) {
      setError(e.message || 'Failed to start exam. Check question bank.')
    } finally {
      setStarting(false)
    }
  }

  // Answer selection
  const handleSelectOption = (questionId, optionLetter) => {
    setAnswers(prev => ({
      ...prev,
      [String(questionId)]: optionLetter
    }))
  }

  const handleClearAnswer = (questionId) => {
    setAnswers(prev => {
      const next = { ...prev }
      delete next[String(questionId)]
      return next
    })
  }

  const toggleMarkForReview = (questionId) => {
    setMarkedForReview(prev => {
      const next = new Set(prev)
      if (next.has(questionId)) next.delete(questionId)
      else next.add(questionId)
      return next
    })
  }

  // Submit Exam
  const handleSubmitExam = async () => {
    if (!window.confirm('Are you sure you want to finish and submit your exam?')) {
      return
    }
    await doSubmit()
  }

  const handleAutoSubmit = async () => {
    alert('Time has expired! Submitting your answers automatically.')
    await doSubmit()
  }

  const doSubmit = async () => {
    clearInterval(timerRef.current)
    setSubmitting(true)
    setError(null)
    const timeSpentSec = Math.round((Date.now() - startTimeRef.current) / 1000)
    try {
      const result = await submitMockExam(activeExam.exam_id, answers, timeSpentSec)
      setExamResult(result)
      setViewState('scorecard')
      loadHistory(book)
    } catch (e) {
      setError(e.message || 'Failed to grade exam.')
    } finally {
      setSubmitting(false)
    }
  }

  // Format Timer
  const formatTimer = (totalSec) => {
    const mins = Math.floor(totalSec / 60)
    const secs = totalSec % 60
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }

  // Active question
  const currentQ = activeExam?.questions?.[currentIdx]
  const isLastQ = activeExam ? currentIdx === activeExam.questions.length - 1 : false

  return (
    <div className="mock-page">
      {/* Back button */}
      <div>
        <button
          className="back-nav-btn"
          onClick={() => {
            if (viewState === 'exam') {
              if (window.confirm('Are you sure you want to exit the active exam? Your progress will not be graded.')) {
                setViewState('launcher')
              }
            } else if (viewState === 'scorecard') {
              setViewState('launcher')
            } else {
              onBackToWorkspace()
            }
          }}
        >
          <ChevronLeft size={15} /> {viewState === 'exam' ? 'Exit Exam' : viewState === 'scorecard' ? 'Back to Exam Catalog' : 'Back to Workspace'}
        </button>
      </div>

      {/* ==================== VIEW 1: EXAM LAUNCHER ==================== */}
      {viewState === 'launcher' && (
        <>
          <div className="mock-header">
            <div className="mock-title">
              <div className="mock-title-icon"><FileText size={20} /></div>
              <div>
                Practice Mock Exam
                <div className="mock-subtitle">Standardized, timed exam simulation for T1, T2 &amp; T3 comprehensive scope</div>
              </div>
            </div>

            <div className="mock-book-select">
              {BOOKS.map(b => (
                <button
                  key={b}
                  className={`mock-book-btn${book === b ? ' active' : ''}`}
                  onClick={() => setBook(b)}
                >
                  {b}
                </button>
              ))}
            </div>
          </div>

          {error && <div style={{ padding: '12px 16px', background: '#351b22', border: '1px solid #632a36', borderRadius: '10px', color: '#ff9ea9', fontSize: '0.82rem' }}>⚠ {error}</div>}

          {/* Exam Category Cards */}
          <div className="mock-categories-grid">
            {/* T1 Midterm */}
            <div className="mock-category-card t1">
              <div className="mock-cat-header">
                <div>
                  <div className="mock-cat-title">Test 1 (T1) Midterm</div>
                  <span style={{ fontSize: '0.75rem', color: '#60a5fa', fontWeight: 600 }}>Foundation Tier</span>
                </div>
                <span className="mock-cat-badge">T1 Scope</span>
              </div>
              <p className="mock-cat-desc">
                Covers the first ~33.3% of the syllabus. Tests mathematical foundations, core syntax, and base structures.
              </p>
              <div className="mock-cat-specs">
                <div className="mock-spec-item">
                  <span className="mock-spec-val">25</span>
                  <span className="mock-spec-label">Questions</span>
                </div>
                <div className="mock-spec-item">
                  <span className="mock-spec-val">30</span>
                  <span className="mock-spec-label">Minutes</span>
                </div>
                <div className="mock-spec-item">
                  <span className="mock-spec-val">25</span>
                  <span className="mock-spec-label">Marks</span>
                </div>
              </div>
              <button
                className="mock-launch-btn"
                disabled={starting}
                onClick={() => handleLaunchExam('T1', 25, 30)}
              >
                {starting ? <LoaderCircle size={15} className="spin" /> : <Play size={15} />}
                Start T1 Examination
              </button>
            </div>

            {/* T2 Midterm */}
            <div className="mock-category-card t2">
              <div className="mock-cat-header">
                <div>
                  <div className="mock-cat-title">Test 2 (T2) Midterm</div>
                  <span style={{ fontSize: '0.75rem', color: '#c084fc', fontWeight: 600 }}>Intermediate Tier</span>
                </div>
                <span className="mock-cat-badge">T2 Scope</span>
              </div>
              <p className="mock-cat-desc">
                Covers the middle ~33.3% of the syllabus. Tests dynamic algorithms, balanced trees, and graph fundamentals (requires T1 prerequisites).
              </p>
              <div className="mock-cat-specs">
                <div className="mock-spec-item">
                  <span className="mock-spec-val">25</span>
                  <span className="mock-spec-label">Questions</span>
                </div>
                <div className="mock-spec-item">
                  <span className="mock-spec-val">30</span>
                  <span className="mock-spec-label">Minutes</span>
                </div>
                <div className="mock-spec-item">
                  <span className="mock-spec-val">25</span>
                  <span className="mock-spec-label">Marks</span>
                </div>
              </div>
              <button
                className="mock-launch-btn"
                disabled={starting}
                onClick={() => handleLaunchExam('T2', 25, 30)}
              >
                {starting ? <LoaderCircle size={15} className="spin" /> : <Play size={15} />}
                Start T2 Examination
              </button>
            </div>

            {/* T3 Comprehensive Final */}
            <div className="mock-category-card t3">
              <div className="mock-cat-header">
                <div>
                  <div className="mock-cat-title">Test 3 (T3) Comprehensive</div>
                  <span style={{ fontSize: '0.75rem', color: '#f59e0b', fontWeight: 600 }}>End-Term Full Scope</span>
                </div>
                <span className="mock-cat-badge">100% Full Syllabus</span>
              </div>
              <p className="mock-cat-desc">
                Complete course evaluation: balanced across T1 Foundations (30%), T2 Core (30%), and T3 Advanced Mastery (40%).
              </p>
              <div className="mock-cat-specs">
                <div className="mock-spec-item">
                  <span className="mock-spec-val">50</span>
                  <span className="mock-spec-label">Questions</span>
                </div>
                <div className="mock-spec-item">
                  <span className="mock-spec-val">60</span>
                  <span className="mock-spec-label">Minutes</span>
                </div>
                <div className="mock-spec-item">
                  <span className="mock-spec-val">50</span>
                  <span className="mock-spec-label">Marks</span>
                </div>
              </div>
              <button
                className="mock-launch-btn"
                disabled={starting}
                onClick={() => handleLaunchExam('T3', 50, 60)}
              >
                {starting ? <LoaderCircle size={15} className="spin" /> : <Award size={15} />}
                Start Final Examination
              </button>
            </div>
          </div>

          {/* Past Exam History */}
          <div className="panel" style={{ marginTop: '10px' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '0.95rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Clock3 size={16} color="var(--primary)" /> Past Mock Exam Records ({history.length})
            </h3>
            {history.length === 0 ? (
              <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted)' }}>No mock exam attempts recorded yet for this subject.</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {history.map(item => (
                  <div
                    key={item.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      background: 'var(--pill-bg)',
                      border: '1px solid var(--card-border)',
                      borderRadius: '8px',
                      padding: '10px 14px',
                      fontSize: '0.82rem'
                    }}
                  >
                    <div>
                      <b style={{ color: 'var(--text-main)' }}>{item.title}</b>
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '2px' }}>
                        Attempted: {new Date(item.started_at).toLocaleDateString()} · Duration: {item.duration_min}m
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      {item.is_completed ? (
                        <span style={{ fontWeight: 800, color: item.percentage >= 70 ? '#10b981' : '#f59e0b' }}>
                          Score: {item.score}/{item.total_questions} ({item.percentage}%)
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-dim)', fontStyle: 'italic' }}>Incomplete</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {/* ==================== VIEW 2: ACTIVE EXAM SIMULATION ==================== */}
      {viewState === 'exam' && activeExam && currentQ && (
        <div className="exam-hall-container">
          {/* Main Question Panel */}
          <div className="exam-main-panel">
            {/* Top Bar */}
            <div className="exam-top-bar">
              <div>
                <b style={{ fontSize: '0.95rem', color: '#f8fafc' }}>{activeExam.title}</b>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '2px' }}>
                  Section {currentQ.exam_slot}: {currentQ.exam_slot === 'T1' ? 'Foundation' : currentQ.exam_slot === 'T2' ? 'Intermediate' : 'Advanced'} · Ch. {currentQ.chapter} ({currentQ.chapter_title})
                </div>
              </div>

              <div className={`exam-timer-chip${secondsRemaining < 300 ? ' warning' : ''}`}>
                <Clock3 size={16} />
                <span>{formatTimer(secondsRemaining)}</span>
              </div>
            </div>

            {/* Question Text */}
            <div className="exam-q-header">
              <span className="exam-q-num">Question {currentIdx + 1} of {activeExam.total_questions}</span>
              <button
                style={{
                  background: 'none',
                  border: markedForReview.has(currentQ.id) ? '1px solid #7c3aed' : '1px solid #24304a',
                  color: markedForReview.has(currentQ.id) ? '#d8b4fe' : '#94a3b8',
                  borderRadius: '6px',
                  padding: '4px 10px',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
                onClick={() => toggleMarkForReview(currentQ.id)}
              >
                <Flag size={12} /> {markedForReview.has(currentQ.id) ? 'Marked for Review' : 'Mark for Review'}
              </button>
            </div>

            <div className="exam-q-text">{currentQ.question}</div>

            {/* Options */}
            <div className="exam-options-list">
              {(currentQ.options || []).map((opt, optIdx) => {
                const letter = 'ABCD'[optIdx]
                const isSelected = answers[String(currentQ.id)] === letter
                return (
                  <button
                    key={optIdx}
                    className={`exam-opt-card${isSelected ? ' selected' : ''}`}
                    onClick={() => handleSelectOption(currentQ.id, letter)}
                  >
                    <span className="exam-opt-letter">{letter}</span>
                    <span>{opt}</span>
                  </button>
                )
              })}
            </div>

            {/* Controls Footer */}
            <div className="exam-controls-footer">
              <button
                className="back-nav-btn"
                disabled={currentIdx === 0}
                onClick={() => setCurrentIdx(prev => Math.max(0, prev - 1))}
              >
                <ChevronLeft size={14} /> Previous
              </button>

              <button
                style={{ background: 'none', border: 0, color: '#64748b', fontSize: '0.78rem', cursor: 'pointer' }}
                onClick={() => handleClearAnswer(currentQ.id)}
              >
                Clear Response
              </button>

              {isLastQ ? (
                <button
                  className="exam-submit-btn"
                  onClick={handleSubmitExam}
                  disabled={submitting}
                >
                  {submitting ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}
                  Submit Examination
                </button>
              ) : (
                <button
                  className="primary-btn"
                  style={{ padding: '8px 18px', fontSize: '0.82rem' }}
                  onClick={() => setCurrentIdx(prev => Math.min(activeExam.total_questions - 1, prev + 1))}
                >
                  Next <ChevronRight size={14} />
                </button>
              )}
            </div>
          </div>

          {/* Palette Sidebar */}
          <div className="exam-palette-panel">
            <h4 style={{ margin: 0, fontSize: '0.85rem', color: '#f8fafc' }}>Question Navigation</h4>

            <div className="palette-legend">
              <div className="legend-item"><div className="legend-box answered" /> Answered</div>
              <div className="legend-item"><div className="legend-box marked" /> Review</div>
              <div className="legend-item"><div className="legend-box unanswered" /> Unanswered</div>
            </div>

            <div className="palette-grid">
              {activeExam.questions.map((q, qIndex) => {
                const isAns = Boolean(answers[String(q.id)])
                const isMarked = markedForReview.has(q.id)
                const isCurr = qIndex === currentIdx
                let cls = 'palette-btn'
                if (isCurr) cls += ' current'
                if (isMarked) cls += ' marked'
                else if (isAns) cls += ' answered'
                return (
                  <button
                    key={q.id}
                    className={cls}
                    onClick={() => setCurrentIdx(qIndex)}
                  >
                    {qIndex + 1}
                  </button>
                )
              })}
            </div>

            <div style={{ marginTop: 'auto', paddingTop: '10px', borderTop: '1px solid #1e2b42' }}>
              <div style={{ fontSize: '0.75rem', color: '#8899bb', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>Answered: <b>{Object.keys(answers).length} / {activeExam.total_questions}</b></span>
              </div>
              <button
                className="exam-submit-btn"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={handleSubmitExam}
                disabled={submitting}
              >
                {submitting ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}
                Submit Exam Now
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ==================== VIEW 3: SCORECARD & REVIEW ==================== */}
      {viewState === 'scorecard' && examResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Top Scorecard Banner */}
          <div className="scorecard-banner">
            <div className="scorecard-badge">
              <div className="grade-badge-circle">
                <span>{examResult.grade}</span>
              </div>
              <div className="scorecard-numbers">
                <h2>{examResult.score} / {examResult.total_marks} Marks</h2>
                <p>Overall Percentage: <b style={{ color: '#fbbf24' }}>{examResult.percentage}%</b> · Time Spent: {Math.round(examResult.time_spent_sec / 60)} mins</p>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                className="primary-btn"
                onClick={() => setViewState('launcher')}
              >
                <RotateCcw size={14} /> Back to Catalog
              </button>
            </div>
          </div>

          {/* Tier Performance Breakdown */}
          <div className="panel">
            <h3 style={{ margin: '0 0 12px', fontSize: '0.95rem', color: 'var(--text-main)' }}>Performance by Syllabus Exam Tier</h3>
            <div className="tier-breakdown-row">
              {['T1', 'T2', 'T3'].map(slot => {
                const stat = examResult.accuracy_by_tier?.[slot] || { asked: 0, correct: 0, accuracy: 0 }
                return (
                  <div key={slot} className="tier-stat-card">
                    <div className="tier-stat-head">
                      <span>{slot === 'T1' ? 'T1 Foundation' : slot === 'T2' ? 'T2 Intermediate' : 'T3 Comprehensive'}</span>
                      <span style={{ color: 'var(--primary)' }}>{stat.accuracy}%</span>
                    </div>
                    <div className="tier-bar-bg">
                      <div className={`tier-bar-fill ${slot.toLowerCase()}`} style={{ width: `${stat.accuracy}%` }} />
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      {stat.correct} of {stat.asked} questions correct
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Prerequisite Diagnosis for Missed Topics */}
          {examResult.prerequisite_recommendations && examResult.prerequisite_recommendations.length > 0 && (
            <div className="panel" style={{ border: '1px solid var(--card-border)' }}>
              <h3 style={{ margin: '0 0 8px', fontSize: '0.95rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sparkles size={16} color="var(--primary)" /> Diagnostic Prerequisite Recommendations
              </h3>
              <p style={{ margin: '0 0 12px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Based on missed questions, strengthening these foundational prerequisite chapters will yield the greatest score improvement:
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
                {examResult.prerequisite_recommendations.map(pr => (
                  <div key={pr.chapter} style={{ background: 'var(--pill-bg)', border: '1px solid var(--card-border)', borderRadius: '8px', padding: '10px 12px' }}>
                    <b style={{ fontSize: '0.85rem', color: 'var(--text-main)' }}>Ch. {pr.chapter}: {pr.title}</b>
                    <div style={{ fontSize: '0.72rem', color: 'var(--primary)', marginTop: '3px' }}>
                      Tier: {pr.exam_slot} · Impact: {pr.missed_count} missed question(s) dependent on this
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Interactive Solutions Review */}
          <div className="panel">
            <h3 style={{ margin: '0 0 14px', fontSize: '0.95rem', color: 'var(--text-main)' }}>Comprehensive Solutions &amp; Excerpts Review</h3>
            <div className="review-items-list">
              {examResult.review_items?.map((item, rIdx) => (
                <div key={rIdx} className={`review-item-card ${item.is_correct ? 'correct' : 'wrong'}`}>
                  <div className="review-item-top">
                    <div>
                      <span style={{ fontSize: '0.75rem', fontWeight: 800, color: '#94a3b8' }}>Q{item.position}. </span>
                      <span className={`ad-tier-tag ${item.exam_slot?.toLowerCase() || 't1'}`}>{item.exam_slot}</span>
                      <span style={{ fontSize: '0.75rem', color: '#64748b', marginLeft: '6px' }}>Ch. {item.chapter} ({item.chapter_title})</span>
                    </div>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: item.is_correct ? '#34d399' : '#f87171' }}>
                      {item.is_correct ? '✓ Correct (+1.0)' : '✗ Incorrect (0.0)'}
                    </span>
                  </div>

                  <div className="review-item-q">{item.question}</div>

                  <div className="review-answers-box">
                    <div>Your Answer: <b style={{ color: item.is_correct ? '#34d399' : '#f87171' }}>{item.answer_given || 'Unattempted'}</b></div>
                    <div>Correct Answer: <b style={{ color: '#34d399' }}>{item.correct_answer}</b> {item.correct_answer_text ? `(${item.correct_answer_text})` : ''}</div>
                  </div>

                  <p className="review-explanation">{item.explanation}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
