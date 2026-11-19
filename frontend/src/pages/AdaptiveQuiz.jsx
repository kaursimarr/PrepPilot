import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Activity, AlertTriangle, ArrowRight, Award, BookOpen, CheckCircle2,
  ChevronLeft, Clock3, Flag, GraduationCap, Layers, LoaderCircle, Play,
  RefreshCw, RotateCcw, Sparkles, TrendingUp, XCircle, Zap
} from 'lucide-react'
import '../adaptive.css'
import {
  BOOK_TITLES, flagAdaptiveQuestion, getAdaptiveBank, getAdaptiveCurrentQuestion,
  getAdaptiveProgress, getAdaptiveSession, getChapters, getSyllabus, prefillAdaptiveBank,
  startAdaptiveQuiz, submitAdaptiveAnswer
} from '../api'

const STUDENT_ID = 'default'

export default function AdaptiveQuiz({ onBackToWorkspace }) {
  // Configuration state
  const [bookId, setBookId] = useState('CLRS')
  const [chapters, setChapters] = useState([])
  const [selectedChapter, setSelectedChapter] = useState('')
  const [maxQuestions, setMaxQuestions] = useState(10)
  const [syllabusData, setSyllabusData] = useState(null)

  // Bank & Progress states
  const [bankCounts, setBankCounts] = useState({ easy: 0, medium: 0, hard: 0 })
  const [prefillStatus, setPrefillStatus] = useState(null)
  const [isPrefilling, setIsPrefilling] = useState(false)
  const [masteryProgress, setMasteryProgress] = useState([])
  const [loadingConfig, setLoadingConfig] = useState(false)

  // Quiz active session state
  const [quizState, setQuizState] = useState('idle') // 'idle' | 'in_quiz' | 'summary'
  const [sessionId, setSessionId] = useState(null)
  const [sessionAbility, setSessionAbility] = useState(0.5)
  const [sessionLevel, setSessionLevel] = useState('intermediate')
  const [questionNumber, setQuestionNumber] = useState(1)
  const [totalQuestions, setTotalQuestions] = useState(10)
  const [currentQuestion, setCurrentQuestion] = useState(null)
  const [selectedOption, setSelectedOption] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
  const [lastResult, setLastResult] = useState(null)
  const [summaryData, setSummaryData] = useState(null)
  const [flagging, setFlagging] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')
  const [noticeMsg, setNoticeMsg] = useState('')

  // Timer tracking for response time
  const questionStartTimeRef = useRef(Date.now())

  // Load chapters and bank counts when bookId changes
  const loadBookData = useCallback(async (book, chapter) => {
    setLoadingConfig(true)
    setErrorMsg('')
    try {
      const [chList, bank, prog, syl] = await Promise.all([
        getChapters(book).catch(() => []),
        getAdaptiveBank(book, chapter || null).catch(() => ({ counts: { easy: 0, medium: 0, hard: 0 } })),
        getAdaptiveProgress(STUDENT_ID, book).catch(() => []),
        getSyllabus(book).catch(() => null)
      ])
      setChapters(chList || [])
      setBankCounts(bank?.counts || { easy: 0, medium: 0, hard: 0 })
      setPrefillStatus(bank?.prefill || null)
      setMasteryProgress(prog || [])
      setSyllabusData(syl)
    } catch (err) {
      setErrorMsg(err.message || 'Failed to load textbook intelligence data.')
    } finally {
      setLoadingConfig(false)
    }
  }, [])

  useEffect(() => {
    loadBookData(bookId, selectedChapter)
  }, [bookId, selectedChapter, loadBookData])

  // Handle bank prefill
  const handlePrefill = async () => {
    setIsPrefilling(true)
    setErrorMsg('')
    setNoticeMsg('')
    try {
      await prefillAdaptiveBank(bookId, selectedChapter || null, 8)
      setNoticeMsg('Background prefill job started! Populating question bank...')
      setTimeout(() => {
        loadBookData(bookId, selectedChapter)
      }, 2500)
    } catch (err) {
      setErrorMsg(err.message || 'Could not start background prefill.')
    } finally {
      setIsPrefilling(false)
    }
  }

  // Start adaptive assessment
  const handleStartQuiz = async () => {
    setIsStarting(true)
    setErrorMsg('')
    setNoticeMsg('')
    try {
      const res = await startAdaptiveQuiz(STUDENT_ID, bookId, selectedChapter || null, maxQuestions)
      setSessionId(res.session_id)
      setSessionAbility(res.ability)
      setSessionLevel(res.level)
      setQuestionNumber(res.number)
      setTotalQuestions(res.total)
      setCurrentQuestion(res.question)
      setSelectedOption('')
      setLastResult(null)
      questionStartTimeRef.current = Date.now()
      setQuizState('in_quiz')
    } catch (err) {
      setErrorMsg(err.message || 'Failed to start adaptive quiz. Make sure questions exist in the bank.')
    } finally {
      setIsStarting(false)
    }
  }

  // Submit answer
  const handleSubmitAnswer = async () => {
    if (!selectedOption || !currentQuestion || isSubmitting) return
    setIsSubmitting(true)
    setErrorMsg('')
    const timeMs = Date.now() - questionStartTimeRef.current

    try {
      const res = await submitAdaptiveAnswer(sessionId, currentQuestion.id, selectedOption, timeMs)
      setLastResult(res)
      setSessionAbility(res.ability)
      setSessionLevel(res.level)

      if (res.finished) {
        setSummaryData(res.summary)
      }
    } catch (err) {
      setErrorMsg(err.message || 'Error grading response.')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Next Question
  const handleNextQuestion = () => {
    if (lastResult?.finished) {
      setQuizState('summary')
      return
    }
    if (lastResult?.next_question) {
      setCurrentQuestion(lastResult.next_question)
      setQuestionNumber(lastResult.number)
      setSelectedOption('')
      setLastResult(null)
      questionStartTimeRef.current = Date.now()
    }
  }

  // Flag bad/broken question
  const handleFlagQuestion = async () => {
    if (!currentQuestion?.id || flagging) return
    if (!window.confirm('Report this question as inaccurate or broken? It will be removed from the question bank and your ability will be rolled back.')) {
      return
    }
    setFlagging(true)
    try {
      await flagAdaptiveQuestion(currentQuestion.id)
      setNoticeMsg('Question retired and levels rolled back. Fetching replacement question...')
      const refreshed = await getAdaptiveCurrentQuestion(sessionId)
      if (refreshed?.question) {
        setCurrentQuestion(refreshed.question)
        setQuestionNumber(refreshed.number)
        setSessionAbility(refreshed.ability)
        setSessionLevel(refreshed.level)
        setSelectedOption('')
        setLastResult(null)
        questionStartTimeRef.current = Date.now()
      } else {
        setErrorMsg('No more questions available for this session.')
        setQuizState('idle')
      }
    } catch (err) {
      setErrorMsg(err.message || 'Failed to flag question.')
    } finally {
      setFlagging(false)
    }
  }

  // Exit Quiz
  const handleExitQuiz = () => {
    if (quizState === 'in_quiz' && !window.confirm('Are you sure you want to exit? Your progress so far is saved.')) {
      return
    }
    setQuizState('idle')
    setLastResult(null)
    setCurrentQuestion(null)
    loadBookData(bookId, selectedChapter)
  }

  // Prerequisite analysis for the currently selected chapter / scope
  const currentChapterObj = (() => {
    if (!selectedChapter || selectedChapter.startsWith('exam:')) return null
    const allSylChaps = [
      ...(syllabusData?.T1?.chapters || []),
      ...(syllabusData?.T2?.chapters || []),
      ...(syllabusData?.T3?.new_chapters || syllabusData?.T3?.chapters || []),
      ...(syllabusData?.unassigned?.chapters || [])
    ]
    return allSylChaps.find(c => String(c.chapter) === String(selectedChapter))
  })()

  const chapterPrereqs = currentChapterObj?.prerequisites || []

  const prereqReadiness = chapterPrereqs.map(p => {
    const m = masteryProgress.find(item => String(item.chapter) === String(p.chapter))
    const thetaVal = m?.theta != null ? Math.round(m.theta * 100) / 100 : null
    const isMastered = (m?.theta ?? 0) >= 0.5 || m?.level === 'intermediate' || m?.level === 'advanced'
    return {
      ...p,
      isMastered,
      theta: thetaVal,
      level: m?.level ?? 'unassessed'
    }
  })

  const unmasteredPrereqs = prereqReadiness.filter(p => !p.isMastered)

  // Exam scope prerequisites
  const scopePrereqInfo = (() => {
    if (selectedChapter === 'exam:T2') return syllabusData?.exam_prerequisites?.T2
    if (selectedChapter === 'exam:T3') return syllabusData?.exam_prerequisites?.T3
    if (selectedChapter === 'exam:T1+T2') return {
      title: 'Mid-term Combined Scope',
      requirement: 'Cumulative evaluation: T1 Foundations + T2 Intermediate',
      description: 'Tests cumulative understanding across both Test 1 and Test 2 topics.',
      prereq_scope: 'Test 1 (T1) Foundation Scope'
    }
    return null
  })()

  return (
    <div className="adaptive-page">
      {/* Top Header */}
      <div className="ad-page-head">
        <div>
          <div className="eyebrow" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Zap size={14} color="#7b8dff" /> ITEM RESPONSE THEORY · ADAPTIVE TESTING
          </div>
          <h1>Adaptive Quiz Assessment</h1>
          <p>
            Dynamically measures your competence level. Each question is automatically chosen to match your current estimated ability.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {quizState !== 'idle' ? (
            <button className="back-nav-btn" onClick={handleExitQuiz}>
              <ChevronLeft size={14} /> Exit Assessment
            </button>
          ) : (
            onBackToWorkspace && (
              <button className="back-nav-btn" onClick={onBackToWorkspace}>
                <ChevronLeft size={14} /> Back to Overview
              </button>
            )
          )}
        </div>
      </div>

      {noticeMsg && (
        <div style={{ padding: '10px 14px', background: 'rgba(16, 185, 129, 0.12)', border: '1px solid rgba(16, 185, 129, 0.35)', borderRadius: '8px', color: '#10b981', fontSize: '11px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>✓ {noticeMsg}</span>
          <button onClick={() => setNoticeMsg('')} style={{ background: 'none', border: 0, color: '#10b981', cursor: 'pointer' }}>✕</button>
        </div>
      )}

      {errorMsg && (
        <div style={{ padding: '10px 14px', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.35)', borderRadius: '8px', color: '#ef4444', fontSize: '11px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>⚠ {errorMsg}</span>
          <button onClick={() => setErrorMsg('')} style={{ background: 'none', border: 0, color: '#ef4444', cursor: 'pointer' }}>✕</button>
        </div>
      )}

      {/* ===================== VIEW 1: PRE-QUIZ SETUP & MASTERY ===================== */}
      {quizState === 'idle' && (
        <div className="ad-grid">
          {/* Setup Panel */}
          <section className="ad-panel">
            <h2>Configure Adaptive Session</h2>
            <p className="ad-panel-sub">Select your study target. The engine will match question difficulty to your skill curve.</p>

            <div className="ad-form">
              <div className="ad-field">
                <label>Target Textbook</label>
                <select value={bookId} onChange={e => { setBookId(e.target.value); setSelectedChapter('') }}>
                  <option value="CLRS">Introduction to Algorithms (CLRS)</option>
                  <option value="DS">Handbook of Data Structures (DS)</option>
                  <option value="C">Let Us C (C)</option>
                  <option value="CPP">C++ Complete Reference (CPP)</option>
                </select>
              </div>

              <div className="ad-field">
                <label>Assessment Scope / Exam Focus</label>
                <select value={selectedChapter} onChange={e => setSelectedChapter(e.target.value)}>
                  <option value="">Whole Course (Comprehensive Assessment)</option>
                  <optgroup label="Exam Tiers (Syllabus Intelligence)">
                    <option value="exam:T1">Test 1 Scope (T1 Foundations — 33%)</option>
                    <option value="exam:T2">Test 2 Scope (T2 Intermediate — 33% | Requires T1)</option>
                    <option value="exam:T1+T2">T1 + T2 Combined Scope (Mid-term — 66%)</option>
                    <option value="exam:T3">Test 3 Scope (T3 Comprehensive — 100%)</option>
                  </optgroup>
                  <optgroup label="Individual Chapters">
                    {chapters.map(c => {
                      const sylChap = [
                        ...(syllabusData?.T1?.chapters || []),
                        ...(syllabusData?.T2?.chapters || []),
                        ...(syllabusData?.T3?.new_chapters || syllabusData?.T3?.chapters || []),
                      ].find(x => String(x.chapter) === String(c.chapter))
                      const slot = sylChap?.exam_slot ? `[${sylChap.exam_slot}] ` : ''
                      return (
                        <option key={c.chapter} value={c.chapter}>
                          {slot}Ch. {c.chapter}: {c.chapter_title?.slice(0, 32)}...
                        </option>
                      )
                    })}
                  </optgroup>
                </select>
              </div>

              {/* Prerequisite Readiness Assessment for Selected Chapter */}
              {chapterPrereqs.length > 0 && (
                <div className="ad-prereq-card">
                  <div className="ad-prereq-head">
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <GraduationCap size={14} color="#70a3ff" /> Prerequisite Readiness Check
                    </span>
                    <span style={{ fontSize: '10px', color: unmasteredPrereqs.length === 0 ? '#6ee7b7' : '#f59e0b' }}>
                      {unmasteredPrereqs.length === 0 ? '✓ Prerequisites Mastered' : `⚠ ${unmasteredPrereqs.length} Prerequisite(s) Incomplete`}
                    </span>
                  </div>

                  <div className="ad-prereq-items">
                    {prereqReadiness.map(p => (
                      <div key={p.chapter} className={`ad-prereq-row ${p.isMastered ? 'ready' : 'warning'}`}>
                        <span>
                          <b>Ch. {p.chapter}</b>: {p.title} <span className={`ad-tier-tag ${p.exam_slot?.toLowerCase() || 't1'}`}>{p.exam_slot || 'T1'}</span>
                        </span>
                        <span>
                          {p.isMastered ? (
                            <span style={{ color: '#4ade80' }}>✓ Mastered (θ {p.theta})</span>
                          ) : (
                            <span style={{ color: '#fbbf24' }}>⚠ Needs Practice ({p.level})</span>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>

                  {unmasteredPrereqs.length > 0 && (
                    <div className="ad-prereq-warning-banner">
                      <div>
                        <b>Prerequisite Recommendation:</b> This chapter builds directly on <b>Ch. {unmasteredPrereqs[0].chapter} ({unmasteredPrereqs[0].title})</b>. Practicing it first will help improve your assessment score.
                      </div>
                      <button
                        type="button"
                        className="ad-prereq-switch-btn"
                        onClick={() => setSelectedChapter(String(unmasteredPrereqs[0].chapter))}
                      >
                        Practice Prerequisite Ch. {unmasteredPrereqs[0].chapter} First →
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Scope-Level Prerequisite Notice */}
              {scopePrereqInfo && (
                <div className="ad-prereq-card">
                  <div className="ad-prereq-head">
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <GraduationCap size={14} color="#70a3ff" /> {scopePrereqInfo.title}
                    </span>
                    <span style={{ fontSize: '10px', color: '#93c5fd' }}>
                      Prerequisite: {scopePrereqInfo.prereq_scope}
                    </span>
                  </div>
                  <p style={{ margin: 0, fontSize: '11px', color: '#8fa0bd', lineHeight: 1.4 }}>
                    {scopePrereqInfo.description}
                  </p>
                </div>
              )}

              <div className="ad-field">
                <label>Number of Questions</label>
                <select value={maxQuestions} onChange={e => setMaxQuestions(Number(e.target.value))}>
                  <option value={5}>5 Questions (Quick Check)</option>
                  <option value={10}>10 Questions (Standard Assessment)</option>
                  <option value={15}>15 Questions (Deep Evaluation)</option>
                  <option value={20}>20 Questions (Mastery Benchmark)</option>
                </select>
              </div>

              {/* Question Bank Status */}
              <div className="ad-bank-status">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '10px', color: '#8898b3', fontWeight: 700, textTransform: 'uppercase' }}>
                    Available Verified Bank
                  </span>
                  <button
                    onClick={handlePrefill}
                    disabled={isPrefilling}
                    style={{ background: 'none', border: 0, color: '#7a8cff', fontSize: '10px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    <Sparkles size={12} /> {isPrefilling ? 'Generating...' : 'Fill bank in background'}
                  </button>
                </div>
                <div className="ad-bank-counts">
                  <span className="ad-bank-pill easy">Easy: {bankCounts.easy}</span>
                  <span className="ad-bank-pill medium">Medium: {bankCounts.medium}</span>
                  <span className="ad-bank-pill hard">Hard: {bankCounts.hard}</span>
                </div>
              </div>

              <button
                className="primary-btn"
                style={{ padding: '13px', marginTop: '10px', fontSize: '12px' }}
                disabled={isStarting}
                onClick={handleStartQuiz}
              >
                {isStarting ? (
                  <><LoaderCircle size={16} className="spin" /> Initializing Adaptive Assessment...</>
                ) : (
                  <><Play size={16} /> Start Adaptive Assessment</>
                )}
              </button>
            </div>
          </section>

          {/* Student Mastery Progress Panel */}
          <section className="ad-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <h2>Your Skill Profile</h2>
              <button
                onClick={() => loadBookData(bookId, selectedChapter)}
                disabled={loadingConfig}
                style={{ background: 'none', border: 0, color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px' }}
              >
                <RotateCcw size={12} /> Refresh
              </button>
            </div>
            <p className="ad-panel-sub">Chapter mastery levels for {BOOK_TITLES[bookId] || bookId}, weakest topics first.</p>

            {masteryProgress.length === 0 ? (
              <div style={{ padding: '24px 16px', textAlign: 'center', color: 'var(--text-muted)', background: 'var(--pill-bg)', border: '1px solid var(--card-border)', borderRadius: '8px', fontSize: '11px' }}>
                <TrendingUp size={24} style={{ margin: '0 auto 8px', opacity: 0.7 }} />
                <p style={{ margin: 0 }}>No adaptive attempts recorded for this book yet.</p>
                <small>Take your first adaptive quiz to establish your baseline ability rating.</small>
              </div>
            ) : (
              <div style={{ display: 'grid', gap: '8px', maxHeight: '340px', overflowY: 'auto', paddingRight: '4px' }}>
                {masteryProgress.map((m, idx) => {
                  const levelClass = m.level || 'intermediate'
                  return (
                    <div
                      key={idx}
                      style={{
                        padding: '10px 12px',
                        background: 'var(--card-bg)',
                        border: '1px solid var(--card-border)',
                        borderRadius: '8px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <div>
                        <b style={{ fontSize: '11px', color: 'var(--text-main)', display: 'block' }}>
                          Ch. {m.chapter}: {m.chapter_title || 'Topic'}
                        </b>
                        <small style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
                          {m.correct}/{m.attempts} correct ({m.attempts > 0 ? Math.round((m.correct / m.attempts) * 100) : 0}%)
                        </small>
                      </div>

                      <div style={{ textAlign: 'right' }}>
                        <span className={`ad-badge-theta ${levelClass}`}>
                          {m.level} · θ {m.theta}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </section>
        </div>
      )}

      {/* ===================== VIEW 2: ACTIVE ADAPTIVE QUESTION ===================== */}
      {quizState === 'in_quiz' && currentQuestion && (
        <div className="ad-quiz-box">
          {/* Top metadata row */}
          <div className="ad-quiz-meta">
            <div className="ad-meta-left">
              <span className="subject-tag">
                {currentQuestion.book_id} · CH. {currentQuestion.chapter || '1'}
              </span>
              {currentQuestion.exam_slot && (
                <span className={`ad-tier-tag ${currentQuestion.exam_slot.toLowerCase()}`}>
                  {currentQuestion.exam_slot}
                </span>
              )}
              {currentQuestion.prerequisites && currentQuestion.prerequisites.length > 0 && (
                <span className="ad-prereq-tag">
                  <GraduationCap size={12} /> Prereq: Ch. {currentQuestion.prerequisites.map(p => p.chapter).join(', ')}
                </span>
              )}
              <span className={`ad-difficulty-tag ${currentQuestion.difficulty < 0.4 ? 'easy' : currentQuestion.difficulty < 0.65 ? 'medium' : 'hard'}`}>
                {currentQuestion.difficulty < 0.4 ? 'Easy' : currentQuestion.difficulty < 0.65 ? 'Medium' : 'Hard'} (diff: {currentQuestion.difficulty})
              </span>
              <span className={`ad-badge-theta ${sessionLevel}`}>
                Current Ability: θ {sessionAbility} ({sessionLevel})
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className="time-chip">
                <Clock3 size={13} /> Question {questionNumber} of {totalQuestions}
              </span>
              <button
                className="ad-flag-btn"
                onClick={handleFlagQuestion}
                disabled={flagging}
                title="Report flawed question"
              >
                <Flag size={12} /> {flagging ? 'Reporting...' : 'Flag'}
              </button>
            </div>
          </div>

          {/* Question Text */}
          <h2 className="ad-question-text">{currentQuestion.question}</h2>

          {/* Options */}
          <div className="ad-options-list">
            {(currentQuestion.options || []).map((opt, oIdx) => {
              const letter = 'ABCD'[oIdx]
              const isSelected = selectedOption === letter
              let cardClass = isSelected ? 'selected' : ''

              if (lastResult) {
                if (letter === lastResult.correct_answer) {
                  cardClass = 'correct'
                } else if (isSelected && !lastResult.correct) {
                  cardClass = 'wrong'
                }
              }

              return (
                <button
                  key={oIdx}
                  className={`ad-option-card ${cardClass}`}
                  disabled={Boolean(lastResult)}
                  onClick={() => setSelectedOption(letter)}
                >
                  <span className="ad-opt-letter">{letter}</span>
                  <span style={{ marginTop: '2px' }}>{opt}</span>
                </button>
              )
            })}
          </div>

          {/* Immediate Feedback Box */}
          {lastResult && (
            <div className={`ad-feedback ${lastResult.correct ? 'correct' : 'wrong'}`}>
              <div className="ad-feedback-head">
                <span>
                  {lastResult.correct ? '✓ Correct Answer!' : `✗ Incorrect (Correct is ${lastResult.correct_answer})`}
                </span>
                <span className="ad-theta-delta">
                  <Activity size={12} />
                  Ability: θ {lastResult.ability} ({lastResult.level})
                </span>
              </div>
              <p className="ad-explanation">{lastResult.explanation}</p>
              {lastResult.prerequisite_tip && (
                <div className="ad-prereq-remedy">
                  <Sparkles size={14} style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>{lastResult.prerequisite_tip}</span>
                </div>
              )}
              {currentQuestion.chapter_title && (
                <small style={{ display: 'block', marginTop: '6px', opacity: 0.85 }}>
                  <BookOpen size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />
                  Source: {currentQuestion.chapter_title} (Page {currentQuestion.page_start || '—'})
                </small>
              )}
            </div>
          )}

          {/* Actions */}
          <div style={{ marginTop: '22px', display: 'flex', gap: '10px' }}>
            {!lastResult ? (
              <button
                className="primary-btn"
                style={{ width: '100%', padding: '12px' }}
                disabled={!selectedOption || isSubmitting}
                onClick={handleSubmitAnswer}
              >
                {isSubmitting ? (
                  <><LoaderCircle size={15} className="spin" /> Grading &amp; Calibrating...</>
                ) : (
                  <>Submit Answer</>
                )}
              </button>
            ) : (
              <button
                className="primary-btn"
                style={{ width: '100%', padding: '12px' }}
                onClick={handleNextQuestion}
              >
                {lastResult.finished ? 'Finish Assessment & View Results →' : 'Next Adaptive Question →'}
              </button>
            )}
          </div>
        </div>
      )}

      {/* ===================== VIEW 3: SUMMARY & RECOMMENDATIONS ===================== */}
      {quizState === 'summary' && summaryData && (
        <div className="ad-summary">
          <div className="ad-summary-circle">
            <div>
              <b>{summaryData.ability}</b>
              <span>θ ABILITY</span>
            </div>
          </div>

          <h2 style={{ fontSize: '22px', margin: '4px 0 6px' }}>Assessment Complete</h2>
          <p style={{ color: '#8898b3', fontSize: '12px', margin: 0 }}>
            Performance level calibrated to: <b style={{ color: '#aebaff', textTransform: 'capitalize' }}>{summaryData.level}</b>
          </p>

          <div className="ad-summary-stats">
            <div className="ad-stat-box">
              <span>ACCURACY</span>
              <b>{summaryData.answered > 0 ? Math.round((summaryData.correct / summaryData.answered) * 100) : 0}%</b>
              <small style={{ color: '#7f90ad', fontSize: '10px' }}>{summaryData.correct} of {summaryData.answered} correct</small>
            </div>

            <div className="ad-stat-box">
              <span>FINAL RATING</span>
              <b>θ {summaryData.ability}</b>
              <small style={{ color: '#7f90ad', fontSize: '10px' }}>Level: {summaryData.level}</small>
            </div>

            <div className="ad-stat-box">
              <span>CHAPTERS TESTED</span>
              <b>{(summaryData.by_chapter || []).length}</b>
              <small style={{ color: '#7f90ad', fontSize: '10px' }}>Course topics evaluated</small>
            </div>
          </div>

          {/* Ability Progression Trend */}
          {summaryData.ability_trend && summaryData.ability_trend.length > 0 && (
            <div className="ad-trend-box">
              <h3><TrendingUp size={14} style={{ display: 'inline', marginRight: '6px' }} /> Ability Trajectory Over Time</h3>
              <p style={{ fontSize: '10px', color: '#7e8fa9', margin: '0 0 10px' }}>
                Shows how each question dynamically adjusted your estimated competence (θ).
              </p>
              <div className="ad-trend-dots">
                {summaryData.ability_trend.map((val, tIdx) => {
                  const heightPct = Math.max(15, Math.min(100, Math.round(val * 100)))
                  return (
                    <div key={tIdx} className="ad-trend-col">
                      <span style={{ fontSize: '9px', color: '#8fa0bd' }}>{val}</span>
                      <div className="ad-trend-bar" style={{ height: `${heightPct}%` }} />
                      <span className="ad-trend-label">Q{tIdx + 1}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Revise These Topics Recommendation */}
          {summaryData.revise_these && summaryData.revise_these.length > 0 && (
            <div className="ad-revise-box">
              <h3><AlertTriangle size={15} /> Weak Areas &amp; Prerequisite Recommendations</h3>
              <div style={{ display: 'grid', gap: '8px' }}>
                {summaryData.revise_these.map((w, wIdx) => {
                  const sylChap = [
                    ...(syllabusData?.T1?.chapters || []),
                    ...(syllabusData?.T2?.chapters || []),
                    ...(syllabusData?.T3?.new_chapters || syllabusData?.T3?.chapters || []),
                  ].find(x => String(x.chapter) === String(w.chapter))
                  const prereqs = sylChap?.prerequisites || []
                  return (
                    <div key={wIdx} className="ad-revise-item" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <span>
                          <b>Ch. {w.chapter}</b>: {w.chapter_title || 'Topic'}
                        </span>
                        <span style={{ color: '#ffa2b7' }}>
                          Accuracy: {Math.round((w.correct / w.asked) * 100)}% ({w.correct}/{w.asked})
                        </span>
                      </div>
                      {prereqs.length > 0 && (
                        <div style={{ color: '#93c5fd', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <GraduationCap size={12} />
                          <span>Foundational prerequisite to review: <b>{prereqs.map(p => `Ch. ${p.chapter} (${p.title})`).join(', ')}</b></span>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Actions */}
          <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginTop: '24px', flexWrap: 'wrap' }}>
            <button
              className="primary-btn"
              onClick={handleStartQuiz}
            >
              <RefreshCw size={14} /> Retake Assessment
            </button>
            <button
              className="back-nav-btn"
              onClick={handleExitQuiz}
            >
              <RotateCcw size={14} /> Back to Setup &amp; Chapters
            </button>
            {onBackToWorkspace && (
              <button
                className="back-nav-btn"
                onClick={onBackToWorkspace}
              >
                Return to Workspace Overview
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
