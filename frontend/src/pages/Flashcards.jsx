import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  BookOpen, Brain, Check, ChevronLeft, ChevronRight, CircleHelp, Clock3,
  Flame, Layers3, LoaderCircle, RotateCcw, Sparkles, ThumbsDown, ThumbsUp,
} from 'lucide-react'
import '../flashcards.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const STUDENT_ID = 'default'

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  })
  const raw = await response.text()
  let data
  try { data = raw ? JSON.parse(raw) : null } catch { data = raw }
  if (!response.ok) {
    const message = data?.detail || data?.message || (typeof data === 'string' ? data : `Request failed (${response.status})`)
    throw new Error(message)
  }
  return data
}

const unwrapCards = (payload) => {
  if (Array.isArray(payload)) return payload
  if (Array.isArray(payload?.flashcards)) return payload.flashcards
  if (Array.isArray(payload?.cards)) return payload.cards
  if (Array.isArray(payload?.items)) return payload.items
  return []
}
const clean = (value) => value == null || value === '' ? '—' : String(value)

export default function Flashcards() {
  const [cards, setCards] = useState([])
  const [dueCards, setDueCards] = useState([])
  const [stats, setStats] = useState({ total: 0, reviewed: 0, due: 0, new: 0 })
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [bookId, setBookId] = useState('CLRS')
  const [chapter, setChapter] = useState('2')
  const [count, setCount] = useState(10)
  const [generating, setGenerating] = useState(false)
  const [studyMode, setStudyMode] = useState('due')
  const [currentIndex, setCurrentIndex] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const [reviewing, setReviewing] = useState(false)
  const [sessionReviewedIds, setSessionReviewedIds] = useState([])
  const [search, setSearch] = useState('')

  const loadData = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setLoading(true)
    else setRefreshing(true)
    setError('')
    try {
      const [allResult, dueResult, statsResult] = await Promise.all([
        api('/flashcards'),
        api(`/flashcards/due?student_id=${encodeURIComponent(STUDENT_ID)}&limit=500`),
        api(`/flashcards/stats?student_id=${encodeURIComponent(STUDENT_ID)}`),
      ])
      setCards(unwrapCards(allResult))
      setDueCards(unwrapCards(dueResult))
      setStats({ total: Number(statsResult?.total ?? 0), reviewed: Number(statsResult?.reviewed ?? 0), due: Number(statsResult?.due ?? 0), new: Number(statsResult?.new ?? 0) })
    } catch (err) {
      setError(`${err.message}. Check that the FastAPI backend is running at ${API_BASE}.`)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const studyCards = (studyMode === 'due' ? dueCards : cards).filter(card => !sessionReviewedIds.includes(String(card.id)))
  const currentCard = studyCards[currentIndex]
  const filteredCards = useMemo(() => cards.filter(card => {
    const text = `${card.front || ''} ${card.back || ''} ${card.book_id || ''} ${card.chapter_title || ''} ${card.chapter || ''}`.toLowerCase()
    return text.includes(search.toLowerCase())
  }), [cards, search])

  const generate = async (event) => {
    event.preventDefault()
    if (!bookId.trim()) { setError('Enter a book ID (for example, CLRS).'); return }
    setGenerating(true); setError(''); setNotice('')
    try {
      const body = { book_id: bookId.trim(), count: Number(count) }
      if (chapter.trim()) body.chapter = chapter.trim()
      const result = await api('/generate/flashcards', { method: 'POST', body: JSON.stringify(body) })
      const savedCount = Number(result?.saved ?? result?.flashcards?.length ?? 0)
      if (savedCount > 0) setNotice(result?.message || `Generated ${savedCount} flashcards.`)
      else setError(result?.message || 'No flashcards were generated. Try another book or chapter.')
      await loadData({ quiet: true })
      setStudyMode('all'); setCurrentIndex(0); setFlipped(false); setSessionReviewedIds([])
    } catch (err) { setError(err.message) }
    finally { setGenerating(false) }
  }

  const review = async (rating) => {
    if (!currentCard?.id) return
    setReviewing(true); setError('')
    try {
      await api(`/flashcards/${encodeURIComponent(currentCard.id)}/review`, {
        method: 'POST', body: JSON.stringify({ student_id: STUDENT_ID, rating }),
      })
      setNotice(rating === 'know' ? 'Marked as known. Review schedule updated.' : 'Marked for another review. Schedule updated.')
      // Hide reviewed cards from this session so Again does not immediately cycle back.
      setSessionReviewedIds(ids => [...ids, String(currentCard.id)])
      await loadData({ quiet: true })
      setFlipped(false)
      setCurrentIndex(0)
    } catch (err) { setError(err.message) }
    finally { setReviewing(false) }
  }

  const step = (delta) => {
    if (!studyCards.length) return
    setCurrentIndex((i) => (i + delta + studyCards.length) % studyCards.length)
    setFlipped(false)
  }

  return <div className="flashcards-page">
    <div className="fc-page-head">
      <div><div className="eyebrow">ACTIVE RECALL · SPACED REPETITION</div><h1>Flashcards</h1><p>Generate cards from your textbooks, then review them to strengthen recall.</p></div>
      <button className="fc-refresh" onClick={() => loadData({ quiet: true })} disabled={refreshing}><RotateCcw size={15}/>{refreshing ? 'Refreshing…' : 'Refresh'}</button>
    </div>

    <div className="fc-stats">
      <StatCard icon={Layers3} label="Total cards" value={stats.total} hint="In your deck" />
      <StatCard icon={Clock3} label="Due for review" value={stats.due} hint="Ready to study" accent="amber" />
      <StatCard icon={Brain} label="Reviewed" value={stats.reviewed} hint="Cards with review history" accent="violet" />
      <StatCard icon={Sparkles} label="New cards" value={stats.new} hint="Not reviewed yet" accent="cyan" />
    </div>

    <div className="fc-main-grid">
      <section className="fc-panel fc-generator">
        <div className="fc-panel-title"><div className="fc-icon"><Sparkles size={18}/></div><div><h2>Generate flashcards</h2><p>Use indexed textbook chunks as the source.</p></div></div>
        <form onSubmit={generate} className="fc-generate-form">
          <label>BOOK ID<input value={bookId} onChange={e => setBookId(e.target.value)} placeholder="e.g. CLRS" /></label>
          <div className="fc-form-row"><label>CHAPTER <span className="optional">(optional)</span><input value={chapter} onChange={e => setChapter(e.target.value)} placeholder="e.g. 2" /></label><label>NUMBER OF CARDS<select value={count} onChange={e => setCount(e.target.value)}>{[5,10,15,20,25,30].map(n=><option key={n} value={n}>{n}</option>)}</select></label></div>
          <button className="primary-btn fc-generate-btn" disabled={generating}>{generating ? <><LoaderCircle className="spin" size={17}/> Generating…</> : <><Sparkles size={17}/> Generate cards</>}</button>
        </form>
        <div className="fc-source-note"><BookOpen size={15}/><span>Cards include textbook source metadata when available.</span></div>
      </section>

      <section className="fc-panel fc-study-panel">
        <div className="fc-study-heading"><div><h2>Study session</h2><p>Review due cards or browse your full deck.</p></div><div className="fc-mode-switch"><button className={studyMode==='due'?'active':''} onClick={()=>{setStudyMode('due');setCurrentIndex(0);setFlipped(false);setSessionReviewedIds([])}}>Due <span>{dueCards.length}</span></button><button className={studyMode==='all'?'active':''} onClick={()=>{setStudyMode('all');setCurrentIndex(0);setFlipped(false);setSessionReviewedIds([])}}>All cards</button></div></div>
        {loading ? <div className="fc-empty"><LoaderCircle className="spin"/><span>Loading your flashcards…</span></div> : !currentCard ? <div className="fc-empty"><div className="fc-empty-icon"><Check size={24}/></div><h3>{studyMode==='due'?'You’re all caught up!':'No flashcards yet'}</h3><p>{studyMode==='due'?'There are no cards due for review right now. Generate cards or switch to All cards.':'Generate your first set using the textbook form.'}</p><button className="text-btn" onClick={()=>studyMode==='due'?setStudyMode('all'):loadData({quiet:true})}>{studyMode==='due'?'Browse all cards':'Refresh deck'}</button></div> : <>
          <div className="fc-card-count">CARD {currentIndex+1} <span>/ {studyCards.length}</span><span className="fc-source-chip">{clean(currentCard.book_id)} · Ch. {clean(currentCard.chapter)}</span></div>
          <button className={`fc-flip-card ${flipped?'flipped':''}`} onClick={()=>setFlipped(v=>!v)} aria-label="Flip flashcard">
            <div className="fc-card-face"><span className="fc-card-label">{flipped?'ANSWER':'QUESTION'}</span><div className="fc-card-text">{flipped?clean(currentCard.back):clean(currentCard.front)}</div><span className="fc-tap-hint"><CircleHelp size={14}/> Click to {flipped?'see question':'reveal answer'}</span></div>
          </button>
          {flipped ? <div className="fc-review-actions"><button className="fc-again" onClick={()=>review('again')} disabled={reviewing}><ThumbsDown size={16}/><span>Again</span><small>Needs practice</small></button><button className="fc-know" onClick={()=>review('know')} disabled={reviewing}><ThumbsUp size={16}/><span>Know</span><small>Got it</small></button></div> : <div className="fc-card-controls"><button onClick={()=>step(-1)} aria-label="Previous card"><ChevronLeft size={17}/></button><button className="fc-reveal" onClick={()=>setFlipped(true)}>Reveal answer</button><button onClick={()=>step(1)} aria-label="Next card"><ChevronRight size={17}/></button></div>}
          <div className="fc-source-details"><span><BookOpen size={13}/> {clean(currentCard.chapter_title || currentCard.chapter)}</span><span>Section {clean(currentCard.section)}</span><span>Page {clean(currentCard.page_start)}</span></div>
        </>}
      </section>
    </div>

    <section className="fc-panel fc-deck-panel">
      <div className="fc-deck-header"><div><h2>Your deck</h2><p>Browse generated cards and their textbook sources.</p></div><div className="fc-deck-search"><input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search cards…"/><span>{filteredCards.length} cards</span></div></div>
      {loading ? <div className="fc-table-empty">Loading cards…</div> : filteredCards.length===0 ? <div className="fc-table-empty">No cards match your search.</div> : <div className="fc-table-wrap"><table className="fc-table"><thead><tr><th>Question</th><th>Book / Chapter</th><th>Section</th><th>Page</th></tr></thead><tbody>{filteredCards.map((card,i)=><tr key={card.id ?? `${card.chunk_id}-${i}`}><td><b>{card.front || '—'}</b><details><summary>Show answer</summary><p>{card.back || '—'}</p></details></td><td>{clean(card.book_id)} / {clean(card.chapter)}</td><td>{clean(card.section)}</td><td>{clean(card.page_start)}</td></tr>)}</tbody></table></div>}
    </section>
    {notice && <div className="fc-toast success"><Check size={15}/>{notice}<button onClick={()=>setNotice('')} aria-label="Dismiss">×</button></div>}
    {error && <div className="fc-toast error"><CircleHelp size={15}/>{error}<button onClick={()=>setError('')} aria-label="Dismiss">×</button></div>}
    <div className="fc-api-note">API: <code>{API_BASE}</code></div>
  </div>
}

function StatCard({icon:Icon,label,value,hint,accent}) { return <div className={`fc-stat ${accent||''}`}><div className="fc-stat-top"><span>{label}</span><i><Icon size={16}/></i></div><b>{value}</b><small>{hint}</small></div> }
