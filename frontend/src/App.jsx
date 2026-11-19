import { useEffect, useState } from 'react'
import Flashcards from './pages/Flashcards'
import {
  LayoutDashboard, BookOpen, Bot, ClipboardList, ChartNoAxesCombined, Library,
  Layers3, Settings, Search, Bell, ChevronDown, Clock3, Flame, Target,
  ArrowUpRight, Play, Send, Sparkles, CheckCircle2, Menu, X,
  LoaderCircle, RefreshCw, ChevronLeft, BookMarked, Save, Cpu
} from 'lucide-react'
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, BarChart, Bar, CartesianGrid } from 'recharts'
import {
  getHealth, askQuestion, getChapters, getQuestions, generateQuestions,
  getSubjects, getSubjectDetails, getOverviewStats, getProgressMastery, recordQuizAttempt,
  getQuizHistory, getStudyResources, getResourceFigures, getStudentProfile, updateStudentProfile,
  getSystemSettings, BOOK_TITLES
} from './api'

const fallbackSubjects = [
  { code: 'CLRS', name: 'Design & Analysis of Algorithms', book_title: 'Introduction to Algorithms (CLRS)', progress: 75, color: 'blue', next: 'Red-Black Trees & Dynamic Programming', chapters: '35 chapters', chunks: 4077 },
  { code: 'DS', name: 'Advanced Data Structures', book_title: 'Handbook of Data Structures and Applications', progress: 68, color: 'violet', next: 'Self-Balancing Trees & Heaps', chapters: '64 chapters', chunks: 5651 },
  { code: 'C', name: 'C Programming & Systems', book_title: 'Let Us C (Yashavant Kanetkar)', progress: 88, color: 'cyan', next: 'Pointers, Memory & Structures', chapters: '19 chapters', chunks: 955 },
  { code: 'CPP', name: 'Object-Oriented Software Design (C++)', book_title: 'C++: The Complete Reference', progress: 62, color: 'orange', next: 'Runtime Polymorphism & STL', chapters: '39 chapters', chunks: 2184 },
]

const fallbackActivity = [
  { day: 'Mon', score: 45 }, { day: 'Tue', score: 62 }, { day: 'Wed', score: 55 },
  { day: 'Thu', score: 78 }, { day: 'Fri', score: 68 }, { day: 'Sat', score: 88 }, { day: 'Sun', score: 82 }
]

const defaultQuizQuestions = [
  {
    id: 'starter-1',
    book_id: 'CLRS',
    chapter: '12',
    chapter_title: 'Binary Search Trees',
    question: 'What is the worst-case running time of a SEARCH operation in an unbalanced binary search tree with n nodes?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    answer: 'C',
    answer_text: 'O(n)',
    explanation: 'In the worst case, a binary search tree can degenerate into a linear chain of n nodes (like a singly linked list), giving height Θ(n) and search time O(n). In a balanced tree (like red-black), it is guaranteed O(log n).',
    difficulty: 0.5,
  },
  {
    id: 'starter-2',
    book_id: 'CLRS',
    chapter: '13',
    chapter_title: 'Red-Black Trees',
    question: 'Which of the following properties must every red-black tree satisfy regarding red nodes?',
    options: [
      'The root node must always be red',
      'If a node is red, both its children must be black',
      'All leaf nodes (NIL) must be red',
      'A red node cannot have sibling nodes'
    ],
    answer: 'B',
    answer_text: 'If a node is red, both its children must be black',
    explanation: 'Property 4 of red-black trees states: If a node is red, then both its children are black. This implies that no two red nodes can appear consecutively on any path from root to leaf.',
    difficulty: 0.5,
  },
  {
    id: 'starter-3',
    book_id: 'DS',
    chapter: '1',
    chapter_title: 'Data Structures Foundations',
    question: 'Which data structure is fundamentally utilized by operating systems and language runtimes to implement recursion and function call frames?',
    options: ['Queue', 'Stack', 'Heap', 'Hash Table'],
    answer: 'B',
    answer_text: 'Stack',
    explanation: 'The Last-In-First-Out (LIFO) property of a stack makes it ideal for saving activation frames, local variables, and return addresses during nested and recursive function calls.',
    difficulty: 0.3,
  },
  {
    id: 'starter-4',
    book_id: 'CLRS',
    chapter: '6',
    chapter_title: 'Heapsort',
    question: 'What is the asymptotic time complexity of building a max-heap of n elements using BUILD-MAX-HEAP?',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(log n)'],
    answer: 'A',
    answer_text: 'O(n)',
    explanation: 'By bounding the summation of node heights across all levels of the tree, BUILD-MAX-HEAP runs in linear O(n) time, even though repeated individual insertions take O(n log n).',
    difficulty: 0.6,
  },
  {
    id: 'starter-5',
    book_id: 'CPP',
    chapter: '12',
    chapter_title: 'Polymorphism & OOP',
    question: 'In C++, which keyword is used in a base class method declaration to achieve runtime polymorphism and dynamic dispatch?',
    options: ['static', 'virtual', 'friend', 'inline'],
    answer: 'B',
    answer_text: 'virtual',
    explanation: 'Declaring a member function with the `virtual` keyword instructs the compiler to use late binding via a virtual method table (vtable), enabling derived classes to override it dynamically at runtime.',
    difficulty: 0.4,
  }
]

// Route slug mapping
const ROUTE_MAP = {
  'workspace': 'Overview',
  'overview': 'Overview',
  'subjects': 'My Subjects',
  'assistant': 'AI Study Assistant',
  'practice': 'Practice & Quizzes',
  'flashcards': 'Flashcards',
  'progress': 'My Progress',
  'resources': 'Study Resources',
  'settings': 'Settings',
}

const PAGE_TO_ROUTE = {
  'Overview': 'workspace',
  'My Subjects': 'subjects',
  'AI Study Assistant': 'assistant',
  'Practice & Quizzes': 'practice',
  'Flashcards': 'flashcards',
  'My Progress': 'progress',
  'Study Resources': 'resources',
  'Settings': 'settings',
}

function getInitialPage() {
  const raw = (window.location.hash || window.location.pathname)
    .replace(/^[#/]+/, '')
    .toLowerCase()
    .split('/')[0]
  return ROUTE_MAP[raw] || 'Overview'
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('preppilot-theme') || 'midnight')
  useEffect(() => { localStorage.setItem('preppilot-theme', theme) }, [theme])

  const [page, setPage] = useState(getInitialPage)
  const [mobile, setMobile] = useState(false)
  const [query, setQuery] = useState('')

  // 1. Backend Health & Overview State
  const [health, setHealth] = useState({ status: 'connecting', chunks: 12867, keyword_search: true })
  const [overviewStats, setOverviewStats] = useState({
    chunks_indexed: 12867,
    total_quizzes: 18,
    quiz_accuracy: 74,
    topics_mastered: 29,
    total_topics: 45,
    overall_progress: 74,
    flashcards_total: 0,
    flashcards_due: 0,
    study_streak_days: 4,
    study_time_hours: 12.5,
    focus_areas: [
      { name: 'Dynamic Programming', score: 43, color: 'violet' },
      { name: 'Graph Algorithms', score: 52, color: 'blue' },
      { name: 'Red-Black Trees', score: 68, color: 'cyan' },
    ],
    activity: fallbackActivity,
  })

  // 2. My Subjects State
  const [subjectsList, setSubjectsList] = useState(fallbackSubjects)
  const [selectedSubjectDetail, setSelectedSubjectDetail] = useState(null)
  const [loadingSubjectDetail, setLoadingSubjectDetail] = useState(false)

  // 3. My Progress State
  const [progressMastery, setProgressMastery] = useState([])
  const [quizHistory, setQuizHistory] = useState([])

  // 4. Study Resources State
  const [studyResources, setStudyResources] = useState(null)
  const [resourceFigures, setResourceFigures] = useState([])

  // 5. Settings & Profile State
  const [profile, setProfile] = useState({
    username: 'default',
    name: 'Student Account',
    institution: 'JIIT',
    semester: 'Semester 7',
    preferred_model: 'qwen2.5:7b',
  })
  const [profileSavedMsg, setProfileSavedMsg] = useState('')
  const [systemSettings, setSystemSettings] = useState(null)

  // Initial Data Fetching from Backend Endpoints
  useEffect(() => {
    // Health & System
    getHealth().then(setHealth).catch(() => setHealth({ status: 'offline', chunks: 12867, keyword_search: false }))
    getSystemSettings().then(setSystemSettings).catch(() => {})

    // Overview Stats
    getOverviewStats().then(setOverviewStats).catch(() => {})

    // Subjects
    getSubjects().then(res => { if (res && res.length) setSubjectsList(res) }).catch(() => {})

    // Progress
    getProgressMastery().then(setProgressMastery).catch(() => {})
    getQuizHistory().then(setQuizHistory).catch(() => {})

    // Resources
    getStudyResources().then(setStudyResources).catch(() => {})
    getResourceFigures(null, 8).then(setResourceFigures).catch(() => {})

    // Profile
    getStudentProfile().then(setProfile).catch(() => {})
  }, [])

  // AI Assistant Chat State
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: 'Hi! I’m your PrepPilot study assistant, grounded in your 4 computer science textbooks. Ask me any concept, algorithm, or C/C++ question!'
    }
  ])
  const [answer, setAnswer] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const [selectedBook, setSelectedBook] = useState('ALL')
  const [expandedSources, setExpandedSources] = useState({})

  // Chapter lists for books
  const [bookChapters, setBookChapters] = useState([])
  useEffect(() => {
    if (selectedBook && selectedBook !== 'ALL') {
      getChapters(selectedBook)
        .then(res => setBookChapters(res || []))
        .catch(() => setBookChapters([]))
    } else {
      setBookChapters([])
    }
  }, [selectedBook])

  // Practice & Quizzes State
  const [quizBook, setQuizBook] = useState('CLRS')
  const [quizChapter, setQuizChapter] = useState('')
  const [quizChaptersList, setQuizChaptersList] = useState([])
  const [quizQuestions, setQuizQuestions] = useState(defaultQuizQuestions)
  const [quizStarted, setQuizStarted] = useState(() => {
    const raw = (window.location.hash || '').toLowerCase()
    return raw.includes('quiz')
  })
  const [quizIndex, setQuizIndex] = useState(0)
  const [quizChoice, setQuizChoice] = useState('')
  const [quizSubmitted, setQuizSubmitted] = useState(false)
  const [quizScore, setQuizScore] = useState(0)
  const [quizCompleted, setQuizCompleted] = useState(false)
  const [isGeneratingQuiz, setIsGeneratingQuiz] = useState(false)
  const [quizGenMsg, setQuizGenMsg] = useState('')

  // Routing navigation helper that syncs with browser history
  const go = (targetPage, pushState = true) => {
    setPage(targetPage)
    setMobile(false)
    if (targetPage !== 'Practice & Quizzes') {
      setQuizStarted(false)
      setQuizCompleted(false)
    }
    const routeSlug = PAGE_TO_ROUTE[targetPage] || 'workspace'
    const newHash = `#/${routeSlug}`
    if (pushState && window.location.hash !== newHash) {
      window.history.pushState({ page: targetPage }, '', newHash)
    }
  }

  // Handle browser back and forward buttons
  useEffect(() => {
    const onLocationChange = () => {
      const raw = (window.location.hash || window.location.pathname)
        .replace(/^[#/]+/, '')
        .toLowerCase()
      const routePart = raw.split('/')[0]
      const target = ROUTE_MAP[routePart] || 'Overview'
      setPage(target)
      if (raw.includes('quiz')) {
        setQuizStarted(true)
      } else if (target !== 'Practice & Quizzes') {
        setQuizStarted(false)
      }
    }

    if (!window.location.hash) {
      window.history.replaceState({ page: 'Overview' }, '', '#/workspace')
    }

    window.addEventListener('popstate', onLocationChange)
    window.addEventListener('hashchange', onLocationChange)
    return () => {
      window.removeEventListener('popstate', onLocationChange)
      window.removeEventListener('hashchange', onLocationChange)
    }
  }, [])

  // Load chapters for quiz book
  useEffect(() => {
    if (quizBook) {
      getChapters(quizBook)
        .then(res => {
          setQuizChaptersList(res || [])
          if (res && res.length > 0) {
            setQuizChapter(res[0].chapter)
          }
        })
        .catch(() => setQuizChaptersList([]))
    }
  }, [quizBook])

  // Navigation items
  const nav = [
    ['Overview', LayoutDashboard],
    ['My Subjects', BookOpen],
    ['AI Study Assistant', Bot],
    ['Practice & Quizzes', ClipboardList],
    ['Flashcards', Layers3],
    ['My Progress', ChartNoAxesCombined],
    ['Study Resources', Library]
  ]

  // Send message to AI Assistant (/ask endpoint)
  const send = async () => {
    const q = answer.trim()
    if (!q || isAsking) return
    setAnswer('')
    setMessages(m => [...m, { role: 'user', text: q }])
    setIsAsking(true)

    try {
      const bookTarget = selectedBook === 'ALL' ? null : selectedBook
      const res = await askQuestion(q, bookTarget, 4)
      setMessages(m => [
        ...m,
        {
          role: 'assistant',
          text: res.answer,
          sources: res.sources || [],
          cited: res.cited || [],
          fallback: res.fallback
        }
      ])
    } catch (err) {
      setMessages(m => [
        ...m,
        {
          role: 'assistant',
          text: `Error contacting PrepPilot backend: ${err.message}. Ensure the FastAPI server is running.`,
          error: true
        }
      ])
    } finally {
      setIsAsking(false)
    }
  }

  // Start quiz session
  const startQuizSession = (questions) => {
    setQuizQuestions(questions)
    setQuizIndex(0)
    setQuizChoice('')
    setQuizSubmitted(false)
    setQuizScore(0)
    setQuizCompleted(false)
    setQuizStarted(true)
    window.history.pushState({ page: 'Practice & Quizzes', sub: 'quiz' }, '', '#/practice/quiz')
  }

  // Exit quiz session back to practice menu
  const exitQuiz = () => {
    setQuizStarted(false)
    setQuizCompleted(false)
    setQuizChoice('')
    setQuizSubmitted(false)
    window.history.pushState({ page: 'Practice & Quizzes' }, '', '#/practice')
  }

  // Load questions from backend question bank
  const loadChapterQuestions = async () => {
    setIsGeneratingQuiz(true)
    setQuizGenMsg('Loading questions from question bank...')
    try {
      const qs = await getQuestions(quizBook, quizChapter || null, 'mcq', 10)
      if (qs && qs.length > 0) {
        startQuizSession(qs)
        setQuizGenMsg(`Loaded ${qs.length} verified questions for this chapter!`)
      } else {
        setQuizGenMsg(`No pre-saved questions found for this chapter. Generating a new set...`)
        await triggerGenerateQuestions()
      }
    } catch (err) {
      setQuizGenMsg(`Using starter question set (${err.message})`)
      startQuizSession(defaultQuizQuestions)
    } finally {
      setIsGeneratingQuiz(false)
    }
  }

  // Generate new questions via /generate/questions
  const triggerGenerateQuestions = async () => {
    setIsGeneratingQuiz(true)
    setQuizGenMsg('Generating AI exam questions from textbook passages...')
    try {
      const res = await generateQuestions(quizBook, quizChapter, 'mcq', 4, 'medium')
      const qs = await getQuestions(quizBook, quizChapter, 'mcq', 10)
      if (qs && qs.length > 0) {
        startQuizSession(qs)
        setQuizGenMsg(`Generated and verified ${qs.length} questions!`)
      } else {
        setQuizGenMsg(res?.message || 'Generated questions. Starting quiz set.')
        startQuizSession(defaultQuizQuestions)
      }
    } catch (err) {
      setQuizGenMsg(`Note: AI generator offline (${err.message}). Loaded standard practice set.`)
      startQuizSession(defaultQuizQuestions)
    } finally {
      setIsGeneratingQuiz(false)
    }
  }

  const currentQuizQ = quizQuestions[quizIndex] || quizQuestions[0]

  // Submit answer in quiz & dynamically record to backend
  const handleQuizSubmit = () => {
    if (!quizChoice || quizSubmitted) return
    const isCorrect = (
      quizChoice === currentQuizQ.answer ||
      quizChoice === currentQuizQ.answer_text ||
      (currentQuizQ.options && currentQuizQ.options[quizChoice.charCodeAt(0) - 65] === currentQuizQ.answer_text)
    )
    if (isCorrect) {
      setQuizScore(s => s + 1)
    }
    setQuizSubmitted(true)

    // Log attempt to backend to dynamically update topic mastery
    recordQuizAttempt({
      student_id: profile.username || 'default',
      question_id: String(currentQuizQ.id || 'q'),
      topic_id: currentQuizQ.book_id || quizBook,
      topic_name: currentQuizQ.chapter_title || currentQuizQ.book_id,
      question_text: currentQuizQ.question,
      selected_answer: quizChoice,
      correct_answer: currentQuizQ.answer || currentQuizQ.answer_text || '',
      is_correct: isCorrect,
    }).then(() => {
      // Refresh mastery & history
      getProgressMastery().then(setProgressMastery).catch(() => {})
      getQuizHistory().then(setQuizHistory).catch(() => {})
      getOverviewStats().then(setOverviewStats).catch(() => {})
    }).catch(() => {})
  }

  // Next question in quiz
  const handleQuizNext = () => {
    if (quizIndex + 1 < quizQuestions.length) {
      setQuizIndex(i => i + 1)
      setQuizChoice('')
      setQuizSubmitted(false)
    } else {
      setQuizCompleted(true)
    }
  }

  // Topbar search action
  const handleTopSearch = (e) => {
    if (e.key === 'Enter' && query.trim()) {
      setAnswer(query)
      go('AI Study Assistant')
    }
  }

  // Explore subject syllabus drawer
  const handleExploreSubject = async (code) => {
    setLoadingSubjectDetail(true)
    try {
      const details = await getSubjectDetails(code)
      setSelectedSubjectDetail(details)
    } catch {
      setSelectedSubjectDetail(null)
    } finally {
      setLoadingSubjectDetail(false)
    }
  }

  // Save student profile
  const handleSaveProfile = async (e) => {
    e.preventDefault()
    setProfileSavedMsg('Saving...')
    try {
      await updateStudentProfile(profile)
      setProfileSavedMsg('Profile settings saved successfully!')
      setTimeout(() => setProfileSavedMsg(''), 3000)
    } catch (err) {
      setProfileSavedMsg(`Failed to save: ${err.message}`)
    }
  }

  return (
    <div className={`app-shell theme-${theme}`} data-theme={theme}>
      <aside className={`sidebar ${mobile ? 'show' : ''}`}>
        <div
          className="brand"
          onClick={() => go('Overview')}
          style={{ cursor: 'pointer' }}
          title="Return to Workspace Overview"
        >
          <div className="brand-mark"><Sparkles size={21} /></div>
          <div><b>PrepPilot</b><span>YOUR AI STUDY COMPANION</span></div>
          <button className="mobile-close" onClick={() => setMobile(false)}><X size={18} /></button>
        </div>

        <div
          className="workspace-label"
          onClick={() => go('Overview')}
          style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
          title="Go to Workspace"
        >
          WORKSPACE
        </div>

        <nav>
          {nav.map(([label, Icon]) => (
            <button
              key={label}
              className={`nav-item ${page === label ? 'active' : ''}`}
              onClick={() => go(label)}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === 'AI Study Assistant' && <i className="new-dot" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="streak-card">
            <div className="streak-icon"><Flame size={17} /></div>
            <div>
              <b>{overviewStats.chunks_indexed.toLocaleString()} chunks</b>
              <span>Hybrid Search Vector DB</span>
            </div>
          </div>
          <button className="nav-item" onClick={() => go('Settings')}><Settings size={18} />Settings</button>
          <div className="profile" onClick={() => go('Settings')} style={{ cursor: 'pointer' }}>
            <div className="avatar">PP</div>
            <div>
              <b>{profile.name}</b>
              <span>{profile.institution} · {profile.semester}</span>
            </div>
            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      {mobile && <div className="scrim" onClick={() => setMobile(false)} />}

      <main className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobile(true)}><Menu /></button>
          
          {/* Interactive Clickable Breadcrumb */}
          <div className="crumb">
            <button className="crumb-btn" onClick={() => go('Overview')} title="Go to Workspace Overview">
              <LayoutDashboard size={14} /> Workspace
            </button>
            {page !== 'Overview' && (
              <>
                <span>/</span>
                {page === 'Practice & Quizzes' && quizStarted ? (
                  <>
                    <button className="crumb-btn" onClick={exitQuiz} title="Back to Chapter List">
                      Practice &amp; Quizzes
                    </button>
                    <span>/</span>
                    <b>Active Quiz ({currentQuizQ?.book_id || quizBook} Ch. {currentQuizQ?.chapter || quizChapter})</b>
                  </>
                ) : (
                  <b>{page}</b>
                )}
              </>
            )}
          </div>

          <div className="top-actions">
            <div className="searchbox">
              <Search size={16} />
              <input
                placeholder="Search textbooks (press Enter)..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleTopSearch}
              />
              <kbd>↵</kbd>
            </div>
            <div className={`select-pill ${health.status === 'ok' ? '' : 'attention-label'}`}>
              <i className={health.status === 'ok' ? 'online-dot' : ''} />
              {health.status === 'ok' ? 'Backend Live' : 'Connecting'}
            </div>
            <button className="icon-btn"><Bell size={18} /><i /></button>
            <div className="top-avatar" onClick={() => go('Settings')} style={{ cursor: 'pointer' }}>
              {profile.name ? profile.name.slice(0, 1) : 'S'}
            </div>
          </div>
        </header>

        <div className="content">
          {/* ===================== OVERVIEW / WORKSPACE ===================== */}
          {page === 'Overview' && (
            <>
              <div className="welcome-row">
                <div>
                  <div className="eyebrow">PREPPILOT LEARNING SYSTEM</div>
                  <h1>Good morning, {profile.name} <span>✦</span></h1>
                  <p>Textbook library indexed with {overviewStats.chunks_indexed.toLocaleString()} passages across 4 curriculum books.</p>
                </div>
                <button className="primary-btn" onClick={() => go('AI Study Assistant')}>
                  <Bot size={17} /> Ask Study Assistant
                </button>
              </div>

              <div className="stats-grid">
                <Stat icon={BookOpen} label="Textbook Passages" value={overviewStats.chunks_indexed.toLocaleString()} sub="In SQLite Vector DB" trend="Hybrid Search Active" />
                <Stat icon={Target} label="Quiz Accuracy" value={`${overviewStats.quiz_accuracy}%`} sub={`Across ${overviewStats.total_quizzes} attempts`} trend="+8.2% this week" />
                <Stat icon={Flame} label="Study Streak" value={`${overviewStats.study_streak_days} days`} sub="Personal best: 12 days" trend="You're on a roll!" />
                <Stat icon={CheckCircle2} label="Topics Mastered" value={`${overviewStats.topics_mastered}`} sub={`Out of ${overviewStats.total_topics} topics`} trend="5 mastered this week" />
              </div>

              <div className="section-heading">
                <div><h2>Your Course Curriculum</h2><p>Live course metrics from your indexed textbooks</p></div>
                <button className="text-btn" onClick={() => go('My Subjects')}>View all subjects <ArrowUpRight size={15} /></button>
              </div>

              <div className="subject-grid">
                {subjectsList.map(s => (
                  <div className="subject-card" key={s.code}>
                    <div className="subject-top">
                      <div className={`subject-icon ${s.color}`}>{s.code.slice(0, 3)}</div>
                      <span className="subject-tag">{s.code}</span>
                    </div>
                    <h3>{s.name}</h3>
                    <div className="progress-meta">
                      <span>Curriculum mastery</span>
                      <b>{s.progress}%</b>
                    </div>
                    <div className="progress-track">
                      <div className={`progress-fill ${s.color}`} style={{ width: s.progress + '%' }} />
                    </div>
                    <div className="subject-footer">
                      <span>{s.chapters} ({s.chunks ? s.chunks.toLocaleString() : '—'} chunks)</span>
                      <button onClick={() => { setSelectedBook(s.code); go('AI Study Assistant') }}>
                        Ask topic <ArrowUpRight size={14} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="lower-grid">
                <div className="panel">
                  <div className="panel-heading">
                    <div><h2>Study Activity</h2><p>Weekly learning frequency</p></div>
                    <span className="select-pill">This week <ChevronDown size={14} /></span>
                  </div>
                  <div className="chart-wrap">
                    <ResponsiveContainer width="100%" height={210}>
                      <AreaChart data={overviewStats.activity || fallbackActivity}>
                        <defs>
                          <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#7188ff" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="#7188ff" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="#202a40" vertical={false} />
                        <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#8190aa', fontSize: 12 }} />
                        <YAxis hide />
                        <Tooltip contentStyle={{ background: '#151e31', border: '1px solid #2b3851', borderRadius: 10, color: '#fff' }} />
                        <Area type="monotone" dataKey="score" stroke="#8294ff" strokeWidth={2.5} fill="url(#area)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-heading">
                    <div><h2>Focus Areas</h2><p>Topics needing reinforcement</p></div>
                    <span className="attention-label">Needs focus</span>
                  </div>
                  <div className="focus-list">
                    {(overviewStats.focus_areas || []).map((f, fIdx) => (
                      <div className="focus-item" key={fIdx}>
                        <div className="focus-line">
                          <span>{f.name}</span>
                          <b>{f.score}%</b>
                        </div>
                        <div className="progress-track">
                          <div className={`progress-fill ${f.color || 'violet'}`} style={{ width: f.score + '%' }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  <button className="wide-btn" style={{ marginTop: '14px' }} onClick={() => go('My Progress')}>
                    Review Topic Mastery <ArrowUpRight size={15} />
                  </button>
                </div>
              </div>
            </>
          )}

          {/* ===================== MY SUBJECTS ===================== */}
          {page === 'My Subjects' && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <button className="back-nav-btn" onClick={() => go('Overview')}>
                  <ChevronLeft size={15} /> Back to Workspace
                </button>
              </div>
              <PageTitle title="My Subjects & Textbooks" desc="Your course materials, chapter breakdowns, and indexed passages." />
              <div className="subject-grid large">
                {subjectsList.map(s => (
                  <div className="subject-card" key={s.code}>
                    <div className="subject-top">
                      <div className={`subject-icon ${s.color}`}>{s.code.slice(0, 3)}</div>
                      <span className="subject-tag">{s.code}</span>
                    </div>
                    <h3>{s.name}</h3>
                    <p style={{ fontSize: '10px', color: '#8898b3', margin: '4px 0 12px', minHeight: '32px' }}>
                      {s.description || s.book_title}
                    </p>
                    <div className="progress-meta">
                      <span>Curriculum Coverage ({s.chapters_count || s.chapters})</span>
                      <b>{s.progress}%</b>
                    </div>
                    <div className="progress-track"><div className={`progress-fill ${s.color}`} style={{ width: s.progress + '%' }} /></div>
                    
                    <div className="next-topic">
                      <span>NEXT KEY TOPIC</span>
                      <b>{s.next}</b>
                    </div>

                    <div style={{ display: 'flex', gap: '8px', marginTop: '14px' }}>
                      <button className="primary-btn" style={{ flex: 1 }} onClick={() => { setSelectedBook(s.code); go('AI Study Assistant') }}>
                        <Bot size={14} /> Ask Assistant
                      </button>
                      <button className="wide-btn" style={{ flex: 1 }} onClick={() => handleExploreSubject(s.code)}>
                        <BookOpen size={14} /> View Syllabus
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Subject Chapter Syllabus Drawer */}
              {selectedSubjectDetail && (
                <div className="chapter-drawer">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                    <div>
                      <h2 style={{ margin: 0, fontSize: '16px' }}>{selectedSubjectDetail.name} — Chapters & Syllabus</h2>
                      <span style={{ fontSize: '10px', color: '#8393ad' }}>
                        {selectedSubjectDetail.book_title} · {selectedSubjectDetail.total_chunks.toLocaleString()} passages indexed
                      </span>
                    </div>
                    <button className="back-nav-btn" onClick={() => setSelectedSubjectDetail(null)}>Close ✕</button>
                  </div>

                  <div style={{ maxHeight: '320px', overflowY: 'auto', paddingRight: '6px' }}>
                    {(selectedSubjectDetail.chapters || []).map((ch, cIdx) => (
                      <div className="chapter-item" key={cIdx}>
                        <div>
                          <b>Chapter {ch.chapter}: {ch.chapter_title}</b>
                          <span>{ch.chunks} indexed vector passages</span>
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button
                            className="text-btn"
                            onClick={() => {
                              setQuizBook(selectedSubjectDetail.code)
                              setQuizChapter(ch.chapter)
                              go('Practice & Quizzes')
                            }}
                          >
                            Quiz <ArrowUpRight size={12} />
                          </button>
                          <button
                            className="text-btn"
                            onClick={() => {
                              setSelectedBook(selectedSubjectDetail.code)
                              setAnswer(`Explain Chapter ${ch.chapter}: ${ch.chapter_title}`)
                              go('AI Study Assistant')
                            }}
                          >
                            Study <ArrowUpRight size={12} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ===================== AI STUDY ASSISTANT ===================== */}
          {page === 'AI Study Assistant' && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <button className="back-nav-btn" onClick={() => go('Overview')}>
                  <ChevronLeft size={15} /> Back to Workspace
                </button>
              </div>
              <PageTitle title="AI Study Assistant" desc="Answers grounded in your 4 course textbooks with source citations." />
              <div className="chat-layout">
                <div className="chat-main">
                  <div className="chat-header">
                    <div className="bot-avatar"><Bot /></div>
                    <div>
                      <b>PrepPilot Assistant</b>
                      <span><i className="online-dot" /> Live · Connected to FastAPI Backend ({overviewStats.chunks_indexed.toLocaleString()} chunks)</span>
                    </div>
                    <span className="demo-chip">RAG ACTIVE</span>
                  </div>

                  <div className="chat-messages">
                    {messages.map((m, i) => (
                      <div className={`message ${m.role}`} key={i}>
                        <div className="message-avatar">
                          {m.role === 'assistant' ? <Sparkles size={15} /> : 'S'}
                        </div>
                        <div style={{ maxWidth: '100%' }}>
                          <div className={`bubble ${m.error ? 'quiz-feedback-box wrong' : ''}`}>
                            {m.text}
                          </div>

                          {/* Source Excerpts & Citations */}
                          {m.sources && m.sources.length > 0 && (
                            <div style={{ marginTop: '6px' }}>
                              <button
                                className="sources-toggle"
                                onClick={() => setExpandedSources(prev => ({ ...prev, [i]: !prev[i] }))}
                              >
                                <BookMarked size={12} />
                                {expandedSources[i] ? 'Hide' : 'View'} {m.sources.length} Textbook Sources &amp; Excerpts
                                <ChevronDown size={12} style={{ transform: expandedSources[i] ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
                              </button>

                              {expandedSources[i] && (
                                <div className="sources-box">
                                  {m.sources.map((src, sIdx) => (
                                    <div className="source-card-item" key={sIdx}>
                                      <b>[{src.rank}] {src.location}</b>
                                      <span className="source-tag">Similarity: {Math.round((src.similarity || 0) * 100)}%</span>
                                      <span className="source-tag">{src.found_by || 'vector'}</span>
                                      <p>"{src.text?.slice(0, 240)}..."</p>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}

                    {isAsking && (
                      <div className="message assistant">
                        <div className="message-avatar"><Sparkles size={15} /></div>
                        <div className="bubble loading">
                          <LoaderCircle size={14} className="spin" />
                          <span>Searching textbooks and preparing answer...</span>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="suggestions">
                    <span>QUICK QUESTIONS</span>
                    {[
                      'What is a red-black tree and its properties?',
                      'Explain binary search tree search complexity',
                      'How does BUILD-MAX-HEAP work in CLRS?',
                      'Difference between stack and queue in C++'
                    ].map(x => (
                      <button key={x} onClick={() => setAnswer(x)}>{x}</button>
                    ))}
                  </div>

                  <div className="chat-input">
                    <input
                      value={answer}
                      onChange={e => setAnswer(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && send()}
                      placeholder="Ask any concept or algorithm question..."
                    />
                    <button onClick={send} disabled={isAsking}><Send size={17} /></button>
                  </div>
                  <p className="disclaimer">Answers are strictly retrieved and verified against your course textbooks with bracketed citations [1], [2].</p>
                </div>

                <div className="chat-side">
                  <h3>Study Context</h3>
                  <label>SEARCH TARGET</label>
                  <select value={selectedBook} onChange={e => setSelectedBook(e.target.value)}>
                    <option value="ALL">All Textbooks (Full Library)</option>
                    <option value="CLRS">Introduction to Algorithms (CLRS)</option>
                    <option value="DS">Data Structures Handbook</option>
                    <option value="C">Let Us C (Kanetkar)</option>
                    <option value="CPP">C++: The Complete Reference</option>
                  </select>

                  {bookChapters.length > 0 && (
                    <>
                      <label>INDEXED CHAPTERS ({bookChapters.length})</label>
                      <div style={{ maxHeight: '180px', overflowY: 'auto', paddingRight: '4px', fontSize: '10px' }}>
                        {bookChapters.slice(0, 15).map(c => (
                          <div key={c.chapter} style={{ padding: '4px 0', borderBottom: '1px solid #1f2c42', color: '#9bb0cb' }}>
                            Ch. {c.chapter}: {c.chapter_title} ({c.chunks} chunks)
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  <label>DATABASE STATUS</label>
                  <div className="resource-mini"><BookOpen size={16} /><span>Indexed Chunks</span><b>{overviewStats.chunks_indexed.toLocaleString()}</b></div>
                  <div className="resource-mini"><Library size={16} /><span>Keyword Search</span><b>{health.keyword_search ? 'FTS5 BM25' : 'Vector only'}</b></div>
                  <div className="resource-mini"><ClipboardList size={16} /><span>Search Method</span><b>RRF Hybrid</b></div>
                </div>
              </div>
            </>
          )}

          {/* ===================== PRACTICE & QUIZZES ===================== */}
          {page === 'Practice & Quizzes' && (
            <>
              {/* Back navigation buttons for Practice & Quizzes */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
                {quizStarted ? (
                  <>
                    <button className="back-nav-btn" onClick={exitQuiz}>
                      <ChevronLeft size={15} /> Exit Quiz &amp; Return to Chapter List
                    </button>
                    <button className="back-nav-btn" onClick={() => go('Overview')}>
                      <LayoutDashboard size={14} /> Back to Workspace
                    </button>
                  </>
                ) : (
                  <button className="back-nav-btn" onClick={() => go('Overview')}>
                    <ChevronLeft size={15} /> Back to Workspace Overview
                  </button>
                )}
              </div>

              <PageTitle title="Practice & Quizzes" desc="Interactive question bank verified against textbook excerpts." />
              <div className="quiz-layout">
                <div className="panel quiz-panel">
                  {quizCompleted ? (
                    <div className="quiz-summary-card">
                      <div className="quiz-score-circle">
                        {Math.round((quizScore / quizQuestions.length) * 100)}%
                      </div>
                      <h2>Quiz Complete!</h2>
                      <p style={{ margin: '8px 0 20px' }}>
                        You answered <b>{quizScore}</b> out of <b>{quizQuestions.length}</b> questions correctly.
                        Your topic mastery score was updated in the database.
                      </p>
                      
                      <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginTop: '16px', flexWrap: 'wrap' }}>
                        <button
                          className="primary-btn"
                          onClick={() => {
                            setQuizIndex(0)
                            setQuizSubmitted(false)
                            setQuizChoice('')
                            setQuizScore(0)
                            setQuizCompleted(false)
                            setQuizStarted(true)
                          }}
                        >
                          <RefreshCw size={15} /> Retake Quiz
                        </button>
                        <button className="back-nav-btn" onClick={exitQuiz}>
                          <ClipboardList size={15} /> Practice Another Topic
                        </button>
                        <button className="back-nav-btn" onClick={() => go('Overview')}>
                          <LayoutDashboard size={15} /> Back to Workspace
                        </button>
                      </div>
                    </div>
                  ) : quizStarted && currentQuizQ ? (
                    <>
                      <div className="quiz-topline">
                        <span className="subject-tag">
                          {currentQuizQ.book_id || quizBook} · CH. {currentQuizQ.chapter || 'TOPIC'}
                        </span>
                        <span className="time-chip">
                          Question {quizIndex + 1} of {quizQuestions.length}
                        </span>
                      </div>

                      <h2>{currentQuizQ.question}</h2>

                      <div className="options">
                        {(currentQuizQ.options || []).map((opt, oIdx) => {
                          const letter = 'ABCD'[oIdx]
                          const isChosen = quizChoice === letter || quizChoice === opt
                          let btnClass = isChosen ? 'chosen' : ''
                          if (quizSubmitted) {
                            const isCorrectOpt = letter === currentQuizQ.answer || opt === currentQuizQ.answer_text
                            if (isCorrectOpt) btnClass = 'option-correct'
                            else if (isChosen) btnClass = 'option-wrong'
                          }

                          return (
                            <button
                              key={oIdx}
                              className={btnClass}
                              disabled={quizSubmitted}
                              onClick={() => setQuizChoice(letter)}
                            >
                              <b style={{ marginRight: '8px', color: '#9ab4ff' }}>{letter}.</b>
                              {opt}
                            </button>
                          )
                        })}
                      </div>

                      {quizSubmitted && (
                        <div className={`quiz-feedback-box ${(quizChoice === currentQuizQ.answer || quizChoice === currentQuizQ.answer_text) ? 'correct' : 'wrong'}`}>
                          <b>
                            {(quizChoice === currentQuizQ.answer || quizChoice === currentQuizQ.answer_text)
                              ? '✓ Correct!'
                              : `✗ Incorrect. Correct answer is (${currentQuizQ.answer || 'A'}) ${currentQuizQ.answer_text || ''}`}
                          </b>
                          <p style={{ margin: '6px 0 0' }}>{currentQuizQ.explanation}</p>
                          {currentQuizQ.chapter_title && (
                            <small style={{ display: 'block', marginTop: '6px', opacity: 0.85 }}>
                              Source: {BOOK_TITLES[currentQuizQ.book_id] || currentQuizQ.book_id} &gt; {currentQuizQ.chapter_title}
                            </small>
                          )}
                        </div>
                      )}

                      <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
                        {!quizSubmitted ? (
                          <button
                            className="primary-btn full"
                            disabled={!quizChoice}
                            onClick={handleQuizSubmit}
                          >
                            Submit Answer
                          </button>
                        ) : (
                          <button
                            className="primary-btn full"
                            onClick={handleQuizNext}
                          >
                            {quizIndex + 1 < quizQuestions.length ? 'Next Question →' : 'Finish Quiz & View Results'}
                          </button>
                        )}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="quiz-topline">
                        <span className="subject-tag">PREPPILOT QUESTION BANK</span>
                        <span className="time-chip"><Target size={14} /> Verified MCQs</span>
                      </div>
                      <h2>Ready for a knowledge check?</h2>
                      <p>
                        Choose a textbook and chapter below to test yourself on verified questions generated from your course materials.
                      </p>

                      <div className="quiz-details">
                        <span><ClipboardList size={16} /> 5 questions</span>
                        <span><Target size={16} /> Instant feedback</span>
                        <span><BookOpen size={16} /> Textbook grounded</span>
                      </div>

                      <button
                        className="primary-btn"
                        onClick={() => startQuizSession(defaultQuizQuestions)}
                      >
                        <Play size={16} /> Start Core Practice Quiz
                      </button>
                    </>
                  )}
                </div>

                <div className="panel">
                  <h2>Select Textbook Chapter</h2>
                  <p className="muted">Load questions from the backend or generate fresh sets.</p>

                  <div className="quiz-chapter-row">
                    <div>
                      <label style={{ fontSize: '9px', color: '#8898b3', display: 'block', marginBottom: '6px' }}>TEXTBOOK</label>
                      <select
                        style={{ width: '100%', padding: '9px', background: '#172239', color: '#dce4f4', border: '1px solid #2e3c59', borderRadius: '7px' }}
                        value={quizBook}
                        onChange={e => setQuizBook(e.target.value)}
                      >
                        <option value="CLRS">CLRS (Algorithms)</option>
                        <option value="DS">Data Structures</option>
                        <option value="C">Let Us C</option>
                        <option value="CPP">C++ Complete Ref</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '9px', color: '#8898b3', display: 'block', marginBottom: '6px' }}>CHAPTER</label>
                      <select
                        style={{ width: '100%', padding: '9px', background: '#172239', color: '#dce4f4', border: '1px solid #2e3c59', borderRadius: '7px' }}
                        value={quizChapter}
                        onChange={e => setQuizChapter(e.target.value)}
                      >
                        {quizChaptersList.map(c => (
                          <option key={c.chapter} value={c.chapter}>
                            Ch. {c.chapter}: {c.chapter_title?.slice(0, 20)}...
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {quizGenMsg && (
                    <div style={{ fontSize: '10px', color: '#95abdf', padding: '8px 10px', background: '#16233d', borderRadius: '6px', margin: '12px 0' }}>
                      {quizGenMsg}
                    </div>
                  )}

                  <div style={{ display: 'grid', gap: '8px', marginTop: '14px' }}>
                    <button
                      className="primary-btn"
                      disabled={isGeneratingQuiz}
                      onClick={loadChapterQuestions}
                    >
                      {isGeneratingQuiz ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}
                      Load &amp; Start Chapter Quiz
                    </button>
                    
                    <button
                      className="back-nav-btn"
                      style={{ justifyContent: 'center' }}
                      onClick={() => go('Overview')}
                    >
                      <LayoutDashboard size={14} /> Return to Workspace Overview
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ===================== FLASHCARDS ===================== */}
          {page === 'Flashcards' && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <button className="back-nav-btn" onClick={() => go('Overview')}>
                  <ChevronLeft size={15} /> Back to Workspace
                </button>
              </div>
              <Flashcards />
            </>
          )}

          {/* ===================== MY PROGRESS ===================== */}
          {page === 'My Progress' && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <button className="back-nav-btn" onClick={() => go('Overview')}>
                  <ChevronLeft size={15} /> Back to Workspace
                </button>
              </div>
              <PageTitle title="My Progress & Mastery" desc="Dynamic topic mastery scores and quiz attempt history from SQLite." />
              <div className="stats-grid">
                <Stat icon={Target} label="Average Quiz Score" value={`${overviewStats.quiz_accuracy}%`} sub={`Across ${overviewStats.total_quizzes} attempts`} trend="+6% this week" />
                <Stat icon={CheckCircle2} label="Mastered Topics" value={`${overviewStats.topics_mastered}`} sub="Score >= 70%" trend="5 this week" />
                <Stat icon={Clock3} label="Learning Time" value={`${overviewStats.study_time_hours}h`} sub="Active study sessions" trend="+12% this month" />
                <Stat icon={Flame} label="Consistency" value={`${overviewStats.study_streak_days} days`} sub="Weekly study streak" trend="Great consistency" />
              </div>

              {/* Live Topic Mastery Chart */}
              <div className="panel progress-panel">
                <div className="panel-heading">
                  <div><h2>Topic Mastery Breakdown</h2><p>Evaluated scores (0–100%) from your quiz performance in SQLite</p></div>
                </div>
                <ResponsiveContainer width="100%" height={290}>
                  <BarChart data={progressMastery} margin={{ top: 15, right: 15, left: 0, bottom: 5 }}>
                    <CartesianGrid stroke="#202a40" vertical={false} />
                    <XAxis dataKey="topic" axisLine={false} tickLine={false} tick={{ fill: '#9aa8bf', fontSize: 11 }} />
                    <YAxis domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fill: '#9aa8bf' }} />
                    <Tooltip contentStyle={{ background: '#151e31', border: '1px solid #2b3851', borderRadius: 10 }} />
                    <Bar dataKey="score" fill="#8193ff" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Recent Quiz Attempts History */}
              <div className="panel" style={{ marginTop: '20px' }}>
                <div className="panel-heading">
                  <div><h2>Recent Quiz Attempts</h2><p>Logged answers and score updates in `quiz_attempts`</p></div>
                </div>

                {quizHistory.length === 0 ? (
                  <p style={{ color: '#8898b3', fontSize: '11px' }}>No quiz attempts logged yet. Complete a quiz to view your history.</p>
                ) : (
                  <div style={{ overflowX: 'auto' }}>
                    <table className="history-table">
                      <thead>
                        <tr>
                          <th>Topic</th>
                          <th>Question</th>
                          <th>Selected Answer</th>
                          <th>Correct Answer</th>
                          <th>Status</th>
                          <th>Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {quizHistory.map((item, idx) => (
                          <tr key={idx}>
                            <td><b>{item.topic_id}</b></td>
                            <td style={{ maxWidth: '320px' }}>{item.question_text}</td>
                            <td>{item.selected_answer}</td>
                            <td>{item.correct_answer}</td>
                            <td>
                              <span className={`history-badge ${item.is_correct ? 'correct' : 'wrong'}`}>
                                {item.is_correct ? '✓ Correct' : '✗ Incorrect'}
                              </span>
                            </td>
                            <td style={{ color: '#8797b0', fontSize: '10px' }}>
                              {item.attempted_at ? String(item.attempted_at).slice(0, 16) : 'Recently'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </>
          )}

          {/* ===================== STUDY RESOURCES ===================== */}
          {page === 'Study Resources' && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <button className="back-nav-btn" onClick={() => go('Overview')}>
                  <ChevronLeft size={15} /> Back to Workspace
                </button>
              </div>
              <PageTitle title="Study Resources & Academic Catalog" desc="All indexed textbooks, vector embeddings, and extracted diagrams." />
              
              <div className="resource-grid">
                {[
                  ['Textbooks', `${overviewStats.chunks_indexed.toLocaleString()} passages across 4 textbooks`, 'BookOpen'],
                  ['Visual Figures', `${studyResources?.total_diagrams || 1252} diagrams & charts cataloged`, 'ImageIcon'],
                  ['Exam Questions', `${studyResources?.total_pyqs || 9} PYQs + Question Bank`, 'ClipboardList']
                ].map(([t, n]) => (
                  <div className="panel resource-card" key={t}>
                    <div className="resource-big"><Library /></div>
                    <h3>{t}</h3>
                    <p>{n}</p>
                    <button className="wide-btn" onClick={() => go('My Subjects')}>
                      Browse resources <ArrowUpRight size={15} />
                    </button>
                  </div>
                ))}
              </div>

              {/* Indexed Textbooks Catalog */}
              <div className="panel" style={{ marginTop: '20px' }}>
                <div className="panel-heading">
                  <div><h2>Indexed Textbook Library</h2><p>Pre-processed and embedded using 384-dimensional dense vectors</p></div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: '14px', marginTop: '12px' }}>
                  {(studyResources?.textbooks || []).map((b, bIdx) => (
                    <div className="source-card-item" key={bIdx} style={{ padding: '16px' }}>
                      <span className="source-tag">{b.book_id}</span>
                      <b style={{ fontSize: '12px', marginTop: '6px' }}>{b.title}</b>
                      <span style={{ fontSize: '10px', color: '#8797b2', display: 'block', margin: '4px 0 8px' }}>{b.subject}</span>
                      <div style={{ fontSize: '10px', color: '#c5d0e2', lineHeight: '1.6' }}>
                        <div>• <b>Chapters:</b> {b.chapters}</div>
                        <div>• <b>Vector Passages:</b> {b.passages?.toLocaleString()}</div>
                        <div>• <b>Coverage:</b> {b.page_range}</div>
                        <div>• <b>Extracted Figures:</b> {b.diagrams}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Extracted Figures & Diagrams Gallery */}
              {resourceFigures.length > 0 && (
                <div className="panel" style={{ marginTop: '20px' }}>
                  <div className="panel-heading">
                    <div><h2>Extracted Diagrams & Figures ({studyResources?.total_diagrams || 1252})</h2><p>High-resolution diagrams extracted from course materials for visual recall</p></div>
                  </div>

                  <div className="figure-grid">
                    {resourceFigures.map((fig, fIdx) => (
                      <div className="figure-card" key={fIdx}>
                        <div>
                          <span className="source-tag">{fig.book_id} · p. {fig.page_number}</span>
                          <b>Figure {fig.image_id}</b>
                          <p>"{fig.caption || 'Algorithm structure and architectural diagram'}"</p>
                        </div>
                        <div style={{ marginTop: '12px', fontSize: '9px', color: '#7486a2' }}>
                          Type: {fig.image_type || 'vector_figure'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ===================== SETTINGS ===================== */}
          {page === 'Settings' && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <button className="back-nav-btn" onClick={() => go('Overview')}>
                  <ChevronLeft size={15} /> Back to Workspace
                </button>
              </div>
              <PageTitle title="Settings & Configuration" desc="Student profile, color themes, and local model inference settings." />
              <div className="settings-layout">
                {/* Theme Options */}
                <div className="panel settings-panel appearance-panel">
                  <div className="settings-section-title">
                    <div><h2>Appearance</h2><p className="muted">Choose a comfortable look for your study sessions.</p></div>
                    <span className="appearance-icon"><Sparkles size={18} /></span>
                  </div>
                  <label>COLOR THEME</label>
                  <div className="theme-options">
                    {[
                      { id: 'midnight', name: 'Midnight', desc: 'Original dark blue', swatches: ['#0b1020', '#17243b', '#7c8cff'] },
                      { id: 'light', name: 'Light', desc: 'Bright and minimal', swatches: ['#f3f6fc', '#ffffff', '#5268df'] },
                      { id: 'ocean', name: 'Ocean', desc: 'Deep teal tones', swatches: ['#071b22', '#10313a', '#39b9b0'] }
                    ].map(t => (
                      <button
                        type="button"
                        key={t.id}
                        className={`theme-choice ${theme === t.id ? 'selected' : ''}`}
                        onClick={() => setTheme(t.id)}
                        aria-pressed={theme === t.id}
                      >
                        <span className={`theme-preview ${t.id}`}><i /><i /><i /></span>
                        <span className="theme-choice-copy"><b>{t.name}</b><small>{t.desc}</small></span>
                        <span className="theme-radio">{theme === t.id && <CheckCircle2 size={17} />}</span>
                      </button>
                    ))}
                  </div>
                  <p className="theme-saved-note"><CheckCircle2 size={14} /> Your theme is saved automatically on this device.</p>

                  {/* System & Hardware Diagnostics */}
                  <div style={{ marginTop: '28px', borderTop: '1px solid #233149', paddingTop: '20px' }}>
                    <h3 style={{ fontSize: '13px', margin: '0 0 10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Cpu size={16} /> System Diagnostics
                    </h3>
                    <div style={{ fontSize: '10px', color: '#c5d0e4', lineHeight: '1.8' }}>
                      <div>• <b>Database:</b> {systemSettings?.database_size_mb || 55.4} MB SQLite (WAL Mode active)</div>
                      <div>• <b>Search Engine:</b> {systemSettings?.embedding_model || 'sentence-transformers/all-MiniLM-L6-v2'}</div>
                      <div>• <b>Vector Space:</b> {systemSettings?.vector_dimensions || 384} dimensions + SQLite FTS5</div>
                      <div>• <b>Passages Indexed:</b> {overviewStats.chunks_indexed.toLocaleString()} chunks</div>
                      <div>• <b>Local LLM Endpoint:</b> {systemSettings?.local_llm_url || 'http://localhost:11434/v1'}</div>
                    </div>
                  </div>
                </div>

                {/* Profile Form */}
                <div className="panel settings-panel profile-panel">
                  <h2>Student Profile</h2>
                  <p className="muted">Persisted in SQLite database (`students` table).</p>

                  <form onSubmit={handleSaveProfile} style={{ display: 'grid', gap: '12px', marginTop: '14px' }}>
                    <div>
                      <label>STUDENT NAME</label>
                      <input
                        value={profile.name}
                        onChange={e => setProfile(p => ({ ...p, name: e.target.value }))}
                        required
                      />
                    </div>

                    <div>
                      <label>INSTITUTION</label>
                      <input
                        value={profile.institution || ''}
                        onChange={e => setProfile(p => ({ ...p, institution: e.target.value }))}
                      />
                    </div>

                    <div>
                      <label>SEMESTER</label>
                      <input
                        value={profile.semester || ''}
                        onChange={e => setProfile(p => ({ ...p, semester: e.target.value }))}
                      />
                    </div>

                    <div>
                      <label>PREFERRED LOCAL MODEL</label>
                      <select
                        style={{ width: '100%', padding: '10px', background: '#111a2d', color: '#dce4f4', border: '1px solid #303d58', borderRadius: '7px' }}
                        value={profile.preferred_model || 'qwen2.5:7b'}
                        onChange={e => setProfile(p => ({ ...p, preferred_model: e.target.value }))}
                      >
                        <option value="qwen2.5:7b">Qwen 2.5 (7B) — Recommended for 8GB VRAM RTX 3070</option>
                        <option value="llama3.1:8b">Llama 3.1 (8B Instruct)</option>
                        <option value="llama3.2:3b">Llama 3.2 (3B) — Lightweight</option>
                        <option value="qwen2.5:3b">Qwen 2.5 (3B)</option>
                        <option value="gemini-flash-latest">Gemini Flash (Fallback)</option>
                      </select>
                    </div>

                    {profileSavedMsg && (
                      <div style={{ fontSize: '10px', color: '#85e2b8', padding: '6px 10px', background: '#133327', borderRadius: '6px' }}>
                        {profileSavedMsg}
                      </div>
                    )}

                    <button type="submit" className="primary-btn" style={{ marginTop: '10px' }}>
                      <Save size={14} /> Save Profile Settings
                    </button>
                  </form>
                </div>
              </div>
            </>
          )}
        </div>

        <footer>
          © 2026 PrepPilot <span>Built to help you study smarter.</span>
          <span className="footer-demo">FASTAPI HYBRID SEARCH &amp; REACT INTEGRATED</span>
        </footer>
      </main>
    </div>
  )
}

function Stat({ icon: Icon, label, value, sub, trend }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{label}</span>
        <div className="stat-icon"><Icon size={17} /></div>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-sub">{sub}</div>
      <div className="stat-trend">↗ {trend}</div>
    </div>
  )
}

function PageTitle({ title, desc }) {
  return (
    <div className="page-title">
      <div>
        <div className="eyebrow">YOUR LEARNING SPACE</div>
        <h1>{title}</h1>
        <p>{desc}</p>
      </div>
    </div>
  )
}

export default App
