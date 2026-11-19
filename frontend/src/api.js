// Shared PrepPilot API client
export const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  })
  const text = await res.text()
  let data
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!res.ok) {
    const errorMsg = data?.detail || data?.message || (typeof data === 'string' ? data : `Request failed (${res.status})`)
    const err = new Error(errorMsg)
    err.status = res.status
    err.data = data
    throw err
  }
  return data
}

export async function getHealth() {
  return request('/health')
}

export async function askQuestion(question, bookId = null, topK = 5) {
  try {
    return await request('/ask', {
      method: 'POST',
      body: JSON.stringify({ question, top_k: topK, book_id: bookId || null }),
    })
  } catch (err) {
    // If LLM endpoint is not ready (e.g. 502), fall back to hybrid search retrieval
    if (err.status === 502 || !err.status) {
      const searchRes = await searchChunks(question, bookId, topK)
      return {
        answer: "I retrieved the most relevant textbook passages from your library for this topic (local LLM generation is being configured):",
        cited: searchRes.map((c) => c.rank),
        sources: searchRes,
        used_llm: false,
        fallback: true,
      }
    }
    throw err
  }
}

export async function searchChunks(question, bookId = null, topK = 5) {
  return request('/search', {
    method: 'POST',
    body: JSON.stringify({ question, top_k: topK, book_id: bookId || null }),
  })
}

export async function getChapters(bookId) {
  return request(`/books/${encodeURIComponent(bookId)}/chapters`)
}

export async function getQuestions(bookId = null, chapter = null, qtype = null, limit = 20) {
  const params = new URLSearchParams()
  if (bookId) params.append('book_id', bookId)
  if (chapter) params.append('chapter', chapter)
  if (qtype) params.append('qtype', qtype)
  params.append('limit', String(limit))
  return request(`/questions?${params.toString()}`)
}

export async function generateQuestions(bookId, chapter, qtype = 'mcq', count = 4, difficulty = 'medium') {
  return request('/generate/questions', {
    method: 'POST',
    body: JSON.stringify({ book_id: bookId || null, chapter: chapter ? String(chapter) : null, qtype, count, difficulty }),
  })
}

// 1. My Subjects
export async function getSubjects(studentId = 'default') {
  return request(`/subjects?student_id=${encodeURIComponent(studentId)}`)
}

export async function getSubjectDetails(subjectId, studentId = 'default') {
  return request(`/subjects/${encodeURIComponent(subjectId)}?student_id=${encodeURIComponent(studentId)}`)
}

// 2. Overview Stats
export async function getOverviewStats(studentId = 'default') {
  return request(`/overview/stats?student_id=${encodeURIComponent(studentId)}`)
}

// 3. My Progress & Topic Mastery
export async function getProgressMastery(studentId = 'default') {
  return request(`/progress/mastery?student_id=${encodeURIComponent(studentId)}`)
}

export async function recordQuizAttempt(attemptData) {
  return request('/progress/record_quiz_attempt', {
    method: 'POST',
    body: JSON.stringify(attemptData),
  })
}

export async function getQuizHistory(studentId = 'default', limit = 20) {
  return request(`/progress/history?student_id=${encodeURIComponent(studentId)}&limit=${limit}`)
}

// 4. Study Resources & Figures
export async function getStudyResources() {
  return request('/resources')
}

export async function getResourceFigures(bookId = null, limit = 12) {
  const params = new URLSearchParams()
  if (bookId) params.append('book_id', bookId)
  params.append('limit', String(limit))
  return request(`/resources/figures?${params.toString()}`)
}

// 5. Settings & Profile
export async function getStudentProfile(username = 'default') {
  return request(`/settings/profile?username=${encodeURIComponent(username)}`)
}

export async function updateStudentProfile(profileData) {
  return request('/settings/profile', {
    method: 'POST',
    body: JSON.stringify(profileData),
  })
}

export async function getSystemSettings() {
  return request('/settings/system')
}

export const BOOK_TITLES = {
  DS: 'Data Structures (Handbook)',
  C: 'Let Us C',
  CPP: 'C++ Complete Reference',
  CLRS: 'Algorithms (CLRS)',
}
