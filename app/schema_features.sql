CREATE TABLE IF NOT EXISTS syllabus_topics (
  id INTEGER PRIMARY KEY, subject TEXT, unit_no INT, unit_title TEXT,
  name TEXT, weight REAL DEFAULT 1.0);

CREATE TABLE IF NOT EXISTS topic_chunks (
  topic_id INT, chunk_id TEXT, score REAL, PRIMARY KEY (topic_id, chunk_id));

CREATE TABLE IF NOT EXISTS topic_prereqs (
  topic_id INT, prereq_id INT, strength REAL, PRIMARY KEY (topic_id, prereq_id));

CREATE TABLE IF NOT EXISTS question_bank (
  id INTEGER PRIMARY KEY, topic_id INT, chunk_id TEXT, qtype TEXT,
  question TEXT, options_json TEXT, answer TEXT, explanation TEXT,
  difficulty REAL DEFAULT 0.5, marks INT DEFAULT 1,
  source TEXT, verified INT DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS flashcards (
  id INTEGER PRIMARY KEY, topic_id INT, chunk_id TEXT, front TEXT, back TEXT);

CREATE TABLE IF NOT EXISTS student_cards (
  student_id TEXT, card_id INT, ease REAL DEFAULT 2.5, interval_days INT DEFAULT 0,
  reps INT DEFAULT 0, due_date TEXT, PRIMARY KEY (student_id, card_id));

CREATE TABLE IF NOT EXISTS mock_exams (
  id INTEGER PRIMARY KEY, student_id TEXT, blueprint_json TEXT,
  started_at TEXT, duration_min INT, submitted_at TEXT, score REAL);

CREATE TABLE IF NOT EXISTS mock_exam_items (
  exam_id INT, question_id INT, position INT, answer_given TEXT, marks_awarded REAL);