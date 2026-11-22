import sqlite3

conn = sqlite3.connect('preppilot.db')
c = conn.cursor()

# 1. Fix corrupted math in existing CLRS cards
c.execute("""
UPDATE flashcards
SET back = 'The divide step computes the midpoint of the sub-array and takes constant time, Θ(1).'
WHERE id = 3
""")

c.execute("""
UPDATE flashcards
SET back = 'The combine step merges two sorted sub-arrays and takes linear time, Θ(n).'
WHERE id = 5
""")

c.execute("""
UPDATE flashcards
SET back = 'The overall worst-case and average-case time complexity of merge sort is Θ(n log n).'
WHERE id = 6
""")

# 2. Check existing questions to prevent duplicates
existing_fronts = {r[0].strip().lower() for r in c.execute("SELECT front FROM flashcards").fetchall()}

seed_cards = [
    # CLRS (Algorithms)
    {
        "chunk_id": "CLRS_CHUNK_00121",
        "front": "What does Big-O notation (O) represent in algorithm analysis?",
        "back": "Big-O notation describes an asymptotic upper bound on the growth rate of a function. Formally, f(n) = O(g(n)) if there exist positive constants c and n0 such that 0 <= f(n) <= c*g(n) for all n >= n0."
    },
    {
        "chunk_id": "CLRS_CHUNK_00121",
        "front": "What does Big-Omega (Ω) and Big-Theta (Θ) notation signify?",
        "back": "Big-Omega (Ω) provides an asymptotic lower bound (f(n) >= c*g(n)), while Big-Theta (Θ) provides an asymptotically tight bound where f(n) is bounded both above and below by g(n)."
    },
    {
        "chunk_id": "CLRS_CHUNK_00015",
        "front": "What are the three core steps of the Divide-and-Conquer paradigm?",
        "back": "1. Divide: Break the problem into smaller subproblems.\n2. Conquer: Solve subproblems recursively (or solve directly if small enough).\n3. Combine: Merge the subproblem solutions into the solution for the original problem."
    },
    {
        "chunk_id": "CLRS_CHUNK_00010",
        "front": "What are the two key properties required for Dynamic Programming to apply?",
        "back": "1. Optimal Substructure: An optimal solution to the problem contains optimal solutions to subproblems.\n2. Overlapping Subproblems: A recursive algorithm revisits the same subproblems repeatedly rather than generating new ones."
    },
    {
        "chunk_id": "CLRS_CHUNK_01129",
        "front": "What is the Greedy Choice Property?",
        "back": "A globally optimal solution can be arrived at by making locally optimal (greedy) choices at each stage, without needing to reconsider previous choices or solve future subproblems first."
    },
    {
        "chunk_id": "CLRS_CHUNK_00546",
        "front": "When does Dijkstra's shortest-path algorithm fail, and what algorithm is used instead?",
        "back": "Dijkstra's algorithm fails when edges have negative weights because greedy selection cannot account for subsequent reductions. The Bellman-Ford algorithm (or Floyd-Warshall for all-pairs) is used when negative edges may be present."
    },
    {
        "chunk_id": "CLRS_CHUNK_01214",
        "front": "How do Breadth-First Search (BFS) and Depth-First Search (DFS) differ in graph traversal?",
        "back": "BFS explores vertices layer-by-layer using a FIFO Queue and finds shortest paths in unweighted graphs. DFS explores as deep as possible along each branch before backtracking using a LIFO Stack (or recursion)."
    },
    {
        "chunk_id": "CLRS_CHUNK_00923",
        "front": "What is the Binary Search Tree (BST) invariant?",
        "back": "For any node x in a BST: all keys in x's left subtree are <= x.key, and all keys in x's right subtree are >= x.key. In-order traversal of a BST visits keys in sorted ascending order."
    },

    # DS (Data Structures)
    {
        "chunk_id": "DS_CHUNK_00105",
        "front": "What is the primary difference between a Stack and a Queue?",
        "back": "A Stack follows LIFO (Last-In, First-Out) ordering (elements are pushed and popped from the top). A Queue follows FIFO (First-In, First-Out) ordering (elements are enqueued at the rear and dequeued from the front)."
    },
    {
        "chunk_id": "DS_CHUNK_00104",
        "front": "What are the trade-offs between an Array and a Singly Linked List?",
        "back": "Arrays offer O(1) random access by index and better cache locality, but insertions/deletions take O(n) and size is fixed. Linked Lists offer O(1) insertion/deletion at known node positions and dynamic sizing, but require O(n) traversal to access elements and extra memory for pointers."
    },
    {
        "chunk_id": "DS_CHUNK_00163",
        "front": "What is an AVL Tree and how does it maintain balance?",
        "back": "An AVL tree is a self-balancing binary search tree where the height difference (balance factor) between left and right subtrees of any node is at most 1. Balance is restored in O(log n) time via Single (LL, RR) or Double (LR, RL) Rotations."
    },
    {
        "chunk_id": "DS_CHUNK_00532",
        "front": "What is a Hash Collision, and what are the two main collision resolution techniques?",
        "back": "A collision occurs when two distinct keys hash to the same table index. The two primary resolution techniques are:\n1. Separate Chaining (each bucket holds a linked list/tree of entries)\n2. Open Addressing (probing empty slots via Linear Probing, Quadratic Probing, or Double Hashing)."
    },
    {
        "chunk_id": "DS_CHUNK_00118",
        "front": "What is the Heap Order Property for a Min-Heap and a Max-Heap?",
        "back": "In a Min-Heap, every parent node is <= its children, so the minimum key is always at the root. In a Max-Heap, every parent node is >= its children, so the maximum key is always at the root."
    },
    {
        "chunk_id": "DS_CHUNK_00105",
        "front": "What is a Circular Queue and what problem does it solve?",
        "back": "A circular queue connects the last position back to the first in a ring buffer using modulo arithmetic ((rear + 1) % capacity). It prevents 'false overflow' in static array queues where freed front slots cannot be reused."
    },
    {
        "chunk_id": "DS_CHUNK_00104",
        "front": "What is the time complexity of detecting a cycle in a linked list using Floyd's Tortoise and Hare algorithm?",
        "back": "Time complexity is O(n) and auxiliary space complexity is O(1). The slow pointer advances 1 step while the fast pointer advances 2 steps; if a cycle exists, they must meet within the cycle."
    },
    {
        "chunk_id": "DS_CHUNK_00532",
        "front": "What is the load factor (α) of a hash table and why is it important?",
        "back": "Load factor α = n / m (number of stored items n divided by table capacity m). When α exceeds a threshold (typically 0.70 to 0.75), collisions rise sharply and the table is dynamically resized (rehashed) to preserve O(1) average lookup."
    },

    # C Programming (Let Us C)
    {
        "chunk_id": "C_CHUNK_00010",
        "front": "What is the difference between the address-of operator (&) and the dereference operator (*) in C?",
        "back": "& (address-of) returns the memory address where a variable is stored. * (dereference) accesses or modifies the value stored at the memory address pointed to by a pointer variable."
    },
    {
        "chunk_id": "C_CHUNK_00431",
        "front": "What is the difference between malloc() and calloc() in C?",
        "back": "malloc(size) allocates an uninitialized block of raw memory leaving garbage values. calloc(num, size) allocates memory for num elements of given size and initializes all bytes to zero. Both return void* and allocate on the heap."
    },
    {
        "chunk_id": "C_CHUNK_00431",
        "front": "What is a Memory Leak and a Dangling Pointer in C?",
        "back": "A memory leak occurs when heap memory allocated with malloc/calloc is not deallocated with free() and references to it are lost. A dangling pointer points to a memory address that has already been deallocated."
    },
    {
        "chunk_id": "C_CHUNK_00024",
        "front": "What is the difference between Call by Value and Call by Reference (Address) in C?",
        "back": "Call by Value passes copies of arguments; changes made inside the function do not affect the caller's variables. Call by Reference passes pointers to variables, allowing the function to directly modify the caller's values."
    },
    {
        "chunk_id": "C_CHUNK_00010",
        "front": "What does 'array decay' mean in C?",
        "back": "In most expressions (except with sizeof and &), an array name decays (implicitly converts) into a pointer to its first element (type T[] decays to T*). Thus, arr[i] is syntactically equivalent to *(arr + i)."
    },
    {
        "chunk_id": "C_CHUNK_00010",
        "front": "What is Structure Padding in C and why does it occur?",
        "back": "Compilers insert unused alignment bytes between structure members so that multi-byte types align to memory addresses matching multiples of their size (e.g., 4-byte ints on 4-byte boundaries). This maximizes CPU memory bus throughput."
    },
    {
        "chunk_id": "C_CHUNK_00024",
        "front": "What is the purpose of the 'static' keyword when applied to local and global variables in C?",
        "back": "For local variables, 'static' preserves value across function calls and allocates storage in the data segment for the program's lifetime. For global variables or functions, 'static' limits linkage (visibility) to the current translation unit (.c file)."
    },
    {
        "chunk_id": "C_CHUNK_00065",
        "front": "What is undefined behavior in C? Give two common examples.",
        "back": "Undefined behavior means the C standard imposes no requirements; the program may crash, produce wrong results, or behave unpredictably. Common examples: accessing array out of bounds, dereferencing a NULL pointer, and dividing by zero."
    },

    # CPP (C++ Complete Reference)
    {
        "chunk_id": "CPP_CHUNK_00017",
        "front": "What are the four core pillars of Object-Oriented Programming (OOP)?",
        "back": "1. Encapsulation: Bundling data and functions with access control (private/public).\n2. Abstraction: Hiding internal implementation details behind clean interfaces.\n3. Inheritance: Reusing and extending base class properties in derived classes.\n4. Polymorphism: Processing objects of different types through a common interface."
    },
    {
        "chunk_id": "CPP_CHUNK_00045",
        "front": "What is dynamic dispatch and how do virtual functions work in C++?",
        "back": "Virtual functions enable runtime polymorphism. When a class has virtual functions, the compiler creates a vtable (virtual method table) and inserts a vptr into object instances. Calls through base pointers/references look up the derived implementation at runtime."
    },
    {
        "chunk_id": "CPP_CHUNK_00634",
        "front": "Why should a base class destructor always be declared virtual in C++?",
        "back": "If a derived object is deleted via a base class pointer without a virtual destructor, only the base class destructor executes, causing resource leaks and undefined behavior. Declaring it virtual ensures the derived destructor runs first."
    },
    {
        "chunk_id": "CPP_CHUNK_00567",
        "front": "What is the difference between a Reference and a Pointer in C++?",
        "back": "A reference is an alias to an existing object; it must be initialized upon declaration, cannot be NULL, and cannot be reseated to point to another object. Pointers hold addresses, can be NULL, can be reassigned, and support pointer arithmetic."
    },
    {
        "chunk_id": "CPP_CHUNK_00026",
        "front": "What are C++ Function and Class Templates?",
        "back": "Templates enable generic programming. They act as blueprints that allow functions or classes to operate with generic data types (e.g. template <typename T>). The compiler instantiates concrete type-specific code at compile time."
    },
    {
        "chunk_id": "CPP_CHUNK_00634",
        "front": "What is RAII (Resource Acquisition Is Initialization) in C++?",
        "back": "RAII is an idiom where resource lifecycle is bound to object lifetime: resources (heap memory, file handles, mutexes) are acquired in constructors and automatically released in destructors when scope ends (e.g., std::unique_ptr, std::lock_guard)."
    },
    {
        "chunk_id": "CPP_CHUNK_00567",
        "front": "What is the Rule of Three (and Rule of Five) in C++?",
        "back": "If a class manages a resource and explicitly defines any of:\n1. Destructor\n2. Copy Constructor\n3. Copy Assignment Operator\nit should define all three. In C++11, Move Constructor and Move Assignment Operator extend this to the Rule of Five."
    },
    {
        "chunk_id": "CPP_CHUNK_00017",
        "front": "What is the difference between std::vector and std::list in the C++ STL?",
        "back": "std::vector is a dynamic contiguous array providing O(1) random access, minimal memory overhead, and cache efficiency. std::list is a doubly-linked list providing O(1) insertions/deletions anywhere once an iterator is obtained, but has pointer overhead and poor cache locality."
    }
]

added = 0
for sc in seed_cards:
    front_clean = sc["front"].strip().lower()
    if front_clean in existing_fronts:
        continue
    c.execute(
        "INSERT INTO flashcards (topic_id, chunk_id, front, back) VALUES (NULL, ?, ?, ?)",
        (sc["chunk_id"], sc["front"].strip(), sc["back"].strip())
    )
    existing_fronts.add(front_clean)
    added += 1

conn.commit()

count = c.execute("SELECT count(*) FROM flashcards").fetchone()[0]
print(f"Successfully updated seed cards. Added: {added}. Total flashcards now: {count}")

# Breakdown by book
breakdown = c.execute("""
SELECT c.book_id, count(*)
FROM flashcards f
LEFT JOIN chunks c ON c.chunk_id = f.chunk_id
GROUP BY c.book_id
""").fetchall()
print("Breakdown by book:")
for b, cnt in breakdown:
    print(f"  {b}: {cnt} cards")
