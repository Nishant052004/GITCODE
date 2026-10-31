/**
 * Codebase Intelligence Copilot - Frontend Application
 * Multi-theme client with AST chunk exploration, hybrid search telemetry,
 * in-browser repository ingestion, and interactive citation inspector.
 */
"use strict";

(function () {
  // DOM Elements
  const chatStream = document.getElementById("chat-stream");
  const askForm = document.getElementById("ask-form");
  const askInput = document.getElementById("ask-input");
  const askBtn = document.getElementById("ask-btn");
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const themeBtn = document.getElementById("btn-theme");
  const clearChatBtn = document.getElementById("btn-clear-chat");
  const reindexBtn = document.getElementById("btn-reindex");
  const reindexIcon = document.getElementById("reindex-icon");
  const toastContainer = document.getElementById("toast-container");

  // Sidebar sections
  const repoListEl = document.getElementById("sidebar-repo-list");
  const repoCountBadge = document.getElementById("repo-count-badge");
  const suggestionsEl = document.getElementById("sidebar-suggestions");
  const fileListEl = document.getElementById("sidebar-file-list");
  const fileCountBadge = document.getElementById("file-count-badge");
  const fileSearchInput = document.getElementById("file-search-input");
  const specEmbedder = document.getElementById("spec-embedder");
  const specVector = document.getElementById("spec-vector");
  const specReranker = document.getElementById("spec-reranker");
  const specAnswerer = document.getElementById("spec-answerer");

  // Chunk Modal
  const modal = document.getElementById("modal");
  const modalTitle = document.getElementById("modal-title");
  const modalKind = document.getElementById("modal-kind");
  const modalMeta = document.getElementById("modal-meta");
  const modalCode = document.getElementById("modal-code");
  const modalClose = document.getElementById("modal-close");
  const modalCopyBtn = document.getElementById("modal-copy-btn");

  // Ingest Modal
  const openIngestBtn = document.getElementById("btn-open-ingest");
  const ingestModal = document.getElementById("ingest-modal");
  const ingestModalClose = document.getElementById("ingest-modal-close");
  const ingestCancelBtn = document.getElementById("ingest-cancel-btn");
  const ingestForm = document.getElementById("ingest-form");
  const ingestRepoInput = document.getElementById("ingest-repo-input");
  const ingestBranchInput = document.getElementById("ingest-branch-input");
  const ingestFullInput = document.getElementById("ingest-full-input");
  const ingestStatusBox = document.getElementById("ingest-status-box");
  const ingestStatusMsg = document.getElementById("ingest-status-msg");
  const ingestSubmitBtn = document.getElementById("ingest-submit-btn");

  // State
  let currentTopK = 3;
  let allFiles = [];
  let isSubmitting = false;

  // --------------------------------------------------------------------------
  // Theme Management
  // --------------------------------------------------------------------------
  const THEMES = ["cosmos", "emerald", "synthwave"];
  let activeThemeIndex = 0;

  function initTheme() {
    const saved = localStorage.getItem("copilot_theme");
    if (saved && THEMES.includes(saved)) {
      activeThemeIndex = THEMES.indexOf(saved);
    }
    applyTheme(THEMES[activeThemeIndex]);
  }

  function applyTheme(themeName) {
    document.documentElement.dataset.theme = themeName;
    localStorage.setItem("copilot_theme", themeName);
  }

  themeBtn.addEventListener("click", () => {
    activeThemeIndex = (activeThemeIndex + 1) % THEMES.length;
    const nextTheme = THEMES[activeThemeIndex];
    applyTheme(nextTheme);
    showToast(`Theme switched to ${nextTheme.charAt(0).toUpperCase() + nextTheme.slice(1)}`);
  });

  // --------------------------------------------------------------------------
  // Sidebar Navigation Toggle
  // --------------------------------------------------------------------------
  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });

  // --------------------------------------------------------------------------
  // Top-K Selection
  // --------------------------------------------------------------------------
  document.querySelectorAll(".k-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".k-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentTopK = parseInt(btn.dataset.k, 10) || 5;
    });
  });

  // --------------------------------------------------------------------------
  // Toasts
  // --------------------------------------------------------------------------
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    const icon = type === "success" ? "✓" : type === "error" ? "✕" : "ℹ";
    toast.innerHTML = `<span class="toast-icon">${icon}</span><span class="toast-msg">${escapeHtml(message)}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  // --------------------------------------------------------------------------
  // Utility & Markdown Parser
  // --------------------------------------------------------------------------
  function escapeHtml(s) {
    if (!s) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function renderMarkdown(text) {
    if (!text) return "";
    // 1. Extract fenced code blocks to prevent inner replacement
    const codeBlocks = [];
    let processed = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const idx = codeBlocks.length;
      const cleanLang = (lang || "code").toLowerCase();
      codeBlocks.push({ lang: cleanLang, code });
      return `%%CODE_BLOCK_${idx}%%`;
    });

    // 2. Escape HTML
    processed = escapeHtml(processed);

    // 3. Headers
    processed = processed.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    processed = processed.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    processed = processed.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // 4. Blockquotes
    processed = processed.replace(/^> (.*$)/gim, "<blockquote>$1</blockquote>");

    // 5. Bold & Italics
    processed = processed.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    processed = processed.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

    // 6. Inline code
    processed = processed.replace(/`([^`\n]+)`/g, "<code>$1</code>");

    // 7. Citation badges [n]
    processed = processed.replace(/\[(\d{1,2})\]/g, '<button type="button" class="cite-chip inline-cite" data-cite="$1"><span class="cite-num">[$1]</span></button>');

    // 8. Lists
    processed = processed.replace(/^\s*[-*]\s+(.*$)/gim, "<li>$1</li>");
    processed = processed.replace(/(<li>.*<\/li>)/gim, "<ul>$1</ul>");
    processed = processed.replace(/<\/ul>\s*<ul>/gim, "");

    // 9. Paragraph line breaks
    const lines = processed.split("\n\n");
    processed = lines
      .map((p) => {
        p = p.trim();
        if (!p) return "";
        if (p.startsWith("<h") || p.startsWith("<ul") || p.startsWith("<blockquote") || p.startsWith("%%CODE_BLOCK")) {
          return p;
        }
        return `<p>${p.replace(/\n/g, "<br/>")}</p>`;
      })
      .join("\n");

    // 10. Restore code blocks
    codeBlocks.forEach((block, idx) => {
      const placeholder = `%%CODE_BLOCK_${idx}%%`;
      const codeHtml = escapeHtml(block.code);
      const replacement = `
        <div class="code-fence-wrap">
          <div class="code-fence-header">
            <span>${block.lang}</span>
            <button type="button" class="code-copy-btn" data-copy="${encodeURIComponent(block.code)}">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              <span>Copy</span>
            </button>
          </div>
          <pre><code>${codeHtml}</code></pre>
        </div>
      `;
      processed = processed.replace(placeholder, replacement);
    });

    return processed;
  }

  // --------------------------------------------------------------------------
  // Message Rendering
  // --------------------------------------------------------------------------
  function addUserMessage(query) {
    const welcome = document.getElementById("welcome-card");
    if (welcome) welcome.remove();

    const wrap = document.createElement("div");
    wrap.className = "message user";
    wrap.innerHTML = `
      <div class="message-content">
        <div class="bubble">${escapeHtml(query)}</div>
      </div>
      <div class="message-avatar">You</div>
    `;
    chatStream.appendChild(wrap);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  function addPendingAssistant() {
    const wrap = document.createElement("div");
    wrap.className = "message assistant pending";
    wrap.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-content">
        <div class="bubble">
          <div class="pending-indicator">
            <div class="pulse-spinner"></div>
            <span>Analyzing code repository, computing embeddings &amp; ranking hits&hellip;</span>
          </div>
        </div>
      </div>
    `;
    chatStream.appendChild(wrap);
    chatStream.scrollTop = chatStream.scrollHeight;
    return wrap;
  }

  function addAssistantMessage(data) {
    const wrap = document.createElement("div");
    wrap.className = "message assistant";

    const contentBox = document.createElement("div");
    contentBox.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    // Answer Body
    const bodyEl = document.createElement("div");
    bodyEl.className = "assistant-body";
    bodyEl.innerHTML = renderMarkdown(data.answer);
    bubble.appendChild(bodyEl);

    // Citations Tray
    if (data.citations && data.citations.length) {
      const tray = document.createElement("div");
      tray.className = "citations-tray";
      tray.innerHTML = `<span class="citations-label">Sources Cited:</span>`;

      data.citations.forEach((c) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "cite-chip";
        chip.dataset.cite = c.index;
        chip.dataset.chunkId = c.chunk_id;
        const sym = c.symbol ? ` &bull; ${c.symbol}` : "";
        chip.innerHTML = `<span class="cite-num">[${c.index}]</span> <span>${escapeHtml(c.file_path)}:${c.start_line}-${c.end_line}${sym}</span>`;
        tray.appendChild(chip);
      });
      bubble.appendChild(tray);
    }

    // Candidates Accordion (Reranked results)
    if (data.candidates && data.candidates.length) {
      const details = document.createElement("details");
      details.className = "candidates-accordion";

      const summary = document.createElement("summary");
      summary.innerHTML = `
        <span>🔎 ${data.candidates.length} Reranked Candidate Snippets</span>
        <span class="badge">Hybrid Fusion</span>
      `;
      details.appendChild(summary);

      const list = document.createElement("div");
      list.className = "candidate-list";

      data.candidates.forEach((cand, idx) => {
        const card = document.createElement("div");
        card.className = "candidate-card";
        card.dataset.chunkId = cand.chunk_id;
        const sym = cand.symbol ? `<span class="cand-symbol">${escapeHtml(cand.symbol)}</span>` : "";
        card.innerHTML = `
          <div>
            <span class="cand-path">${escapeHtml(cand.file_path)}:${cand.start_line}-${cand.end_line}</span>
            ${sym}
          </div>
          <span class="cand-score">Score: ${(cand.score || 0).toFixed(3)}</span>
        `;
        list.appendChild(card);
      });

      details.appendChild(list);
      bubble.appendChild(details);
    }

    // Telemetry & Latencies
    if (data.timings) {
      const t = data.timings;
      const tele = document.createElement("div");
      tele.className = "telemetry-bar";
      tele.innerHTML = `
        <span class="telemetry-item">Retrieve: <strong>${t.retrieve_ms || 0}ms</strong></span> &bull;
        <span class="telemetry-item">Rerank: <strong>${t.rerank_ms || 0}ms</strong></span> &bull;
        <span class="telemetry-item">Answer: <strong>${t.answer_ms || 0}ms</strong></span> &bull;
        <span class="telemetry-item">Total: <strong>${t.total_ms || 0}ms</strong></span> &bull;
        <span class="telemetry-item">Engine: <strong>${escapeHtml(data.answerer || "extractive")}</strong></span>
      `;
      bubble.appendChild(tele);
    }

    // Bubble interaction (chunk clicks, copy buttons)
    bubble.addEventListener("click", (e) => {
      // 1. Copy code button
      const copyBtn = e.target.closest(".code-copy-btn");
      if (copyBtn) {
        const raw = decodeURIComponent(copyBtn.dataset.copy || "");
        navigator.clipboard.writeText(raw).then(() => {
          showToast("Code copied to clipboard!", "success");
          copyBtn.querySelector("span").textContent = "Copied!";
          setTimeout(() => (copyBtn.querySelector("span").textContent = "Copy"), 2000);
        });
        return;
      }

      // 2. Citation chip or candidate card click -> open chunk modal
      const target = e.target.closest(".cite-chip, .candidate-card");
      if (target) {
        let chunkId = target.dataset.chunkId;
        if (!chunkId && target.dataset.cite) {
          const match = (data.citations || []).find((c) => String(c.index) === target.dataset.cite);
          if (match) chunkId = match.chunk_id;
        }
        if (chunkId) openChunkModal(chunkId);
      }
    });

    contentBox.appendChild(bubble);
    wrap.innerHTML = `<div class="message-avatar">AI</div>`;
    wrap.appendChild(contentBox);

    chatStream.appendChild(wrap);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  // --------------------------------------------------------------------------
  // Ask Question Submission
  // --------------------------------------------------------------------------
  async function submitQuery(queryText) {
    const q = (queryText || askInput.value).trim();
    if (!q || isSubmitting) return;

    isSubmitting = true;
    askInput.value = "";
    askInput.style.height = "auto";
    askBtn.disabled = true;

    addUserMessage(q);
    const pendingMsg = addPendingAssistant();

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: currentTopK }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      pendingMsg.remove();
      addAssistantMessage(data);
    } catch (err) {
      pendingMsg.remove();
      const wrap = document.createElement("div");
      wrap.className = "message assistant";
      wrap.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
          <div class="bubble" style="border-color: var(--danger);">
            <strong style="color: var(--danger);">Query Error:</strong>
            <p style="margin-top: 0.4rem;">${escapeHtml(err.message)}</p>
          </div>
        </div>
      `;
      chatStream.appendChild(wrap);
      showToast(err.message, "error");
    } finally {
      isSubmitting = false;
      askBtn.disabled = false;
      askInput.focus();
    }
  }

  askForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitQuery();
  });

  askInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  });

  // Auto-resize textarea
  askInput.addEventListener("input", () => {
    askInput.style.height = "auto";
    askInput.style.height = Math.min(askInput.scrollHeight, 140) + "px";
  });

  // Global keyboard shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== askInput && !modal.classList.contains("hidden") === false) {
      e.preventDefault();
      askInput.focus();
    }
    if (e.key === "Escape") {
      closeAllModals();
    }
  });

  // Quick card clicks in welcome hero
  chatStream.addEventListener("click", (e) => {
    const card = e.target.closest(".quick-card");
    if (card && card.dataset.query) {
      submitQuery(card.dataset.query);
    }
  });

  // Clear Chat
  clearChatBtn.addEventListener("click", () => {
    chatStream.innerHTML = `
      <div class="welcome-card" id="welcome-card">
        <div class="welcome-badge"><span class="welcome-spark">✦</span> Hybrid Lexical &bull; Vector &bull; Cross-Encoder</div>
        <h2>Explore &amp; Ask Anything About Your Codebase</h2>
        <p class="welcome-desc">
          Ingest any repository, chunk code semantically, and ask questions with exact file, line, and author citations.
        </p>
        <div class="quick-cards-grid">
          <div class="quick-card" data-query="How do we verify a user's password?">
            <div class="quick-card-icon">🔐</div>
            <div class="quick-card-body"><strong>Password Verification</strong><span>Authentication &amp; verification logic</span></div>
          </div>
          <div class="quick-card" data-query="Where and how are invoices charged?">
            <div class="quick-card-icon">💳</div>
            <div class="quick-card-body"><strong>Payment Pipeline</strong><span>Billing &amp; invoice handlers</span></div>
          </div>
          <div class="quick-card" data-query="Show the database storage implementation and interface">
            <div class="quick-card-icon">💾</div>
            <div class="quick-card-body"><strong>Data Storage Layer</strong><span>Persistence schema &amp; operations</span></div>
          </div>
          <div class="quick-card" data-query="Why does login fail for unknown or inactive users?">
            <div class="quick-card-icon">⚠️</div>
            <div class="quick-card-body"><strong>Auth Error Handling</strong><span>Exception types &amp; status codes</span></div>
          </div>
        </div>
      </div>
    `;
    showToast("Chat conversation cleared");
  });

  // --------------------------------------------------------------------------
  // Chunk Modal
  // --------------------------------------------------------------------------
  function openChunkModal(chunkId) {
    modalTitle.textContent = "Loading chunk metadata...";
    modalKind.textContent = "CHUNK";
    modalMeta.innerHTML = "Fetching code from index...";
    modalCode.textContent = "";
    modal.classList.remove("hidden");

    fetch(`/chunks/${chunkId}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Chunk not found"))))
      .then((chunk) => {
        modalTitle.textContent = `${chunk.file_path}:${chunk.start_line}-${chunk.end_line}`;
        modalKind.textContent = (chunk.kind || "block").toUpperCase();

        const tags = [
          chunk.symbol ? `<span class="meta-tag">symbol: <strong>${escapeHtml(chunk.symbol)}</strong></span>` : null,
          chunk.language ? `<span class="meta-tag">lang: <strong>${escapeHtml(chunk.language)}</strong></span>` : null,
          chunk.commit_id ? `<span class="meta-tag">commit: <strong>${escapeHtml(chunk.commit_id.slice(0, 8))}</strong></span>` : `<span class="meta-tag">commit: <strong>worktree</strong></span>`,
          chunk.author ? `<span class="meta-tag">author: <strong>${escapeHtml(chunk.author)}</strong></span>` : null,
          chunk.last_modified ? `<span class="meta-tag">date: <strong>${escapeHtml(chunk.last_modified.slice(0, 10))}</strong></span>` : null,
        ].filter(Boolean);

        modalMeta.innerHTML = tags.join("&nbsp;&bull;&nbsp;");
        modalCode.textContent = chunk.text || "";
      })
      .catch((err) => {
        modalTitle.textContent = "Chunk Unavailable";
        modalMeta.textContent = String(err.message);
        modalCode.textContent = "";
      });
  }

  modalClose.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });

  modalCopyBtn.addEventListener("click", () => {
    const code = modalCode.textContent;
    if (!code) return;
    navigator.clipboard.writeText(code).then(() => {
      showToast("Snippet copied to clipboard!", "success");
    });
  });

  function closeAllModals() {
    modal.classList.add("hidden");
    ingestModal.classList.add("hidden");
  }

  // --------------------------------------------------------------------------
  // Repository Ingestion Modal
  // --------------------------------------------------------------------------
  openIngestBtn.addEventListener("click", () => {
    ingestStatusBox.classList.add("hidden");
    ingestModal.classList.remove("hidden");
  });

  ingestModalClose.addEventListener("click", () => ingestModal.classList.add("hidden"));
  ingestCancelBtn.addEventListener("click", () => ingestModal.classList.add("hidden"));
  ingestModal.addEventListener("click", (e) => {
    if (e.target === ingestModal) ingestModal.classList.add("hidden");
  });

  ingestForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const repo = ingestRepoInput.value.trim();
    const branch = ingestBranchInput.value.trim() || null;
    const full = ingestFullInput.checked;
    if (!repo) return;

    ingestStatusBox.classList.remove("hidden");
    ingestStatusMsg.textContent = "Submitting ingestion job...";
    ingestSubmitBtn.disabled = true;

    try {
      const res = await fetch("/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo, branch, full }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const job = await res.json();
      if (job.status === "done") {
        ingestStatusMsg.textContent = "Done! Repository ingested successfully.";
        showToast("Repository ingested successfully!", "success");
        setTimeout(() => {
          ingestModal.classList.add("hidden");
          refreshAll();
        }, 1200);
        return;
      }

      // Poll background job
      ingestStatusMsg.textContent = `Job ${job.job_id} running: scanning AST & embedding...`;
      pollIngestJob(job.job_id);
    } catch (err) {
      ingestStatusMsg.textContent = `Error: ${err.message}`;
      showToast(err.message, "error");
      ingestSubmitBtn.disabled = false;
    }
  });

  async function pollIngestJob(jobId) {
    const interval = setInterval(async () => {
      try {
        const r = await fetch(`/ingest/${jobId}`);
        if (!r.ok) return;
        const data = await r.json();
        if (data.status === "done") {
          clearInterval(interval);
          ingestStatusMsg.textContent = "Ingestion complete!";
          ingestSubmitBtn.disabled = false;
          showToast("Ingestion completed!", "success");
          setTimeout(() => {
            ingestModal.classList.add("hidden");
            refreshAll();
          }, 1200);
        } else if (data.status === "failed") {
          clearInterval(interval);
          ingestStatusMsg.textContent = `Ingestion failed: ${data.detail || "unknown error"}`;
          ingestSubmitBtn.disabled = false;
          showToast(`Ingest failed: ${data.detail}`, "error");
        }
      } catch {
        // continue polling
      }
    }, 1500);
  }

  // --------------------------------------------------------------------------
  // Reindex Button
  // --------------------------------------------------------------------------
  reindexBtn.addEventListener("click", async () => {
    reindexIcon.style.animation = "spin 1s linear infinite";
    reindexBtn.disabled = true;
    showToast("Reindexing changed files...", "info");

    try {
      const res = await fetch("/reindex", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const s = data.stats || {};
      showToast(
        `Reindexed: ${s.files_reindexed || 0} files, +${s.chunks_added || 0}/-${s.chunks_removed || 0} chunks`,
        "success"
      );
      refreshAll();
    } catch (err) {
      showToast(`Reindex failed: ${err.message}`, "error");
    } finally {
      reindexIcon.style.animation = "none";
      reindexBtn.disabled = false;
    }
  });

  // --------------------------------------------------------------------------
  // Status, Repositories, Files, and Suggestions
  // --------------------------------------------------------------------------
  async function refreshStatus() {
    try {
      const res = await fetch("/status");
      if (!res.ok) throw new Error("Offline");
      const s = await res.json();

      statusPill.className = "status-pill ok";
      statusText.textContent = `${s.chunks} chunks \u2022 ${s.files} files \u2022 ${s.embedder} \u2022 ${s.vector_backend} \u2022 ${s.reranker}` +
        (s.stale_files && s.stale_files.length ? ` (${s.stale_files.length} stale)` : "");

      // Pipeline specs
      specEmbedder.textContent = s.embedder || "-";
      specVector.textContent = s.vector_backend || "-";
      specReranker.textContent = s.reranker || "-";
      specAnswerer.textContent = s.answerer || "-";

      // Active repos
      renderRepoList(s.repos || []);
    } catch {
      statusPill.className = "status-pill bad";
      statusText.textContent = "Backend offline or unreachable";
    }
  }

  function renderRepoList(repos) {
    repoCountBadge.textContent = repos.length;
    if (!repos.length) {
      repoListEl.innerHTML = `<div class="empty-hint">No active repository indexed yet. Click "Ingest Repo" to begin.</div>`;
      return;
    }

    repoListEl.innerHTML = "";
    repos.forEach((r) => {
      const card = document.createElement("div");
      card.className = "repo-card";
      const head = r.head_commit ? r.head_commit.slice(0, 7) : "HEAD";
      const branch = r.branch ? ` &bull; ${escapeHtml(r.branch)}` : "";
      card.innerHTML = `
        <div class="repo-card-url">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>
          <span>${escapeHtml(r.url)}</span>
        </div>
        <div class="repo-card-meta">
          <span>commit: <code>${escapeHtml(head)}</code>${branch}</span>
        </div>
      `;
      repoListEl.appendChild(card);
    });
  }

  async function loadFiles() {
    try {
      const res = await fetch("/files");
      if (!res.ok) return;
      const data = await res.json();
      allFiles = data.files || [];
      fileCountBadge.textContent = allFiles.length;
      renderFileList(allFiles);
    } catch {
      fileListEl.innerHTML = `<div class="empty-hint">File explorer unavailable</div>`;
    }
  }

  function renderFileList(files) {
    if (!files.length) {
      fileListEl.innerHTML = `<div class="empty-hint">No files indexed</div>`;
      return;
    }

    fileListEl.innerHTML = "";
    files.forEach((f) => {
      const item = document.createElement("div");
      item.className = "file-item";
      item.innerHTML = `
        <span>${escapeHtml(f.path)}</span>
        <span class="file-lang-pill">${escapeHtml(f.language || "text")}</span>
      `;
      item.addEventListener("click", () => {
        submitQuery(`Explain the logic and responsibility of ${f.path}`);
      });
      fileListEl.appendChild(item);
    });
  }

  fileSearchInput.addEventListener("input", () => {
    const term = fileSearchInput.value.toLowerCase().trim();
    if (!term) {
      renderFileList(allFiles);
      return;
    }
    const filtered = allFiles.filter((f) => f.path.toLowerCase().includes(term));
    renderFileList(filtered);
  });

  async function loadSuggestions() {
    try {
      const res = await fetch("/suggestions");
      if (!res.ok) return;
      const data = await res.json();
      const list = data.suggestions || [];
      suggestionsEl.innerHTML = "";
      list.forEach((s) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "suggestion-chip";
        chip.textContent = s;
        chip.addEventListener("click", () => submitQuery(s));
        suggestionsEl.appendChild(chip);
      });
    } catch {
      // ignore
    }
  }

  function refreshAll() {
    refreshStatus();
    loadFiles();
    loadSuggestions();
  }

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------
  initTheme();
  refreshAll();
  setInterval(refreshStatus, 25000);
})();
