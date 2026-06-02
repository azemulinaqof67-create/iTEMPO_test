// ═══════════════════════════════════════════════════
//   iTEMPO Admin Panel — Frontend Logic
// ═══════════════════════════════════════════════════

const API = '';  // Same origin
let authToken = localStorage.getItem('admin_token') || '';
let currentPage = 'dashboard';
let usersOffset = 0;
let logsOffset = 0;
const PAGE_SIZE = 50;
let currentLogs = [];
let allDocuments = [];
let currentCompanyFilter = 'all';
let currentExplorerPath = '';
let currentUser = null;

const PERMISSION_NAMES = {
  'view_stats': 'Дашборд (Статистика)',
  'view_logs': 'Логи запросов',
  'manage_bot_users': 'Управление пользователями',
  'view_documents': 'Просмотр документов',
  'add_documents': 'Добавление документов',
  'edit_documents': 'Редактирование документов',
  'delete_documents': 'Удаление документов',
  'apply_changes': 'Применение изменений',
  'send_broadcast': 'Рассылка сообщений',
  'manage_api_keys': 'Gemini API ключи'
};

function hasPerm(p) {
  if (!currentUser) return false;
  return currentUser.role === 'superadmin' || (currentUser.permissions || []).includes(p);
}

// Companies dict (будет загружен с сервера)
let COMPANIES = {};

async function loadCompanies() {
  try {
    const res = await fetch('/api/companies');
    if (res.ok) {
      COMPANIES = await res.json();
    }
  } catch (e) {
    console.error('Error loading companies:', e);
  }
}

// ── Auth ──────────────────────────────────────────────────────────────────

async function doLogin(e) {
  e.preventDefault();
  const user = document.getElementById('loginUsername').value;
  const pw = document.getElementById('loginPassword').value;
  const btn = document.getElementById('loginBtn');
  btn.disabled = true;
  btn.innerHTML = '<span>Вход...</span>';

  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username: user, password: pw}),
    });
    if (res.ok) {
      const data = await res.json();
      authToken = data.token;
      localStorage.setItem('admin_token', authToken);
      currentUser = data.user;
      showApp();
    } else {
      const errData = await res.json().catch(() => ({}));
      document.getElementById('loginError').textContent = errData.detail || 'Неверное имя пользователя или пароль';
      document.getElementById('loginError').classList.remove('hidden');
    }
  } catch(err) {
    document.getElementById('loginError').textContent = 'Ошибка соединения';
    document.getElementById('loginError').classList.remove('hidden');
  }
  btn.disabled = false;
  btn.innerHTML = '<span>Войти</span>';
}

async function doLogout() {
  await fetch('/api/auth/logout', {method: 'POST'});
  localStorage.removeItem('admin_token');
  authToken = '';
  currentUser = null;
  document.getElementById('mainApp').classList.add('hidden');
  document.getElementById('loginScreen').classList.remove('hidden');
}

async function checkAuth() {
  if (!authToken) return false;
  try {
    const res = await apiFetch('/api/auth/check');
    if (res && res.authenticated) {
      currentUser = res.user;
      return true;
    }
    return false;
  } catch(e) {
    return false;
  }
}

function showApp() {
  if (!currentUser) return;

  document.getElementById('loginScreen').classList.add('hidden');
  document.getElementById('mainApp').classList.remove('hidden');
  
  // 1. Управление видимостью разделов в сайдбаре
  
  const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
  
  const pagePermissions = {
    'dashboard': 'view_stats',
    'users': 'manage_bot_users',
    'logs': 'view_logs',
    'documents': 'view_documents',
    'broadcast': 'send_broadcast',
    'keys': 'manage_api_keys'
  };
  
  navItems.forEach(item => {
    const onclickAttr = item.getAttribute('onclick') || '';
    const match = onclickAttr.match(/showPage\('([^']+)'/);
    if (match) {
      const pageName = match[1];
      const reqPerm = pagePermissions[pageName];
      if (reqPerm && !hasPerm(reqPerm)) {
        item.classList.add('hidden');
      } else {
        item.classList.remove('hidden');
      }
    }
  });
  
  // Сайдбар "Администраторы" (только для superadmin)
  const navAdmins = document.getElementById('navItemAdmins');
  if (navAdmins) {
    if (currentUser.role === 'superadmin') {
      navAdmins.classList.remove('hidden');
    } else {
      navAdmins.classList.add('hidden');
    }
  }
  
  // Заполняем селекты компаниями
  populateCompanySelects();
  
  // Ограничения для локальных администраторов
  const userCompany = currentUser.company_id;
  const isRestricted = currentUser.role !== 'superadmin' && userCompany && userCompany !== 'all';
  
  if (isRestricted) {
    const selectsToLock = ['broadcastCompany'];
    selectsToLock.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.value = userCompany;
        el.disabled = true;
      }
    });
  } else {
    const selectsToLock = ['broadcastCompany'];
    selectsToLock.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.disabled = false;
      }
    });
  }

  // Скрываем кнопку добавления документов во фронтенде, если нет прав add_documents
  const btnAddDoc = document.querySelector('#pageDocuments .page-header button[onclick="showDocUpload()"]');
  if (btnAddDoc) {
    if (hasPerm('add_documents')) {
      btnAddDoc.classList.remove('hidden');
    } else {
      btnAddDoc.classList.add('hidden');
    }
  }

  // Скрываем кнопку создания папок во фронтенде, если нет прав add_documents
  const btnCreateFolder = document.getElementById('btnCreateFolder');
  if (btnCreateFolder) {
    if (hasPerm('add_documents')) {
      btnCreateFolder.classList.remove('hidden');
    } else {
      btnCreateFolder.classList.add('hidden');
    }
  }

  // Скрываем кнопку применить изменения, если нет прав apply_changes
  const btnApply = document.getElementById('btnApplyChanges');
  if (btnApply) {
    if (hasPerm('apply_changes')) {
      btnApply.classList.remove('hidden');
    } else {
      btnApply.classList.add('hidden');
    }
  }

  // Загружаем первую доступную страницу (или сохраненную из localStorage при наличии прав)
  let savedPage = localStorage.getItem('admin_current_page');
  let targetPage = 'dashboard';
  
  if (savedPage) {
    const reqPerm = pagePermissions[savedPage];
    if (savedPage === 'admins' && currentUser.role === 'superadmin') {
      targetPage = 'admins';
    } else if (reqPerm && hasPerm(reqPerm)) {
      targetPage = savedPage;
    } else {
      savedPage = null;
    }
  }
  
  if (!savedPage) {
    if (hasPerm('view_stats')) targetPage = 'dashboard';
    else if (hasPerm('manage_bot_users')) targetPage = 'users';
    else if (hasPerm('view_logs')) targetPage = 'logs';
    else if (hasPerm('view_documents')) targetPage = 'documents';
    else if (hasPerm('send_broadcast')) targetPage = 'broadcast';
    else if (hasPerm('manage_api_keys')) targetPage = 'keys';
    else if (currentUser.role === 'superadmin') targetPage = 'admins';
    else targetPage = '';
  }
  
  if (targetPage) {
    const activeNav = Array.from(document.querySelectorAll('.sidebar-nav .nav-item')).find(item => {
      const onclickAttr = item.getAttribute('onclick') || '';
      return onclickAttr.includes(`'${targetPage}'`);
    });
    showPage(targetPage, activeNav);
  }
}

// ── API Helpers ───────────────────────────────────────────────────────────

async function apiFetch(url, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    'Cookie': `admin_token=${authToken}`,
    ...(options.headers || {}),
  };
  // Добавляем токен в cookie через Authorization header
  const res = await fetch(url, {
    ...options,
    credentials: 'include',
    headers,
  });
  if (res.status === 401) {
    doLogout();
    return null;
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({error: 'Ошибка сервера'}));
    throw new Error(err.error || 'Ошибка запроса');
  }
  return res.json();
}

// ── Navigation ────────────────────────────────────────────────────────────

let hasPendingChanges = false;

function showPage(name, navEl) {
  if (currentPage === 'documents' && name !== 'documents' && hasPendingChanges) {
    const leave = confirm("Вы уверены что хотите уйти со страницы? У вас есть не сохраненные изменения. Для сохранения нажмите кнопку в правом верхнем углу.");
    if (!leave) {
      return false;
    }
  }

  // Скрываем все страницы
  document.querySelectorAll('.page').forEach(p => {
    p.classList.remove('active');
    p.classList.add('hidden');
  });
  // Снимаем активный класс
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  // Показываем нужную
  const page = document.getElementById(`page${capitalize(name)}`);
  if (page) {
    page.classList.remove('hidden');
    page.classList.add('active');
  }
  if (navEl) navEl.classList.add('active');
  currentPage = name;
  localStorage.setItem('admin_current_page', name);

  // Загрузка данных
  const loaders = {
    dashboard: loadDashboard,
    users: loadUsers,
    logs: loadLogs,
    documents: loadDocuments,
    broadcast: () => {},
    keys: loadKeys,
    admins: loadAdmins,
  };
  if (loaders[name]) loaders[name]();
  return false;
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ── Dashboard ─────────────────────────────────────────────────────────────

let weeklyChart = null;
let hourlyChart = null;

async function loadDashboard() {
  try {
    const data = await apiFetch('/api/stats');
    if (!data) return;

    // Stats cards
    animateNumber('statTotalMsg', data.total_messages || 0);
    animateNumber('statTodayMsg', data.today_messages || 0);
    animateNumber('statTotalUsers', data.total_users || 0);
    animateNumber('statActiveToday', data.active_today || 0);

    // Bot status
    updateBotStatus('tg', data.tg_status);
    updateBotStatus('max', data.max_status);

    // Charts
    renderWeeklyChart(data.daily || []);
    renderHourlyChart(data.hourly || []);

  } catch(e) {
    toast('Ошибка загрузки дашборда: ' + e.message, 'error');
  }
}

function updateBotStatus(bot, status) {
  const badge = document.getElementById(`${bot}Badge`);
  const dot = document.getElementById(`${bot}StatusDot`);
  const text = document.getElementById(`${bot}StatusText`);
  if (badge) {
    badge.textContent = status === 'online' ? '🟢 Работает' : '🔴 Офлайн';
    badge.className = `status-badge ${status}`;
  }
  if (dot) {
    dot.className = `status-dot ${status}`;
  }
}

function animateNumber(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = parseInt(el.textContent) || 0;
  const diff = target - start;
  const steps = 30;
  let step = 0;
  const timer = setInterval(() => {
    step++;
    el.textContent = Math.round(start + diff * (step / steps)).toLocaleString('ru');
    if (step >= steps) clearInterval(timer);
  }, 16);
}

function renderWeeklyChart(daily) {
  const ctx = document.getElementById('weeklyChart');
  if (!ctx) return;
  if (weeklyChart) weeklyChart.destroy();

  const labels = daily.map(d => {
    const dt = new Date(d.day);
    return dt.toLocaleDateString('ru', {day: 'numeric', month: 'short'});
  });
  const values = daily.map(d => d.count);

  weeklyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: 'rgba(79,156,249,0.5)',
        borderColor: 'rgba(79,156,249,1)',
        borderWidth: 2,
        borderRadius: 4,
      }]
    },
    options: getChartOptions('Запросов'),
  });
}

function renderHourlyChart(hourly) {
  const ctx = document.getElementById('hourlyChart');
  if (!ctx) return;
  if (hourlyChart) hourlyChart.destroy();

  const labels = hourly.map(h => {
    const dt = new Date(h.ts * 1000);
    return dt.toLocaleTimeString('ru', {hour: '2-digit', minute: '2-digit'});
  });
  const values = hourly.map(h => h.count);

  hourlyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: '#a78bfa',
        backgroundColor: 'rgba(167,139,250,0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: '#a78bfa',
      }]
    },
    options: getChartOptions('Запросов'),
  });
}

function getChartOptions(label) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1a2035',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#4a5568', font: { size: 11 } }
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#4a5568', font: { size: 11 } },
        beginAtZero: true,
      }
    }
  };
}

// ── Users ─────────────────────────────────────────────────────────────────

let allUsers = [];

async function loadUsers() {
  usersOffset = 0;
  try {
    const data = await apiFetch(`/api/users?limit=200`);
    if (!data) return;
    allUsers = data.users || [];
    renderUsersTable(allUsers);
  } catch(e) {
    document.getElementById('usersBody').innerHTML = `<tr><td colspan="6" class="loading-cell">Ошибка: ${e.message}</td></tr>`;
  }
}

function filterUsers() {
  const q = document.getElementById('userSearch').value.toLowerCase();
  const filtered = allUsers.filter(u =>
    u.user_id.toLowerCase().includes(q) ||
    (u.company_id || '').toLowerCase().includes(q) ||
    (u.company_name || '').toLowerCase().includes(q)
  );
  renderUsersTable(filtered);
}

function renderUsersTable(users) {
  const tbody = document.getElementById('usersBody');
  if (!users.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="loading-cell">Нет пользователей</td></tr>';
    return;
  }
  tbody.innerHTML = users.map(u => {
    const blocked = u.is_blocked;
    const lastActivity = u.last_activity
      ? new Date(u.last_activity * 1000).toLocaleString('ru', {day:'2-digit',month:'2-digit',year:'2-digit',hour:'2-digit',minute:'2-digit'})
      : '—';
    const platform = u.platform || '?';
    const platformTag = platform === 'telegram'
      ? `<span class="tag tag-tg">TG</span>`
      : platform === 'max'
      ? `<span class="tag tag-max">MAX</span>`
      : `<span class="tag">${platform}</span>`;

    return `
      <tr>
        <td><code style="font-size:12px">${u.user_id}</code></td>
        <td>${platformTag}</td>
        <td>${u.company_name || '—'}</td>
        <td style="font-size:12px">${lastActivity}</td>
        <td>
          ${blocked ? '<span class="tag" style="background:rgba(248,113,113,0.15);color:#f87171">🚫 Блок</span>'
                    : '<span class="tag" style="background:rgba(52,211,153,0.15);color:#34d399">✓ Активен</span>'}
        </td>
        <td>
          <div class="cell-actions">
            <button class="btn btn-ghost btn-sm" onclick="showUserModal('${u.user_id}', '${u.company_id || ''}', ${blocked})">⚙️</button>
          </div>
        </td>
      </tr>`;
  }).join('');
}

function showUserModal(userId, currentCompany, isBlocked) {
  const userCompany = currentUser.company_id;
  const isRestricted = currentUser.role !== 'superadmin' && userCompany && userCompany !== 'all';

  let companyOptions = '';
  if (isRestricted) {
    companyOptions = `<option value="${userCompany}" selected>${COMPANIES[userCompany] || userCompany}</option>`;
  } else {
    companyOptions = `<option value="">— Не выбрано —</option>` + Object.entries(COMPANIES).map(([k,v]) =>
      `<option value="${k}" ${k===currentCompany?'selected':''}>${v}</option>`
    ).join('');
  }

  const bodyHtml = `
    <div class="form-group">
      <label>Предприятие</label>
      <select class="select-input" id="modalCompany" ${isRestricted ? 'disabled' : ''}>
        ${companyOptions}
      </select>
    </div>
    <p style="font-size:13px;color:var(--text-secondary);margin-top:8px;">
      Статус: ${isBlocked ? '🚫 Заблокирован' : '✓ Активен'}
    </p>
  `;

  const footerHtml = `
    <button class="btn btn-ghost" onclick="closeModal()">Отмена</button>
    ${isBlocked
      ? `<button class="btn btn-secondary" onclick="doUnblock('${userId}')">✓ Разблокировать</button>`
      : `<button class="btn btn-danger" onclick="doBlock('${userId}')">🚫 Заблокировать</button>`
    }
    <button class="btn btn-danger btn-sm" onclick="doClearHistory('${userId}')">🗑 Очистить историю</button>
    <button class="btn btn-primary" onclick="doSetCompany('${userId}')">💾 Сохранить</button>
  `;

  openModal(`Пользователь ${userId}`, bodyHtml, footerHtml, false);
}

async function doSetCompany(userId) {
  const companyId = document.getElementById('modalCompany').value;
  try {
    await apiFetch(`/api/users/${userId}/company`, {
      method: 'POST',
      body: JSON.stringify({company_id: companyId}),
    });
    toast('Предприятие обновлено', 'success');
    closeModal();
    loadUsers();
  } catch(e) { toast(e.message, 'error'); }
}

async function doBlock(userId) {
  try {
    await apiFetch(`/api/users/${userId}/block`, {method: 'POST'});
    toast('Пользователь заблокирован', 'success');
    closeModal();
    loadUsers();
  } catch(e) { toast(e.message, 'error'); }
}

async function doUnblock(userId) {
  try {
    await apiFetch(`/api/users/${userId}/unblock`, {method: 'POST'});
    toast('Пользователь разблокирован', 'success');
    closeModal();
    loadUsers();
  } catch(e) { toast(e.message, 'error'); }
}

async function doClearHistory(userId) {
  if (!confirm(`Очистить историю диалога пользователя ${userId}?`)) return;
  try {
    await apiFetch(`/api/users/${userId}/history`, {method: 'DELETE'});
    toast('История очищена', 'success');
    closeModal();
  } catch(e) { toast(e.message, 'error'); }
}

// ── Logs ──────────────────────────────────────────────────────────────────

async function loadLogs() {
  logsOffset = 0;
  await searchLogs();
}

async function searchLogs() {
  const search = document.getElementById('logSearch')?.value || '';
  const platform = document.getElementById('logPlatform')?.value || '';
  const params = new URLSearchParams({limit: PAGE_SIZE, offset: logsOffset});
  if (search) params.set('search', search);
  if (platform) params.set('platform', platform);

  try {
    const data = await apiFetch(`/api/logs?${params}`);
    if (!data) return;
    currentLogs = data.logs || [];
    renderLogsTable(currentLogs);
    renderLogsPagination(data.total || 0);
  } catch(e) {
    document.getElementById('logsBody').innerHTML = `<tr><td colspan="6" class="loading-cell">Ошибка: ${e.message}</td></tr>`;
  }
}

function renderLogsTable(logs) {
  const tbody = document.getElementById('logsBody');
  if (!logs.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="loading-cell">Нет записей</td></tr>';
    return;
  }
  tbody.innerHTML = logs.map((log, index) => {
    const time = log.timestamp
      ? new Date(log.timestamp * 1000).toLocaleString('ru', {day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'})
      : '—';
    const roleTag = log.role === 'user'
      ? '<span class="tag tag-user">Пользователь</span>'
      : '<span class="tag tag-assistant">Ассистент</span>';
    const platformTag = log.platform === 'telegram'
      ? '<span class="tag tag-tg">TG</span>'
      : `<span class="tag tag-max">${log.platform}</span>`;
    const msg = escapeHtml(log.message || '').substring(0, 100) + (log.message && log.message.length > 100 ? '...' : '');
    return `
      <tr>
        <td style="font-size:11px;white-space:nowrap">${time}</td>
        <td><code style="font-size:11px">${log.session_id}</code></td>
        <td>${platformTag}</td>
        <td>${roleTag}</td>
        <td><span class="msg-preview" title="${escapeHtml(log.message||'')}">${msg}</span></td>
        <td>
          <div class="cell-actions">
            <button class="btn btn-ghost btn-sm" onclick="showLogDetailModal(${index})">👁️ Подробнее</button>
          </div>
        </td>
      </tr>`;
  }).join('');
}

function renderLogsPagination(total) {
  const pages = Math.ceil(total / PAGE_SIZE);
  const current = Math.floor(logsOffset / PAGE_SIZE);
  const el = document.getElementById('logsPagination');
  if (!el || pages <= 1) { if(el) el.innerHTML=''; return; }

  let html = `<span>${total} записей</span>`;
  html += `<button class="page-btn" onclick="logsGoPage(${Math.max(0,current-1)})">‹</button>`;
  for (let i = Math.max(0, current-2); i <= Math.min(pages-1, current+2); i++) {
    html += `<button class="page-btn ${i===current?'active':''}" onclick="logsGoPage(${i})">${i+1}</button>`;
  }
  html += `<button class="page-btn" onclick="logsGoPage(${Math.min(pages-1,current+1)})">›</button>`;
  el.innerHTML = html;
}

function logsGoPage(page) {
  logsOffset = page * PAGE_SIZE;
  searchLogs();
}

async function exportLogs() {
  window.location.href = '/api/logs/export';
}

// ── Documents ─────────────────────────────────────────────────────────────

let selectedFile = null;
let docMode = 'file';

async function loadDocuments() {
  const loading = document.getElementById('docsLoading');
  const grid = document.getElementById('docsGrid');
  if (loading) loading.classList.remove('hidden');
  if (grid) grid.innerHTML = '';

  try {
    const data = await apiFetch(`/api/documents?path=${encodeURIComponent(currentExplorerPath)}`);
    if (!data) return;
    if (loading) loading.classList.add('hidden');

    allDocuments = data.items || [];
    renderBreadcrumbs(data.breadcrumbs || []);
    buildDocFilterTabs();
    renderDocuments();
    checkPendingChanges();
  } catch(e) {
    if (loading) loading.textContent = 'Ошибка: ' + e.message;
  }
}

function renderBreadcrumbs(crumbs) {
  const container = document.getElementById('explorerBreadcrumbs');
  if (!container) return;
  
  container.innerHTML = crumbs.map((crumb, idx) => {
    const isLast = idx === crumbs.length - 1;
    if (isLast) {
      return `<span style="color: var(--text-normal); font-weight: 600;">${escapeHtml(crumb.name)}</span>`;
    }
    return `
      <span class="breadcrumb-item" style="cursor: pointer; text-decoration: underline; color: var(--accent);" onclick="navigateExplorer('${escapeHtml(crumb.path)}')">${escapeHtml(crumb.name)}</span>
      <span style="opacity: 0.5; margin: 0 4px;">/</span>
    `;
  }).join('');
}

window.navigateExplorer = function(path) {
  currentExplorerPath = path;
  loadDocuments();
};

function buildDocFilterTabs() {
  const container = document.getElementById('docCompanyFilterTabs');
  if (container) container.classList.add('hidden');
}

function filterDocsByCompany(companyId, btnEl) {
  currentCompanyFilter = companyId;
  
  const container = document.getElementById('docCompanyFilterTabs');
  if (container) {
    container.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  }
  if (btnEl) btnEl.classList.add('active');
  
  renderDocuments();
}

function renderDocuments() {
  const grid = document.getElementById('docsGrid');
  if (!grid) return;
  
  let docs = allDocuments;
  
  if (!docs.length) {
    grid.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-muted);grid-column: 1 / -1;">Папка пуста</div>';
    return;
  }
  
  const canEdit = hasPerm('edit_documents');
  const canDelete = hasPerm('delete_documents');
  
  grid.innerHTML = docs.map(doc => {
    if (doc.is_dir) {
      return `
        <div class="doc-card folder-card" style="cursor: pointer; border-color: rgba(255,255,255,0.1); transition: transform 0.2s, border-color 0.2s;" onclick="navigateExplorer('${escapeHtml(doc.path)}')" onmouseover="this.style.borderColor='var(--accent)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.1)'">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span class="company-badge common">${escapeHtml(doc.company_name || 'Папка')}</span>
          </div>
          <div class="doc-name" style="font-weight: 600; color: var(--accent); font-size: 15px;">📁 ${escapeHtml(doc.name)}</div>
          <div class="doc-meta" style="font-size:12px;color:var(--text-muted); margin-top: 4px;">Файлов: ${doc.files_count !== undefined ? doc.files_count : 0}</div>
          <div class="doc-meta" style="font-size:11px;color:var(--text-muted)">${escapeHtml(doc.path)}</div>
        </div>
      `;
    }
    
    const size = doc.size < 1024 ? `${doc.size} B` : doc.size < 1048576 ? `${(doc.size/1024).toFixed(1)} KB` : `${(doc.size/1048576).toFixed(1)} MB`;
    const modified = doc.modified ? new Date(doc.modified * 1000).toLocaleDateString('ru') : '';
    const firstSegment = doc.path.split('/')[0];
    
    return `
      <div class="doc-card">
        <div>
          <span class="company-badge">${escapeHtml(doc.company_name)}</span>
        </div>
        <div class="doc-name">📄 ${escapeHtml(doc.name)}</div>
        ${doc.title ? `<div class="doc-title" style="font-size: 13px; color: var(--accent); font-weight: 500; margin-top: -2px;">${escapeHtml(doc.title)}</div>` : ''}
        <div class="doc-meta">${size} · ${modified}</div>
        <div class="doc-meta" style="font-size:11px;color:var(--text-muted)">${escapeHtml(doc.path)}</div>
        <div class="doc-actions">
          <button class="btn btn-secondary" onclick="event.stopPropagation(); viewDocumentContent('${escapeHtml(doc.path)}')" title="Посмотреть">
            <span>👁</span><span>Посмотреть</span>
          </button>
          ${canEdit ? `
          <button class="btn btn-secondary" onclick="event.stopPropagation(); editDocument('${escapeHtml(doc.path)}', '${firstSegment}')" title="Изменить">
            <span>✏️</span><span>Изменить</span>
          </button>
          <button class="btn btn-secondary" onclick="event.stopPropagation(); showMoveDocModal('${escapeHtml(doc.path)}', '${firstSegment}')" title="Перенести">
            <span>📦</span><span>Перенести</span>
          </button>
          ` : ''}
          ${canDelete ? `
          <button class="btn btn-danger" onclick="event.stopPropagation(); deleteDocument('${escapeHtml(doc.path)}')" title="Удалить">
            <span>🗑</span><span>Удалить</span>
          </button>
          ` : ''}
        </div>
      </div>`;
  }).join('');
}

function showDocUpload() {
  document.getElementById('docUploadPanel').classList.remove('hidden');
  document.getElementById('docAiResultSection').classList.add('hidden');
  
  // Clear inputs
  document.getElementById('docTitleDraft').value = '';
  document.getElementById('docTextContent').value = '';
  document.getElementById('aiLastUpdated').value = '';
  document.getElementById('fileInput').value = '';
  document.getElementById('selectedFile').classList.add('hidden');
  document.getElementById('selectedFile').textContent = '';
  selectedFile = null;

  // Автозаполнение по текущему пути проводника
  const parts = currentExplorerPath ? currentExplorerPath.split('/') : [];
  const orgSelect = document.getElementById('uploadOrganization');
  const catSelect = document.getElementById('uploadCategory');

  if (parts.length > 0 && parts[0]) {
    const orgValue = parts[0] === 'common' ? 'shared' : parts[0];
    orgSelect.value = orgValue;
  } else {
    orgSelect.selectedIndex = 0; // Дефолт (Общие)
  }

  if (parts.length > 1 && parts[1]) {
    catSelect.value = parts[1];
  } else {
    catSelect.selectedIndex = 0; // Дефолт (Кадры)
  }
}

function hideDocUpload() {
  document.getElementById('docUploadPanel').classList.add('hidden');
  selectedFile = null;
}

function switchDocTab(mode, btn) {
  docMode = mode;
  document.querySelectorAll('#docUploadPanel .tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('docTabFile').classList.toggle('hidden', mode !== 'file');
  document.getElementById('docTabText').classList.toggle('hidden', mode !== 'text');
}

function fileSelected() {
  const input = document.getElementById('fileInput');
  selectedFile = input.files[0];
  if (selectedFile) {
    const el = document.getElementById('selectedFile');
    el.textContent = `📎 ${selectedFile.name} (${(selectedFile.size/1024).toFixed(1)} KB)`;
    el.classList.remove('hidden');
  }
}

function handleDrop(e) {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    const el = document.getElementById('selectedFile');
    el.textContent = `📎 ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
    el.classList.remove('hidden');
  }
}

async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file, 'utf-8');
  });
}

async function processDocThroughAI() {
  const btn = document.getElementById('aiProcessBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Обрабатываю...';

  try {
    const draftTitle = document.getElementById('docTitleDraft').value;
    const organization = document.getElementById('uploadOrganization').value;
    const category = document.getElementById('uploadCategory').value;

    let text = '';
    if (docMode === 'file' && selectedFile) {
      text = await readFileAsText(selectedFile);
    } else if (docMode === 'text') {
      text = document.getElementById('docTextContent').value;
    }

    if (!text.trim()) {
      toast('Введите текст или выберите файл', 'error');
      btn.disabled = false;
      btn.textContent = '🤖 Обработать через ИИ';
      return;
    }

    const payload = {
      text: text,
      draft_title: draftTitle,
      organization: organization,
      category: category
    };

    const res = await fetch('/api/generate_metadata', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({error: 'Ошибка при вызове ИИ'}));
      throw new Error(err.error || 'Ошибка при вызове ИИ');
    }

    const data = await res.json();

    // Populate AI fields
    document.getElementById('aiTitle').value = data.title || '';
    document.getElementById('aiDescription').value = data.description || '';
    document.getElementById('aiFilename').value = data.file_name || '';
    
    if (Array.isArray(data.tags)) {
      document.getElementById('aiTags').value = data.tags.join(', ');
    } else {
      document.getElementById('aiTags').value = data.tags || '';
    }

    if (Array.isArray(data.questions_answered)) {
      document.getElementById('aiQuestions').value = data.questions_answered.join('\n');
    } else {
      document.getElementById('aiQuestions').value = data.questions_answered || '';
    }

    document.getElementById('docAiResultSection').classList.remove('hidden');
    document.getElementById('docAiResultSection').scrollIntoView({ behavior: 'smooth' });
    toast('Разметка ИИ получена. Отредактируйте при необходимости.', 'success');

  } catch (e) {
    toast('Ошибка обработки: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '🤖 Обработать через ИИ';
  }
}

async function finalUploadDocument() {
  const organization = document.getElementById('uploadOrganization').value;
  const category = document.getElementById('uploadCategory').value;
  
  const title = document.getElementById('aiTitle').value;
  const description = document.getElementById('aiDescription').value;
  const file_name = document.getElementById('aiFilename').value;
  const tagsStr = document.getElementById('aiTags').value;
  const questionsStr = document.getElementById('aiQuestions').value;
  const last_updated = document.getElementById('aiLastUpdated').value;

  let text = '';
  if (docMode === 'file' && selectedFile) {
    text = await readFileAsText(selectedFile);
  } else if (docMode === 'text') {
    text = document.getElementById('docTextContent').value;
  }

  if (!text.trim()) {
    toast('Текст документа пустой', 'error');
    return;
  }

  // Parse tags and questions
  const tags = tagsStr.split(',').map(t => t.trim()).filter(t => t);
  const questions_answered = questionsStr.split('\n').map(q => q.trim()).filter(q => q);

  const payload = {
    text,
    organization,
    category,
    title,
    description,
    file_name,
    tags,
    questions_answered,
    last_updated: last_updated || null
  };

  const msgEl = document.getElementById('uploadResultMsg');
  msgEl.className = 'hint';
  msgEl.textContent = 'Сохраняю...';
  msgEl.classList.remove('hidden');

  try {
    const res = await fetch('/upload', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({error: 'Ошибка при сохранении'}));
      throw new Error(err.error || 'Ошибка при сохранении');
    }

    msgEl.textContent = 'Документ успешно сохранен на диск!';
    msgEl.className = 'hint success';
    toast('Документ сохранен!', 'success');

    setTimeout(async () => {
      hideDocUpload();
      await loadDocuments();
    }, 2000);

  } catch (e) {
    msgEl.textContent = 'Ошибка: ' + e.message;
    msgEl.className = 'hint error';
    toast('Не удалось сохранить документ: ' + e.message, 'error');
  }
}

async function deleteDocument(path) {
  if (!confirm(`Удалить документ "${path}"?\nПереиндексация потребуется.`)) return;
  try {
    await apiFetch('/api/documents', {
      method: 'DELETE',
      body: JSON.stringify({path}),
    });
    toast('Документ удалён с диска. Примените изменения для очистки векторной базы.', 'info');
    await loadDocuments();
  } catch(e) { toast(e.message, 'error'); }
}

async function viewDocumentContent(path) {
  const filename = path.split('/').pop();
  openModal(
    filename,
    '<div class="loading-cell">Загрузка содержимого...</div>',
    '<button class="btn btn-ghost" onclick="closeModal()">Закрыть</button>',
    true
  );

  try {
    const data = await apiFetch(`/api/documents/content?path=${encodeURIComponent(path)}`);
    if (!data || data.content === undefined) {
      document.getElementById('modalBody').innerHTML = '<div style="padding:20px;text-align:center;">Не удалось загрузить содержимое</div>';
      return;
    }
    
    const safeContent = escapeHtml(data.content);
    const bodyHtml = `
      <div class="detail-view">
        <div class="detail-row">
          <span class="detail-label">Путь: ${escapeHtml(path)}</span>
          <div class="detail-value-box" style="max-height: 60vh; font-family: monospace; white-space: pre-wrap;">${safeContent}</div>
        </div>
      </div>
    `;
    document.getElementById('modalBody').innerHTML = bodyHtml;
  } catch (e) {
    document.getElementById('modalBody').innerHTML = `<div style="padding:20px;text-align:center;color:var(--danger)">Ошибка: ${e.message}</div>`;
  }
}

function showMoveDocModal(path, currentCompanyId) {
  const filename = path.split('/').pop();
  
  const companyOptions = Object.entries(COMPANIES).map(([k,v]) =>
    `<option value="${k}" ${k===currentCompanyId?'selected':''}>${v}</option>`
  ).join('');

  const bodyHtml = `
    <div class="form-group">
      <label>Текущий путь</label>
      <input type="text" class="text-input" value="${escapeHtml(path)}" readonly style="opacity: 0.7;">
    </div>
    <div class="form-group">
      <label>Перенести в предприятие (или общие)</label>
      <select class="select-input" id="moveDocTargetCompany">
        <option value="" ${!currentCompanyId?'selected':''}>📁 Все предприятия (общий)</option>
        ${companyOptions}
      </select>
    </div>
  `;

  const footerHtml = `
    <button class="btn btn-ghost" onclick="closeModal()">Отмена</button>
    <button class="btn btn-primary" onclick="doMoveDocument('${escapeHtml(path)}')">💾 Перенести</button>
  `;

  openModal(`Перенос документа ${filename}`, bodyHtml, footerHtml, false);
}

async function doMoveDocument(path) {
  const targetCompanyId = document.getElementById('moveDocTargetCompany').value;
  try {
    const res = await apiFetch('/api/documents/move', {
      method: 'POST',
      body: JSON.stringify({
        path: path,
        company_id: targetCompanyId || null
      }),
    });
    toast(res.message || 'Документ успешно перенесён', 'success');
    closeModal();
    await loadDocuments();
  } catch(e) {
    toast(e.message || 'Ошибка переноса', 'error');
  }
}

async function editDocument(path, companyId) {
  try {
    const data = await apiFetch(`/api/documents/content?path=${encodeURIComponent(path)}`);
    if (!data || data.content === undefined) {
      toast('Не удалось загрузить содержимое документа', 'error');
      return;
    }

    // Открываем панель загрузки
    showDocUpload();
    
    // Переключаемся на вкладку ручного ввода текста
    const tabsContainer = document.querySelector('#docUploadPanel .tabs');
    if (tabsContainer) {
      const tabTextBtn = tabsContainer.children[1];
      if (tabTextBtn) switchDocTab('text', tabTextBtn);
    }

    // Разбираем путь
    const parts = path.split('/');
    const org = parts[0] || 'shared';
    const cat = parts[1] || 'routine';
    const filename = parts[parts.length - 1] || 'document.md';

    document.getElementById('uploadOrganization').value = org;
    document.getElementById('uploadCategory').value = cat;
    document.getElementById('docTitleDraft').value = filename.replace('.md', '').replace(/_/g, ' ');

    let text = data.content;
    let title = filename.replace('.md', '');
    let description = '';
    let tags = '';
    let questions = '';
    let lastUpdated = '';
    
    const match = text.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
    if (match) {
      const yamlStr = match[1];
      text = text.substring(match[0].length).trim();
      
      const yamlLines = yamlStr.split('\n');
      let inQuestions = false;
      let qList = [];
      
      for (let line of yamlLines) {
        line = line.trim();
        if (!line) continue;
        
        if (line.startsWith('questions_answered:')) {
          inQuestions = true;
          const inlineMatch = line.match(/questions_answered:\s*\[(.*)\]/);
          if (inlineMatch) {
            qList = inlineMatch[1].split(',').map(x => x.trim().replace(/^["']|["']$/g, ''));
            inQuestions = false;
          }
          continue;
        }
        
        if (inQuestions) {
          if (line.startsWith('-')) {
            qList.push(line.substring(1).trim().replace(/^["']|["']$/g, ''));
            continue;
          } else if (line.includes(':')) {
            inQuestions = false;
          }
        }
        
        if (line.startsWith('title:')) {
          title = line.substring(6).trim().replace(/^["']|["']$/g, '');
        } else if (line.startsWith('description:')) {
          description = line.substring(12).trim().replace(/^["']|["']$/g, '');
        } else if (line.startsWith('last_updated:')) {
          lastUpdated = line.substring(13).trim().replace(/^["']|["']$/g, '');
        } else if (line.startsWith('tags:')) {
          const tagsMatch = line.match(/tags:\s*\[(.*)\]/);
          if (tagsMatch) {
            tags = tagsMatch[1].split(',').map(x => x.trim().replace(/^["']|["']$/g, '')).join(', ');
          } else {
            tags = line.substring(5).trim().replace(/^["']|["']$/g, '');
          }
        }
      }
      if (qList.length > 0) {
        questions = qList.join('\n');
      }
    }

    document.getElementById('docTextContent').value = text;
    document.getElementById('aiTitle').value = title;
    document.getElementById('aiDescription').value = description;
    document.getElementById('aiFilename').value = filename;
    document.getElementById('aiTags').value = tags;
    document.getElementById('aiQuestions').value = questions;
    document.getElementById('aiLastUpdated').value = lastUpdated;

    document.getElementById('docAiResultSection').classList.remove('hidden');

    // Скроллим к форме
    document.getElementById('docUploadPanel').scrollIntoView({ behavior: 'smooth' });
    toast('Документ загружен в редактор', 'info');
  } catch(e) {
    toast('Ошибка загрузки: ' + e.message, 'error');
  }
}

async function checkPendingChanges() {
  try {
    const data = await apiFetch('/api/documents/pending');
    const btn = document.getElementById('btnApplyChanges');
    if (btn) {
      if (data && data.has_changes) {
        btn.disabled = false;
        hasPendingChanges = true;
      } else {
        btn.disabled = true;
        hasPendingChanges = false;
      }
    }
  } catch(e) {
    console.error('Ошибка проверки изменений:', e);
  }
}

async function applyPendingChanges() {
  try {
    const data = await apiFetch('/api/documents/pending');
    if (!data || (!data.to_index.length && !data.to_delete.length)) {
      toast('Нет изменений для применения', 'info');
      return;
    }

    // Формируем красивый HTML для списка изменений
    let html = '<div class="detail-view">';
    
    if (data.to_index.length > 0) {
      html += `
        <div class="detail-row">
          <span class="detail-label" style="color:var(--success)">📝 Добавленные / измененные файлы (${data.to_index.length}):</span>
          <div class="detail-value-box" style="max-height: 200px; font-family: monospace;">
            ${data.to_index.map(p => `• ${escapeHtml(p)}`).join('<br>')}
          </div>
        </div>
      `;
    }
    
    if (data.to_delete.length > 0) {
      html += `
        <div class="detail-row" style="margin-top: 12px;">
          <span class="detail-label" style="color:var(--danger)">🗑️ Удаленные файлы (${data.to_delete.length}):</span>
          <div class="detail-value-box" style="max-height: 200px; font-family: monospace;">
            ${data.to_delete.map(p => `• ${escapeHtml(p)}`).join('<br>')}
          </div>
        </div>
      `;
    }
    
    html += `
      <p style="font-size:13px; color:var(--text-secondary); margin-top:12px;">
        После подтверждения новые файлы будут нарезаны на фрагменты и проиндексированы в векторной базе, а удаленные — стерты из базы. Бот сразу же сможет отвечать по обновленным документам.
      </p>
    </div>`;

    const footer = `
      <button class="btn btn-ghost" onclick="closeModal()">Отмена</button>
      <button class="btn btn-success" id="btnConfirmApply" onclick="doApplyPendingChanges()">⚡ Подтвердить применение</button>
    `;

    openModal('Применение изменений в базе знаний', html, footer, false);
  } catch(e) {
    toast('Ошибка получения изменений: ' + e.message, 'error');
  }
}

async function doApplyPendingChanges() {
  const btn = document.getElementById('btnConfirmApply');
  const mainBtn = document.getElementById('btnApplyChanges');
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Применяю...';
  }
  if (mainBtn) {
    mainBtn.disabled = true;
    mainBtn.textContent = '⏳ Применяю...';
  }

  try {
    const data = await apiFetch('/api/documents/apply', { method: 'POST' });
    toast(data.message || 'Изменения успешно применены!', 'success');
    hasPendingChanges = false;
    closeModal();
    if (mainBtn) mainBtn.textContent = '⚡ Применить изменения';
    await checkPendingChanges();
    await loadDocuments();
  } catch(e) {
    toast('Ошибка применения изменений: ' + e.message, 'error');
    if (btn) {
      btn.disabled = false;
      btn.textContent = '⚡ Подтвердить применение';
    }
    if (mainBtn) {
      mainBtn.disabled = false;
      mainBtn.textContent = '⚡ Применить изменения';
    }
  }
}

// ── Broadcast ─────────────────────────────────────────────────────────────

function previewBroadcast() {
  const text = document.getElementById('broadcastText').value;
  const preview = document.getElementById('broadcastPreview');
  if (!text.trim()) {
    preview.innerHTML = '<div class="preview-placeholder">Введите текст сообщения</div>';
    return;
  }
  preview.innerHTML = `<div class="tg-preview">${text}</div>`;
}

async function sendBroadcast() {
  const text = document.getElementById('broadcastText').value;
  if (!text.trim()) { toast('Введите текст', 'error'); return; }

  const platform = document.getElementById('broadcastPlatform').value;
  const company_id = document.getElementById('broadcastCompany').value || null;
  const activeDays = document.getElementById('broadcastDays').value;

  const btn = document.getElementById('sendBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Отправляю...';

  try {
    const body = {text, platform};
    if (company_id) body.company_id = company_id;
    if (activeDays) body.active_days = parseInt(activeDays);

    const data = await apiFetch('/api/broadcast', {method: 'POST', body: JSON.stringify(body)});
    const el = document.getElementById('broadcastResult');
    el.innerHTML = `✅ Отправлено: <b>${data.sent}</b> из <b>${data.total_targeted}</b> пользователей${data.failed ? ` (ошибок: ${data.failed})` : ''}`;
    el.className = 'broadcast-result success';
    el.classList.remove('hidden');
    toast(`Рассылка завершена: ${data.sent} сообщений`, 'success');
  } catch(e) {
    const el = document.getElementById('broadcastResult');
    el.textContent = 'Ошибка: ' + e.message;
    el.className = 'broadcast-result error';
    el.classList.remove('hidden');
    toast('Ошибка рассылки', 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '📨 Отправить';
  }
}

// ── API Keys ──────────────────────────────────────────────────────────────

async function loadKeys() {
  try {
    const data = await apiFetch('/api/keys');
    if (!data) return;
    const el = document.getElementById('keysBody');
    if (!data.keys.length) {
      el.innerHTML = '<div class="loading-cell">Ключи не найдены</div>';
      return;
    }
    el.innerHTML = `<div class="keys-list">` + data.keys.map(k => `
      <div class="key-item">
        <div class="key-info">
          <span class="key-index">#${k.index + 1}</span>
          <span class="key-value">${k.masked}</span>
        </div>
        ${k.is_active ? '<span class="key-active">● Активный</span>' : ''}
      </div>
    `).join('') + `</div>`;
  } catch(e) {
    document.getElementById('keysBody').innerHTML = `<div class="loading-cell">Ошибка: ${e.message}</div>`;
  }
}

// ── Modal ─────────────────────────────────────────────────────────────────

function openModal(title, bodyHtml, footerHtml, isLarge = false) {
  const modal = document.querySelector('#modalOverlay .modal');
  if (modal) {
    if (isLarge) {
      modal.classList.add('modal-lg');
    } else {
      modal.classList.remove('modal-lg');
    }
  }
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('modalBody').innerHTML = bodyHtml;
  document.getElementById('modalFooter').innerHTML = footerHtml;
  document.getElementById('modalOverlay').classList.remove('hidden');
}

function closeModal() {
  document.getElementById('modalOverlay').classList.add('hidden');
  const modal = document.querySelector('#modalOverlay .modal');
  if (modal) modal.classList.remove('modal-lg');
}

function showLogDetailModal(index) {
  const log = currentLogs[index];
  if (!log) return;

  const time = log.timestamp
    ? new Date(log.timestamp * 1000).toLocaleString('ru')
    : '—';
  
  const platform = log.platform || '?';
  const platformTag = platform === 'telegram'
    ? `<span class="tag tag-tg">TG</span>`
    : `<span class="tag tag-max">${platform.toUpperCase()}</span>`;

  let responseHtml = '';
  if (log.metadata && log.metadata.response) {
    responseHtml = `
      <div class="detail-row" style="margin-top: 12px;">
        <span class="detail-label">Ответ ассистента</span>
        <div class="detail-value-box">${escapeHtml(log.metadata.response)}</div>
      </div>
    `;
  }

  const bodyHtml = `
    <div class="detail-view">
      <div style="display: flex; gap: 16px; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 12px;">
        <div><span class="detail-label">Время:</span> ${time}</div>
        <div><span class="detail-label">Платформа:</span> ${platformTag}</div>
        <div><span class="detail-label">Пользователь:</span> <code style="font-size:12px;">${log.session_id}</code></div>
      </div>
      <div class="detail-row">
        <span class="detail-label">Текст сообщения (${log.role === 'user' ? 'Пользователь' : 'Ассистент'})</span>
        <div class="detail-value-box">${escapeHtml(log.message)}</div>
      </div>
      ${responseHtml}
    </div>
  `;

  const footerHtml = `
    <button class="btn btn-secondary" onclick="viewUserHistory('${log.session_id}')">👤 Диалог пользователя</button>
    <button class="btn btn-ghost" onclick="closeModal()">Закрыть</button>
  `;

  openModal('Детали запроса', bodyHtml, footerHtml, true);
}

function viewUserHistory(userId) {
  closeModal();
  showPage('users');
  const searchInput = document.getElementById('userSearch');
  if (searchInput) {
    searchInput.value = userId;
    filterUsers();
  }
}

async function showUsersListModal(type) {
  const title = type === 'active' ? 'Активные сегодня пользователи' : 'Все зарегистрированные пользователи';
  openModal(title, '<div class="loading-cell">Загрузка списка пользователей...</div>', '<button class="btn btn-ghost" onclick="closeModal()">Закрыть</button>', true);

  try {
    const data = await apiFetch('/api/users?limit=1000');
    if (!data || !data.users) {
      document.getElementById('modalBody').innerHTML = '<div style="padding:20px;text-align:center;">Данные недоступны</div>';
      return;
    }

    let users = data.users;
    if (type === 'active') {
      const dayAgo = Date.now() / 1000 - 86400;
      users = users.filter(u => u.last_activity && u.last_activity >= dayAgo);
    }

    if (!users.length) {
      document.getElementById('modalBody').innerHTML = '<div style="padding:20px;text-align:center;">Нет пользователей для отображения</div>';
      return;
    }

    const rows = users.map(u => {
      const lastActivity = u.last_activity
        ? new Date(u.last_activity * 1000).toLocaleString('ru', {day:'2-digit',month:'2-digit',year:'2-digit',hour:'2-digit',minute:'2-digit'})
        : '—';
      const platform = u.platform || '?';
      const platformTag = platform === 'telegram'
        ? `<span class="tag tag-tg">TG</span>`
        : platform === 'max'
        ? `<span class="tag tag-max">MAX</span>`
        : `<span class="tag">${platform}</span>`;

      return `
        <tr>
          <td><code style="font-size:12px">${u.user_id}</code></td>
          <td>${platformTag}</td>
          <td>${u.company_name || '—'}</td>
          <td style="font-size:12px">${lastActivity}</td>
          <td>
            ${u.is_blocked ? '<span class="tag" style="background:rgba(248,113,113,0.15);color:#f87171">🚫 Блок</span>'
                           : '<span class="tag" style="background:rgba(52,211,153,0.15);color:#34d399">✓ Активен</span>'}
          </td>
        </tr>`;
    }).join('');

    const html = `
      <div class="table-wrapper" style="max-height: 50vh; overflow-y: auto;">
        <table class="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Платформа</th>
              <th>Предприятие</th>
              <th>Последняя активность</th>
              <th>Статус</th>
            </tr>
          </thead>
          <tbody>
            ${rows}
          </tbody>
        </table>
      </div>
    `;
    document.getElementById('modalBody').innerHTML = html;
  } catch (e) {
    document.getElementById('modalBody').innerHTML = `<div style="padding:20px;text-align:center;color:var(--danger)">Ошибка: ${e.message}</div>`;
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────

function toast(msg, type = 'info') {
  const container = document.getElementById('toastContainer');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateX(20px)'; }, 3000);
  setTimeout(() => el.remove(), 3300);
}

// ── Helpers ───────────────────────────────────────────────────────────────

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function populateCompanySelects() {
  const selects = ['broadcastCompany'];
  selects.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const defaultOpt = el.options[0];
    el.innerHTML = '';
    if (defaultOpt) el.appendChild(defaultOpt);
    Object.entries(COMPANIES).forEach(([k,v]) => {
      const opt = document.createElement('option');
      opt.value = k; opt.textContent = v;
      el.appendChild(opt);
    });
  });
}

// ── Admins Management ─────────────────────────────────────────────────────

let allAdmins = [];

async function loadAdmins() {
  const tbody = document.getElementById('adminsBody');
  if (tbody) tbody.innerHTML = '<tr><td colspan="6" class="loading-cell">Загрузка...</td></tr>';
  
  try {
    const data = await apiFetch('/api/admin/users');
    if (!data) return;
    allAdmins = data.users || [];
    renderAdminsTable(allAdmins);
  } catch(e) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="loading-cell">Ошибка: ${e.message}</td></tr>`;
  }
}

function renderAdminsTable(admins) {
  const tbody = document.getElementById('adminsBody');
  if (!tbody) return;
  if (!admins.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="loading-cell">Нет администраторов</td></tr>';
    return;
  }
  
  tbody.innerHTML = admins.map(a => {
    let companyName = 'Все';
    if (a.company_ids && a.company_ids.length > 0) {
      if (a.company_ids.includes('all')) {
        companyName = 'Все';
      } else {
        companyName = a.company_ids.map(cid => cid === 'common' ? 'Общие' : (COMPANIES[cid] || cid)).join(', ');
      }
    } else if (a.company_id) {
      if (a.company_id === 'all') {
        companyName = 'Все';
      } else {
        try {
          if (a.company_id.startsWith('[') && a.company_id.endsWith(']')) {
            const parsed = JSON.parse(a.company_id);
            companyName = parsed.map(cid => cid === 'common' ? 'Общие' : (COMPANIES[cid] || cid)).join(', ');
          } else {
            companyName = COMPANIES[a.company_id] || a.company_id;
          }
        } catch(e) {
          companyName = COMPANIES[a.company_id] || a.company_id;
        }
      }
    }
    const roleClass = a.role === 'superadmin' ? 'superadmin' : 'admin';
    const roleLabel = a.role === 'superadmin' ? 'Суперадмин' : 'Администратор организации';
    
    const permsList = (a.permissions || []).map(p => {
      return `<span class="permission-tag" title="${p}">${PERMISSION_NAMES[p] || p}</span>`;
    }).join('');
    
    return `
      <tr>
        <td><code>${a.id}</code></td>
        <td><b>${escapeHtml(a.username)}</b></td>
        <td><span class="role-badge ${roleClass}">${roleLabel}</span></td>
        <td>${escapeName(companyName)}</td>
        <td>${permsList || '<i style="color:var(--text-muted)">нет прав</i>'}</td>
        <td>
          <div class="cell-actions">
            <button class="btn btn-ghost btn-sm" onclick="showAdminModal(${a.id})">✏️ Изменить</button>
            ${a.username !== 'admin' ? `<button class="btn btn-danger btn-sm" onclick="deleteAdmin(${a.id}, '${escapeHtml(a.username)}')">🗑 Удалить</button>` : ''}
          </div>
        </td>
      </tr>`;
  }).join('');
}

function escapeName(name) {
  if (!name) return '—';
  return name.replace('АО ', '').replace('ООО ', '').replace('\"', '').replace('\"', '');
}

function showAdminModal(adminId = null) {
  const admin = adminId ? allAdmins.find(a => a.id === adminId) : null;
  const title = admin ? `Редактирование администратора: ${admin.username}` : 'Создание нового администратора';
  
  const roleOptions = `
    <option value="admin" ${admin && admin.role !== 'superadmin' ? 'selected' : ''}>Администратор организации</option>
    <option value="superadmin" ${admin && admin.role === 'superadmin' ? 'selected' : ''}>Суперадминистратор (полный доступ)</option>
  `;
  
  // Получаем список привязанных компаний
  let selectedCompanies = [];
  if (admin) {
    if (admin.company_ids) {
      selectedCompanies = admin.company_ids;
    } else if (admin.company_id) {
      if (admin.company_id === 'all') {
        selectedCompanies = ['all'];
      } else {
        try {
          if (admin.company_id.startsWith('[') && admin.company_id.endsWith(']')) {
            selectedCompanies = JSON.parse(admin.company_id);
          } else {
            selectedCompanies = [admin.company_id];
          }
        } catch(e) {
          selectedCompanies = [admin.company_id];
        }
      }
    }
  }

  const allChecked = selectedCompanies.includes('all') ? 'checked' : '';
  const commonChecked = selectedCompanies.includes('common') || selectedCompanies.includes('all') ? 'checked' : '';
  const commonDisabled = selectedCompanies.includes('all') ? 'disabled' : '';
  
  const companyCheckboxes = [
    `<label class="permission-item" style="grid-column: span 2; border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 4px;">
      <input type="checkbox" name="adminCompanies" value="all" ${allChecked} onchange="toggleAdminAllCompaniesCheckbox(this)">
      <b>🌍 Доступ ко всем организациям</b>
     </label>`,
    `<label class="permission-item">
      <input type="checkbox" name="adminCompanies" value="common" ${commonChecked} ${commonDisabled}>
      <span>📁 Доступ к общим файлам</span>
     </label>`
  ].concat(
    Object.entries(COMPANIES).map(([k, v]) => {
      const checked = selectedCompanies.includes(k) || selectedCompanies.includes('all') ? 'checked' : '';
      const disabled = selectedCompanies.includes('all') ? 'disabled' : '';
      return `
        <label class="permission-item">
          <input type="checkbox" name="adminCompanies" value="${k}" ${checked} ${disabled}>
          <span>${v}</span>
        </label>
      `;
    })
  ).join('');

  const permsCheckboxes = Object.entries(PERMISSION_NAMES).map(([key, name]) => {
    const checked = admin && (admin.permissions || []).includes(key) ? 'checked' : '';
    return `
      <label class="permission-item">
        <input type="checkbox" name="permissions" value="${key}" ${checked}>
        <span>${name}</span>
      </label>
    `;
  }).join('');

  const bodyHtml = `
    <form id="adminForm" onsubmit="saveAdmin(event, ${adminId})">
      <div class="form-group">
        <label>Имя пользователя (Логин)</label>
        <input type="text" id="adminUsername" class="text-input" value="${admin ? escapeHtml(admin.username) : ''}" required ${admin && admin.username === 'admin' ? 'readonly style="opacity:0.7"' : ''}>
      </div>
      <div class="form-group">
        <label>Пароль ${admin ? '(оставьте пустым для сохранения текущего)' : ''}</label>
        <input type="password" id="adminPassword" class="text-input" ${admin ? '' : 'required'}>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Роль</label>
          <select id="adminRole" class="select-input" onchange="toggleAdminRoleFields()">
            ${roleOptions}
          </select>
        </div>
      </div>
      <div class="form-group" id="adminCompanyGroup" style="grid-column: span 2;">
        <label>Доступные организации</label>
        <div class="permissions-grid">
          ${companyCheckboxes}
        </div>
      </div>
      <div class="form-group" id="adminPermissionsGroup">
        <label>Права доступа</label>
        <div class="permissions-grid">
          ${permsCheckboxes}
        </div>
      </div>
    </form>
  `;

  const footerHtml = `
    <button class="btn btn-ghost" onclick="closeModal()">Отмена</button>
    <button class="btn btn-primary" onclick="submitAdminForm()">💾 Сохранить</button>
  `;

  openModal(title, bodyHtml, footerHtml, false);
  toggleAdminRoleFields();
}

function toggleAdminAllCompaniesCheckbox(el) {
  const checkboxes = document.querySelectorAll('input[name="adminCompanies"]');
  checkboxes.forEach(cb => {
    if (cb !== el) {
      if (el.checked) {
        cb.checked = true;
        cb.disabled = true;
      } else {
        cb.disabled = false;
        cb.checked = false;
      }
    }
  });
}

function toggleAdminRoleFields() {
  const role = document.getElementById('adminRole').value;
  const companyGroup = document.getElementById('adminCompanyGroup');
  const permsGroup = document.getElementById('adminPermissionsGroup');
  
  if (role === 'superadmin') {
    if (companyGroup) companyGroup.style.display = 'none';
    if (permsGroup) permsGroup.style.display = 'none';
  } else {
    if (companyGroup) companyGroup.style.display = '';
    if (permsGroup) permsGroup.style.display = '';
  }
}

function submitAdminForm() {
  const form = document.getElementById('adminForm');
  if (form) {
    form.dispatchEvent(new Event('submit', { cancelable: true }));
  }
}

async function saveAdmin(e, adminId = null) {
  e.preventDefault();
  
  const username = document.getElementById('adminUsername').value;
  const password = document.getElementById('adminPassword').value;
  const role = document.getElementById('adminRole').value;
  
  let company_id = ['all'];
  let permissions = [];
  
  if (role !== 'superadmin') {
    const checkedCompanies = document.querySelectorAll('input[name="adminCompanies"]:checked');
    const companyIds = Array.from(checkedCompanies).map(cb => cb.value);
    
    if (companyIds.includes('all')) {
      company_id = ['all'];
    } else if (companyIds.length === 0) {
      toast('Выберите хотя бы одну организацию для администратора', 'error');
      return;
    } else {
      company_id = companyIds;
    }
    
    const checkedBoxes = document.querySelectorAll('input[name="permissions"]:checked');
    permissions = Array.from(checkedBoxes).map(cb => cb.value);
  } else {
    permissions = Object.keys(PERMISSION_NAMES);
  }
  
  const payload = {
    username,
    role,
    company_id, // отправляем массив компаний на бэкенд
    permissions
  };
  
  if (password) {
    payload.password = password;
  }

  try {
    if (adminId) {
      await apiFetch(`/api/admin/users/${adminId}`, {
        method: 'PUT',
        body: JSON.stringify(payload)
      });
      toast('Данные администратора обновлены', 'success');
    } else {
      await apiFetch('/api/admin/users', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      toast('Администратор успешно создан', 'success');
    }
    closeModal();
    loadAdmins();
  } catch(err) {
    toast('Ошибка сохранения: ' + err.message, 'error');
  }
}

async function deleteAdmin(adminId, username) {
  if (!confirm(`Вы действительно хотите удалить администратора "${username}"?`)) return;
  
  try {
    await apiFetch(`/api/admin/users/${adminId}`, {
      method: 'DELETE'
    });
    toast('Администратор удален', 'success');
    loadAdmins();
  } catch(err) {
    toast('Ошибка удаления: ' + err.message, 'error');
  }
}

window.showCreateFolderModal = function() {
  const userCompanyIds = currentUser.company_ids || [];
  const isSuper = currentUser.role === 'superadmin' || userCompanyIds.includes('all');
  
  if (!currentExplorerPath && !isSuper) {
    toast('Пожалуйста, выберите папку организации для создания подпапки', 'error');
    return;
  }
  
  const bodyHtml = `
    <div class="form-group">
      <label>Название новой папки</label>
      <input type="text" class="text-input" id="newFolderName" placeholder="Например: policies">
    </div>
  `;
  openModal(
    'Создать папку',
    bodyHtml,
    `
      <button class="btn btn-secondary" onclick="closeModal()">Отмена</button>
      <button class="btn btn-primary" onclick="submitCreateFolder()">📁 Создать</button>
    `
  );
};

window.submitCreateFolder = async function() {
  const name = document.getElementById('newFolderName').value.trim();
  if (!name) {
    toast('Введите имя папки', 'error');
    return;
  }
  
  const targetPath = currentExplorerPath ? `${currentExplorerPath}/${name}` : name;
  
  try {
    await apiFetch('/api/documents/mkdir', {
      method: 'POST',
      body: JSON.stringify({ path: targetPath })
    });
    toast('Папка создана!', 'success');
    closeModal();
    loadDocuments();
  } catch(e) {
    toast('Ошибка: ' + e.message, 'error');
  }
};

// ── Init ──────────────────────────────────────────────────────────────────

(async function init() {
  await loadCompanies();
  const authenticated = await checkAuth();
  if (authenticated) {
    showApp();
  }
  
  window.addEventListener('beforeunload', (e) => {
    if (hasPendingChanges) {
      e.preventDefault();
      e.returnValue = 'У вас есть непримененные изменения в базе знаний.';
      return e.returnValue;
    }
  });

  setInterval(() => {
    if (currentPage === 'dashboard') loadDashboard();
  }, 60000);
})();