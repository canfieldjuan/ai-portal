// frontend/script.js (v3.0 - Unified)

// --- STATE MANAGEMENT & DOM ELEMENTS ---
function getSessionId() {
    let sessionId = localStorage.getItem('commandCenterSessionId');
    if (!sessionId) {
        sessionId = `session-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
        localStorage.setItem('commandCenterSessionId', sessionId);
    }
    return sessionId;
}

const dom = {
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    chatMessages: document.getElementById('chatMessages'),
    taskTypeSelect: document.getElementById('taskTypeSelect'),
    routingModeToggle: document.getElementById('routingModeToggle'),
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    fileAttachments: document.getElementById('fileAttachments')
};

let currentSessionId = getSessionId();
let attachedFiles = []; // To keep track of files to be uploaded
document.getElementById('session-status').textContent = `Session: ${currentSessionId.split('-')[2]}`;
toggleTaskTypeDropdown(); // Set initial state

// --- EVENT LISTENERS ---
dom.sendButton.addEventListener('click', handleSendMessage);
dom.messageInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } });
dom.routingModeToggle.addEventListener('change', toggleTaskTypeDropdown);

// RESTORED: File Upload Listeners
dom.dropZone.addEventListener('click', () => dom.fileInput.click());
dom.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dom.dropZone.classList.add('dragover'); });
dom.dropZone.addEventListener('dragleave', () => dom.dropZone.classList.remove('dragover'));
dom.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) handleFiles(files);
});
dom.fileInput.addEventListener('change', () => { if (dom.fileInput.files.length) handleFiles(dom.fileInput.files); });

// --- UI HELPER FUNCTIONS ---
function toggleTaskTypeDropdown() { dom.taskTypeSelect.disabled = dom.routingModeToggle.checked; }
function addMessage(content, type, meta = {}) {
    // ... (same as previous version)
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    let metaText = type === 'user' ? 'You' : (meta.model || 'AI');
    metaText += ` • ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    if (meta.reasoning) metaText += ` • ${meta.reasoning}`;
    if (meta.cached) metaText += ` (Cached)`;
    metaDiv.textContent = metaText;
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(metaDiv);
    dom.chatMessages.appendChild(messageDiv);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    return messageDiv;
}
function updateStatusBar(data) { /* ... same as previous ... */ }

// --- RESTORED: File Handling Logic ---
function handleFiles(files) {
    for (const file of files) {
        if (!attachedFiles.some(f => f.name === file.name)) {
            attachedFiles.push(file);
        }
    }
    renderAttachments();
}

function renderAttachments() {
    dom.fileAttachments.innerHTML = '';
    attachedFiles.forEach((file, index) => {
        const attachmentEl = document.createElement('div');
        attachmentEl.className = 'file-attachment';
        attachmentEl.textContent = `📄 ${file.name}`;
        const removeBtn = document.createElement('button');
        removeBtn.innerHTML = '&times;';
        removeBtn.onclick = () => {
            attachedFiles.splice(index, 1);
            renderAttachments();
        };
        attachmentEl.appendChild(removeBtn);
        dom.fileAttachments.appendChild(attachmentEl);
    });
}

async function uploadFiles() {
    if (attachedFiles.length === 0) return true; // No files to upload
    addMessage(`Uploading ${attachedFiles.length} file(s)...`, 'ai', { model: 'System' });
    const formData = new FormData();
    for (const file of attachedFiles) {
        formData.append('file', file); // The backend expects the key to be 'file'
        try {
            const response = await fetch('/upload-file', { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Upload failed for ${file.name}`);
            const result = await response.json();
            addMessage(`✅ Successfully uploaded ${result.filename}`, 'ai', { model: 'System' });
        } catch (error) {
            addMessage(`❌ Error uploading ${file.name}: ${error.message}`, 'ai', { model: 'System Error' });
            return false;
        }
    }
    attachedFiles = []; // Clear files after upload
    renderAttachments();
    return true;
}

// --- UNIFIED CORE SEND LOGIC ---
async function handleSendMessage() {
    const message = dom.messageInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    dom.messageInput.value = '';

    // First, handle any pending file uploads
    const filesUploaded = await uploadFiles();
    if (!filesUploaded) return; // Stop if uploads failed

    // Check for slash commands
    if (message.startsWith('/')) {
        handleSlashCommand(message);
    } else {
        handleStandardChat(message);
    }
}

async function handleSlashCommand(message) {
    const [command, ...args] = message.substring(1).split(' ');
    const taskDescription = args.join(' ');
    let endpoint = '';

    switch (command.toLowerCase()) {
        case 'develop':
            endpoint = '/develop-feature';
            break;
        case 'delegate-marketing':
            endpoint = '/delegate/marketing-copy';
            break;
        default:
            addMessage(`Unknown command: ${command}`, 'ai', { model: 'System Error' });
            return;
    }

    addMessage(`Executing command: ${command}...`, 'ai', { model: 'System' });
    try {
        const formData = new FormData();
        formData.append('task_description', taskDescription); // Match Pydantic model
        if(endpoint === '/delegate/marketing-copy') {
             formData.set('topic', taskDescription); // Match endpoint signature
             formData.delete('task_description');
        }

        const response = await fetch(endpoint, { method: 'POST', body: new URLSearchParams(formData) });
        if (!response.ok) throw new Error(`Command failed with status ${response.status}`);
        const result = await response.json();
        addMessage(`✅ Command successful: ${JSON.stringify(result)}`, 'ai', { model: 'Agent Response' });
    } catch (error) {
        addMessage(`❌ Command failed: ${error.message}`, 'ai', { model: 'System Error' });
    }
}

async function handleStandardChat(message) {
    const typingMessage = addMessage('...', 'ai', { model: 'Routing' });
    try {
        const isAutoMode = dom.routingModeToggle.checked;
        const payload = {
            message: message,
            user_id: currentSessionId,
            task_type: isAutoMode ? "auto" : dom.taskTypeSelect.value,
            user_tier: "free", // Placeholder
        };
        const response = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        typingMessage.remove();
        const data = await response.json();
        if (!response.ok) throw new Error(data.response || `HTTP error! status: ${response.status}`);
        if (data.success) {
            addMessage(data.response, 'ai', { model: data.model, reasoning: data.reasoning, cached: data.cached });
        } else {
            addMessage(data.response, 'ai', { model: 'System Error' });
        }
    } catch (error) {
        typingMessage.remove();
        addMessage(`I apologize, but I've encountered an error.\n\nDetails: ${error.message}`, 'ai', { model: 'Client Error' });
    }
}
