// グローバル変数
let selectedFiles = [];
let currentDiagnosisResults = [];  // 現在の診断セッションの結果を保持

// リクエスト制御用の定数と変数
const API_CONFIG = {
    MAX_RETRIES: 3,              // 最大リトライ回数
    INITIAL_DELAY: 3000,         // 初期待機時間を3秒に短縮（1.5 Flash用）
    MAX_DELAY: 15000,            // 最大待機時間も短縮
    BACKOFF_FACTOR: 1.5,         // バックオフ係数も調整
};

let isProcessing = false;
let requestQueue = [];
let currentRetries = 0;

// DOM要素の取得
const uploadForm = document.getElementById('upload-form');
const dropZone = document.querySelector('.drop-zone');
const imageInput = document.getElementById('image-input');
const previewContainer = document.getElementById('preview-container');
const submitButton = document.querySelector('button[type="submit"]');
const cameraButton = document.getElementById('camera-button');
const cameraModal = document.getElementById('camera-modal');
const cameraPreview = document.getElementById('camera-preview');
const takePhotoButton = document.getElementById('take-photo');
const toggleViewButton = document.getElementById('toggle-view');
const saveResultButton = document.getElementById('save-result');
const shareResultButton = document.getElementById('share-result');
const historyList = document.getElementById('history-list');
const resultCard = document.getElementById('result-card');
const diagnosisResult = document.getElementById('diagnosis-result');

// ドラッグ&ドロップの処理
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
    handleFiles(files);
});

// ファイル選択の処理
imageInput.addEventListener('change', () => {
    const files = Array.from(imageInput.files);
    handleFiles(files);
});

// ファイル処理
function handleFiles(files) {
    files.forEach(file => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                addPreview(file, e.target.result);
            };
            reader.readAsDataURL(file);
            selectedFiles.push(file);
        }
    });
    updateSubmitButton();
}

// プレビューの追加
function addPreview(file, dataUrl) {
    const previewItem = document.createElement('div');
    previewItem.className = 'preview-item';
    previewItem.innerHTML = `
        <img src="${dataUrl}" alt="プレビュー">
        <button type="button" class="remove-btn">
            <i class="fas fa-times"></i>
        </button>
    `;

    previewItem.querySelector('.remove-btn').addEventListener('click', () => {
        selectedFiles = selectedFiles.filter(f => f !== file);
        previewItem.remove();
        updateSubmitButton();
    });

    previewContainer.appendChild(previewItem);
    previewContainer.classList.remove('d-none');
}

// 送信ボタンの更新
function updateSubmitButton() {
    submitButton.disabled = selectedFiles.length === 0;
}

// カメラ機能
let stream = null;

cameraButton.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraPreview.srcObject = stream;
        new bootstrap.Modal(cameraModal).show();
    } catch (err) {
        alert('カメラの起動に失敗しました: ' + err.message);
    }
});

takePhotoButton.addEventListener('click', () => {
    const canvas = document.getElementById('camera-canvas');
    canvas.width = cameraPreview.videoWidth;
    canvas.height = cameraPreview.videoHeight;
    canvas.getContext('2d').drawImage(cameraPreview, 0, 0);
    
    canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
        handleFiles([file]);
        bootstrap.Modal.getInstance(cameraModal).hide();
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    }, 'image/jpeg');
});

cameraModal.addEventListener('hidden.bs.modal', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});

// 指数バックオフによる待機時間の計算
function calculateBackoffDelay(retryCount) {
    const delay = API_CONFIG.INITIAL_DELAY * Math.pow(API_CONFIG.BACKOFF_FACTOR, retryCount);
    return Math.min(delay, API_CONFIG.MAX_DELAY);
}

// APIリクエストの実行（リトライロジック付き）
async function executeRequest(url, options, retryCount = 0) {
    try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
            if (response.status === 429) {
                if (retryCount < API_CONFIG.MAX_RETRIES) {
                    const delay = calculateBackoffDelay(retryCount);
                    console.log(`Rate limit exceeded. Retrying in ${delay/1000} seconds...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    return executeRequest(url, options, retryCount + 1);
                }
                throw new Error('APIの制限に達しました。しばらく待ってから再試行してください。');
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response;
    } catch (error) {
        console.error('Request error:', error);
        throw error;
    }
}

// エラー表示の改善
function showError(error) {
    const errorHtml = `
        <div class="alert alert-danger">
            <h6 class="alert-heading">エラーが発生しました</h6>
            <p class="mb-1">${error.message}</p>
            <hr>
            <p class="mb-0 text-muted">
                <i class="fas fa-info-circle"></i> 
                ${error.action}
            </p>
            ${error.detail ? `
                <details class="mt-2">
                    <summary class="text-muted">詳細情報</summary>
                    <small class="text-muted">${error.detail}</small>
                </details>
            ` : ''}
        </div>
    `;
    diagnosisResult.insertAdjacentHTML('beforeend', errorHtml);
}

// 診断結果の処理（結果と画像の追加表示）
async function processResult(result, file) {
    try {
        console.log('Processing new diagnosis result:', result);
        const resultCard = document.getElementById('result-card');
        const diagnosisResult = document.getElementById('diagnosis-result');
        const resultsContainer = diagnosisResult.querySelector('.diagnosis-results-container');
        
        if (!resultsContainer) {
            console.error('Results container not found');
            return;
        }
        
        // 結果表示エリアの表示
        resultCard.classList.remove('d-none');
        
        // 画像のプレビューURLを作成
        const imageUrl = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
        });
        
        // 新しい診断結果カードを作成
        const resultDiv = document.createElement('div');
        resultDiv.className = 'diagnosis-card mb-4';
        resultDiv.innerHTML = `
            <div class="card">
                <div class="row g-0">
                    <div class="col-md-4">
                        <div class="diagnosis-image-container">
                            <img src="${imageUrl}" class="img-fluid rounded diagnosis-image" alt="診断対象の画像">
                        </div>
                    </div>
                    <div class="col-md-8">
                        <div class="card-body">
                            <div class="diagnosis-content">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div class="flex-grow-1">
                                        <p class="diagnosis-text mb-2">${result.diagnosis}</p>
                                        <div class="diagnosis-meta">
                                            <small class="text-muted">
                                                使用モデル: ${result.model_info}<br>
                                                処理時間: ${result.processing_time}秒
                                            </small>
                                        </div>
                                    </div>
                                    <div class="ms-3">
                                        <small class="text-muted">${new Date().toLocaleString('ja-JP')}</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 新しい結果を追加（先頭に）
        resultsContainer.insertBefore(resultDiv, resultsContainer.firstChild);
        
        console.log('Added new result card, current results count:', resultsContainer.children.length);
        
    } catch (error) {
        console.error('Error processing diagnosis result:', error);
        showError({
            message: '診断結果の表示中にエラーが発生しました',
            action: 'ページを更新して再度お試しください',
            detail: error.message
        });
    }
}

// 診断履歴の表示を更新
async function updateHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        
        if (data.success && data.history) {
            const historyList = document.getElementById('history-list');
            if (!historyList) {
                console.error('History list element not found');
                return;
            }
            
            // 履歴を新しい順に並び替え
            const sortedHistory = data.history.sort((a, b) => b.timestamp - a.timestamp);
            
            // 履歴リストを更新
            historyList.innerHTML = sortedHistory.map(entry => `
                <div class="list-group-item">
                    <div class="d-flex align-items-start">
                        ${entry.image_path ? `
                            <img src="${entry.image_path}" class="history-thumbnail me-3" alt="診断画像">
                        ` : ''}
                        <div class="flex-grow-1">
                            <div class="d-flex justify-content-between">
                                <h6 class="mb-1">${new Date(entry.timestamp * 1000).toLocaleString('ja-JP')}</h6>
                                <small class="text-muted">${entry.model_info}</small>
                            </div>
                            <p class="mb-1">${entry.diagnosis}</p>
                            <small class="text-muted">処理時間: ${entry.processing_time}秒</small>
                        </div>
                    </div>
                </div>
            `).join('');
            
            console.log('History updated successfully');
        }
    } catch (error) {
        console.error('履歴の取得に失敗しました:', error);
        showError({
            message: '履歴の取得に失敗しました',
            action: 'ページを更新して再度お試しください',
            detail: error.message
        });
    }
}

// フォームの送信処理を修正
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isProcessing) {
        alert('他の画像を処理中です。しばらくお待ちください。');
        return;
    }

    const loadingDiv = document.getElementById('loading');
    const progressBar = document.querySelector('.progress-bar');
    const resultCard = document.getElementById('result-card');
    const diagnosisResult = document.getElementById('diagnosis-result');
    let successCount = 0;
    let errorCount = 0;

    // 新しい診断セッション開始時の状態をログ
    console.log('Starting new diagnosis session');
    console.log('Selected files count:', selectedFiles.length);
    
    // 結果表示エリアの初期化（毎回）
    diagnosisResult.innerHTML = '';
    const resultsContainer = document.createElement('div');
    resultsContainer.className = 'diagnosis-results-container';
    diagnosisResult.appendChild(resultsContainer);
    
    loadingDiv.classList.remove('d-none');
    resultCard.classList.remove('d-none');
    
    try {
        isProcessing = true;
        const selectedModel = document.getElementById('model-select').value;
        const filesToProcess = [...selectedFiles]; // 処理対象のファイルをコピー
        
        // 入力をクリア（先に実行）
        clearInputs();
        
        for (let i = 0; i < filesToProcess.length; i++) {
            console.log(`Processing file ${i + 1} of ${filesToProcess.length}`);
            
            const file = filesToProcess[i];
            const formData = new FormData();
            formData.append('file', file);
            
            const queryParams = selectedModel ? `?model_name=${selectedModel}` : '';
            const progress = ((i + 1) / filesToProcess.length) * 100;
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);

            try {
                const response = await executeRequest('/diagnose' + queryParams, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.error) {
                    errorCount++;
                    showError(result);
                } else {
                    successCount++;
                    await processResult(result, file);
                }

                // 各診断後に履歴を更新
                await updateHistory();

                if (i < filesToProcess.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            } catch (error) {
                errorCount++;
                showError({
                    message: 'リクエストの処理中にエラーが発生しました',
                    action: 'ネットワーク接続を確認して、再度お試しください',
                    detail: error.message
                });
            }
        }
    } finally {
        isProcessing = false;
        loadingDiv.classList.add('d-none');
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        updateStatusMessage(successCount, errorCount);
    }
});

// 入力のクリア処理を関数化
function clearInputs() {
    // 選択されたファイルをクリア
    selectedFiles = [];
    
    // ファイル入力をリセット
    const imageInput = document.getElementById('image-input');
    imageInput.value = '';
    
    // プレビューをクリア
    const previewContainer = document.getElementById('preview-container');
    previewContainer.innerHTML = '';
    previewContainer.classList.add('d-none');
    
    // 送信ボタンの状態を更新
    updateSubmitButton();
}

// ステータスメッセージの更新（結果は保持）
function updateStatusMessage(successCount, errorCount) {
    const diagnosisResult = document.getElementById('diagnosis-result');
    const resultsContainer = diagnosisResult.querySelector('.diagnosis-results-container');
    
    // 処理結果のメッセージを作成
    const statusMessage = document.createElement('div');
    statusMessage.className = `alert ${errorCount === 0 ? 'alert-success' : 'alert-warning'} mb-4`;
    statusMessage.innerHTML = `
        <h6>処理完了</h6>
        <p class="mb-0">
            成功: ${successCount}件 / エラー: ${errorCount}件
        </p>
    `;
    
    // 新しいステータスメッセージを追加
    diagnosisResult.insertBefore(statusMessage, resultsContainer);
}

// 履歴の削除
document.getElementById('clear-history').addEventListener('click', async () => {
    if (confirm('診断履歴を削除してもよろしいですか？')) {
        try {
            const response = await fetch('/clear-history', { method: 'POST' });
            if (response.ok) {
                await updateHistory();
            }
        } catch (error) {
            console.error('履歴の削除に失敗しました:', error);
        }
    }
});

// 履歴タブが表示された時に履歴を更新
document.addEventListener('DOMContentLoaded', () => {
    const historyTab = document.querySelector('[data-bs-target="#history-tab"]');
    if (historyTab) {
        historyTab.addEventListener('shown.bs.tab', () => {
            updateHistory();
        });
    }
});

// 表示切替
let isDetailedView = true;
toggleViewButton.addEventListener('click', () => {
    isDetailedView = !isDetailedView;
    const diagnosisContents = document.querySelectorAll('.diagnosis-content');
    diagnosisContents.forEach(content => {
        if (isDetailedView) {
            content.style.maxHeight = 'none';
        } else {
            content.style.maxHeight = '100px';
            content.style.overflow = 'hidden';
        }
    });
    toggleViewButton.innerHTML = isDetailedView ? 
        '<i class="fas fa-list"></i> 簡易表示' : 
        '<i class="fas fa-list"></i> 詳細表示';
});

// 結果の保存
saveResultButton.addEventListener('click', () => {
    const resultText = Array.from(document.querySelectorAll('.diagnosis-card'))
        .map(card => card.textContent.trim())
        .join('\n\n');
    
    const blob = new Blob([resultText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `診断結果_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// 結果の共有
shareResultButton.addEventListener('click', async () => {
    if (navigator.share) {
        try {
            await navigator.share({
                title: '植物病気診断結果',
                text: Array.from(document.querySelectorAll('.diagnosis-card'))
                    .map(card => card.textContent.trim())
                    .join('\n\n'),
            });
        } catch (err) {
            console.error('共有に失敗しました:', err);
        }
    } else {
        alert('お使いのブラウザは共有機能に対応していません。');
    }
}); 