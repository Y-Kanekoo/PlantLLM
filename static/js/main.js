document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const resultCard = document.getElementById('result-card');
    const loading = document.getElementById('loading');
    const diagnosisResult = document.getElementById('diagnosis-result');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');

    // 画像プレビュー機能
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // 結果をリセット
            resultCard.classList.add('d-none');
            diagnosisResult.innerHTML = '';

            // プレビューの表示
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('d-none');
            };
            reader.readAsDataURL(file);
        } else {
            // プレビューを非表示
            previewContainer.classList.add('d-none');
            imagePreview.src = '';
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // 画像ファイルの取得
        const file = imageInput.files[0];
        if (!file) {
            alert('画像を選択してください');
            return;
        }

        // フォームデータの作成
        const formData = new FormData();
        formData.append('file', file);

        try {
            // UIの更新
            resultCard.classList.remove('d-none');
            loading.classList.remove('d-none');
            diagnosisResult.innerHTML = '';

            // APIリクエスト
            const response = await fetch('/diagnose', {
                method: 'POST',
                body: formData
            });

            // レスポンスの処理
            const data = await response.json();
            if (response.ok) {
                // 診断結果の表示
                diagnosisResult.innerHTML = `
                    <div class="diagnosis-content">
                        ${data.diagnosis.replace(/\n/g, '<br>')}
                    </div>
                `;
            } else {
                // エラーメッセージの表示
                diagnosisResult.innerHTML = `
                    <div class="alert alert-danger">
                        エラーが発生しました: ${data.detail || '不明なエラー'}
                    </div>
                `;
            }
        } catch (error) {
            // エラーメッセージの表示
            diagnosisResult.innerHTML = `
                <div class="alert alert-danger">
                    システムエラーが発生しました。しばらく待ってから再度お試しください。
                </div>
            `;
            console.error('Error:', error);
        } finally {
            // ローディング表示の終了
            loading.classList.add('d-none');
        }
    });
}); 