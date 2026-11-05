/**
 * app.js
 * Disease Prediction System - Main Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State
    const AppState = {
        currentDisease: 'heart',
        currentModel: 'ensemble',
        diseasesMeta: {},
        benchmarksData: null,
        lastPrediction: null,
        isDarkMode: false
    };

    // DOM Elements
    const elements = {
        themeToggleBtn: document.getElementById('themeToggleBtn'),
        navBtns: document.querySelectorAll('.nav-btn'),
        tabContents: document.querySelectorAll('.tab-content'),
        diseaseBtns: document.querySelectorAll('.disease-btn'),
        diseaseFormTitle: document.getElementById('diseaseFormTitle'),
        diseaseFormSub: document.getElementById('diseaseFormSub'),
        modelSelect: document.getElementById('modelSelect'),
        presetContainer: document.getElementById('presetButtonsContainer'),
        dynamicFormFields: document.getElementById('dynamicFormFields'),
        patientForm: document.getElementById('patientForm'),
        resetFormBtn: document.getElementById('resetFormBtn'),
        runDiagnosisBtn: document.getElementById('runDiagnosisBtn'),
        resultLoading: document.getElementById('resultLoading'),
        resultContent: document.getElementById('resultContent'),
        resultTimestamp: document.getElementById('resultTimestamp'),
        riskStatusBox: document.getElementById('riskStatusBox'),
        riskScoreVal: document.getElementById('riskScoreVal'),
        riskTierBadge: document.getElementById('riskTierBadge'),
        diagnosisLabel: document.getElementById('diagnosisLabel'),
        urgencyLabel: document.getElementById('urgencyLabel'),
        consensusRate: document.getElementById('consensusRate'),
        consensusGrid: document.getElementById('consensusGrid'),
        featureImpactList: document.getElementById('featureImpactList'),
        recommendationsList: document.getElementById('recommendationsList'),
        downloadPdfBtn: document.getElementById('downloadPdfBtn'),
        labDiseaseSelect: document.getElementById('labDiseaseSelect'),
        cmModelSelect: document.getElementById('cmModelSelect'),
        benchmarkTableBody: document.getElementById('benchmarkTableBody')
    };

    init();

    async function init() {
        initTheme();
        initNavigation();
        initDiseaseButtons();
        initModelSelector();
        initFormHandlers();
        window.PulseBatch.init();

        await loadDiseasesMetadata();
        switchDisease(AppState.currentDisease);
    }

    // 1. Theme Toggle
    function initTheme() {
        const savedTheme = localStorage.getItem('dps_theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('theme-dark');
            AppState.isDarkMode = true;
        }

        if (elements.themeToggleBtn) {
            elements.themeToggleBtn.addEventListener('click', () => {
                AppState.isDarkMode = !AppState.isDarkMode;
                if (AppState.isDarkMode) {
                    document.body.classList.add('theme-dark');
                    localStorage.setItem('dps_theme', 'dark');
                } else {
                    document.body.classList.remove('theme-dark');
                    localStorage.setItem('dps_theme', 'light');
                }
            });
        }
    }

    // 2. Navigation Tabs
    function initNavigation() {
        elements.navBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.dataset.tab;

                elements.navBtns.forEach(b => b.classList.remove('active'));
                elements.tabContents.forEach(c => c.classList.remove('active'));

                btn.classList.add('active');
                const targetContent = document.getElementById(`tab-${targetTab}`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }

                if (targetTab === 'compare') {
                    loadAndRenderBenchmarks();
                }
            });
        });
    }

    // 3. Disease Switching
    function initDiseaseButtons() {
        elements.diseaseBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                elements.diseaseBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const disease = btn.dataset.disease;
                switchDisease(disease);
            });
        });

        if (elements.labDiseaseSelect) {
            elements.labDiseaseSelect.addEventListener('change', () => {
                renderBenchmarksForDisease(elements.labDiseaseSelect.value);
            });
        }

        if (elements.cmModelSelect) {
            elements.cmModelSelect.addEventListener('change', () => {
                renderConfusionMatrix();
            });
        }
    }

    function initModelSelector() {
        if (elements.modelSelect) {
            elements.modelSelect.addEventListener('change', () => {
                AppState.currentModel = elements.modelSelect.value;
                executeDiagnosis();
            });
        }
    }

    // 4. Fetch Disease Metadata
    async function loadDiseasesMetadata() {
        try {
            const res = await fetch('/api/diseases');
            const data = await res.json();
            if (data.status === 'success') {
                AppState.diseasesMeta = data.diseases;
            }
        } catch (err) {
            console.error('Failed to load diseases metadata:', err);
        }
    }

    function switchDisease(diseaseKey) {
        AppState.currentDisease = diseaseKey;
        const meta = AppState.diseasesMeta[diseaseKey];
        if (!meta) return;

        if (elements.diseaseFormTitle) elements.diseaseFormTitle.textContent = `${meta.name} Prediction`;
        if (elements.diseaseFormSub) elements.diseaseFormSub.textContent = meta.subheading;

        renderPresets(meta.presets);
        renderFormFields(meta.features_meta);

        if (meta.presets && meta.presets.length > 0) {
            loadPreset(meta.presets[0].id);
        }
    }

    function renderPresets(presets) {
        elements.presetContainer.innerHTML = '';
        if (!presets) return;

        presets.forEach((p, idx) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = `preset-chip ${idx === 0 ? 'active' : ''}`;
            chip.dataset.presetId = p.id;
            chip.textContent = p.name;
            chip.addEventListener('click', () => {
                document.querySelectorAll('.preset-chip').forEach(b => b.classList.remove('active'));
                chip.classList.add('active');
                loadPreset(p.id);
            });
            elements.presetContainer.appendChild(chip);
        });
    }

    function loadPreset(presetId) {
        const diseaseMeta = AppState.diseasesMeta[AppState.currentDisease];
        if (!diseaseMeta) return;

        const preset = diseaseMeta.presets.find(p => p.id === presetId);
        if (!preset) return;

        for (const [key, val] of Object.entries(preset.data)) {
            const numInput = document.getElementById(`num_${key}`);
            const sliderInput = document.getElementById(`slider_${key}`);
            const selectField = document.getElementById(`select_${key}`);

            if (numInput) numInput.value = val;
            if (sliderInput) sliderInput.value = val;
            if (selectField) selectField.value = val;
        }

        executeDiagnosis();
    }

    function renderFormFields(features) {
        elements.dynamicFormFields.innerHTML = '';
        if (!features) return;

        features.forEach(feat => {
            const groupDiv = document.createElement('div');
            groupDiv.className = 'form-group';

            if (feat.type === 'select') {
                groupDiv.innerHTML = `
                    <div class="form-group-header">
                        <label for="select_${feat.key}" class="field-label">${feat.label}</label>
                    </div>
                    <select id="select_${feat.key}" name="${feat.key}" class="form-select" style="width: 100%;">
                        ${feat.options.map(opt => `<option value="${opt.val}" ${opt.val === feat.default ? 'selected' : ''}>${opt.label}</option>`).join('')}
                    </select>
                `;
                const sel = groupDiv.querySelector('select');
                sel.addEventListener('change', () => executeDiagnosis());
            } else {
                const hintText = feat.normal ? `Normal: ${feat.normal} ${feat.unit}` : feat.unit;
                groupDiv.innerHTML = `
                    <div class="form-group-header">
                        <label for="num_${feat.key}" class="field-label">${feat.label}</label>
                        <span class="field-hint">${hintText}</span>
                    </div>
                    <div class="field-row">
                        <input type="range" id="slider_${feat.key}" class="field-slider" 
                               min="${feat.min}" max="${feat.max}" step="${feat.step}" value="${feat.default}">
                        <input type="number" id="num_${feat.key}" name="${feat.key}" class="field-number" 
                               min="${feat.min}" max="${feat.max}" step="${feat.step}" value="${feat.default}">
                    </div>
                `;

                const slider = groupDiv.querySelector('.field-slider');
                const num = groupDiv.querySelector('.field-number');

                slider.addEventListener('input', () => { num.value = slider.value; });
                slider.addEventListener('change', () => executeDiagnosis());

                num.addEventListener('input', () => { slider.value = num.value; });
                num.addEventListener('change', () => executeDiagnosis());
            }

            elements.dynamicFormFields.appendChild(groupDiv);
        });
    }

    // 5. Submit Prediction
    function initFormHandlers() {
        elements.patientForm.addEventListener('submit', (e) => {
            e.preventDefault();
            executeDiagnosis();
        });

        elements.resetFormBtn.addEventListener('click', () => {
            const meta = AppState.diseasesMeta[AppState.currentDisease];
            if (meta && meta.presets && meta.presets.length > 0) {
                loadPreset(meta.presets[0].id);
            }
        });

        elements.downloadPdfBtn.addEventListener('click', async () => {
            if (!AppState.lastPrediction) {
                alert('Please run a prediction first.');
                return;
            }
            try {
                elements.downloadPdfBtn.disabled = true;
                elements.downloadPdfBtn.textContent = 'Generating PDF...';

                const res = await fetch('/api/export-pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(AppState.lastPrediction)
                });

                if (res.ok) {
                    const blob = await res.blob();
                    const downloadUrl = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = downloadUrl;
                    a.download = `Disease_Prediction_Report_${AppState.currentDisease}_${Date.now()}.pdf`;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    window.URL.revokeObjectURL(downloadUrl);
                } else {
                    alert('Error generating PDF report.');
                }
            } catch (err) {
                alert(`Export error: ${err.message}`);
            } finally {
                elements.downloadPdfBtn.disabled = false;
                elements.downloadPdfBtn.textContent = 'Download PDF Diagnostic Report';
            }
        });
    }

    async function executeDiagnosis() {
        const formData = new FormData(elements.patientForm);
        const inputs = {};
        formData.forEach((val, key) => {
            inputs[key] = parseFloat(val);
        });

        elements.resultLoading.style.display = 'flex';
        elements.resultContent.style.opacity = '0.4';

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    disease: AppState.currentDisease,
                    model: AppState.currentModel,
                    inputs: inputs
                })
            });

            const responseData = await res.json();
            if (responseData.status === 'success') {
                AppState.lastPrediction = responseData.data;
                renderDiagnosisResult(responseData.data);
            } else {
                alert(`Inference Error: ${responseData.message}`);
            }
        } catch (err) {
            console.error('Prediction failed:', err);
        } finally {
            elements.resultLoading.style.display = 'none';
            elements.resultContent.style.opacity = '1.0';
        }
    }

    function renderDiagnosisResult(result) {
        const riskPct = result.risk_percentage;
        elements.riskScoreVal.textContent = riskPct.toFixed(1);
        elements.resultTimestamp.textContent = `Updated: ${new Date().toLocaleTimeString()}`;

        // Update status box class & tag
        elements.riskStatusBox.className = 'risk-status-box';
        elements.riskTierBadge.className = 'tag';

        if (riskPct >= 65.0) {
            elements.riskStatusBox.classList.add('status-high');
            elements.riskTierBadge.classList.add('tag-high');
            elements.riskTierBadge.textContent = 'HIGH RISK';
        } else if (riskPct >= 30.0) {
            elements.riskStatusBox.classList.add('status-moderate');
            elements.riskTierBadge.classList.add('tag-moderate');
            elements.riskTierBadge.textContent = 'MODERATE RISK';
        } else {
            elements.riskStatusBox.classList.add('status-normal');
            elements.riskTierBadge.classList.add('tag-normal');
            elements.riskTierBadge.textContent = 'LOW RISK';
        }

        elements.diagnosisLabel.textContent = result.diagnostic_label;
        elements.urgencyLabel.textContent = result.clinical_urgency;

        // Consensus Breakdown
        const consensus = result.consensus;
        elements.consensusRate.textContent = `Consensus: ${consensus.consensus_confidence}%`;
        elements.consensusGrid.innerHTML = '';

        consensus.model_breakdowns.forEach(m => {
            const card = document.createElement('div');
            card.className = 'model-stat-card';
            const color = m.probability >= 50.0 ? 'color: var(--danger);' : 'color: var(--success);';
            card.innerHTML = `
                <div class="model-name">${m.name}</div>
                <div class="model-val" style="${color}">${m.probability}%</div>
            `;
            elements.consensusGrid.appendChild(card);
        });

        // Top factors
        elements.featureImpactList.innerHTML = '';
        result.feature_contributions.slice(0, 5).forEach(c => {
            const row = document.createElement('div');
            row.className = 'factor-item';
            const barClass = c.direction === 'elevating' ? 'elevating' : (c.direction === 'protective' ? 'protective' : '');
            const sign = c.z_score >= 0 ? '+' : '';

            row.innerHTML = `
                <div class="factor-name">${c.feature.replace('_', ' ').toUpperCase()}</div>
                <div class="factor-bar-bg">
                    <div class="factor-bar ${barClass}" style="width: ${Math.min(100, Math.max(10, c.impact_magnitude))}%;"></div>
                </div>
                <div class="factor-val">${sign}${c.z_score} σ</div>
            `;
            elements.featureImpactList.appendChild(row);
        });

        // Recommendations
        elements.recommendationsList.innerHTML = '';
        result.recommendations.forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            elements.recommendationsList.appendChild(li);
        });
    }

    // 6. Compare Models & Benchmarks
    async function loadAndRenderBenchmarks() {
        if (!AppState.benchmarksData) {
            try {
                const res = await fetch('/api/benchmarks');
                const data = await res.json();
                if (data.status === 'success') {
                    AppState.benchmarksData = data.benchmarks;
                }
            } catch (err) {
                console.error('Failed to load benchmarks:', err);
                return;
            }
        }
        renderBenchmarksForDisease(elements.labDiseaseSelect.value);
    }

    function renderBenchmarksForDisease(diseaseKey) {
        if (!AppState.benchmarksData || !AppState.benchmarksData[diseaseKey]) return;
        const diseaseData = AppState.benchmarksData[diseaseKey];
        const models = diseaseData.models;

        let bestAcc = 0, bestAccModel = '';
        let bestAuc = 0, bestAucModel = '';
        let bestRecall = 0, bestRecallModel = '';

        for (const [mKey, mVal] of Object.entries(models)) {
            if (mVal.accuracy > bestAcc) { bestAcc = mVal.accuracy; bestAccModel = mKey; }
            if (mVal.roc_auc > bestAuc) { bestAuc = mVal.roc_auc; bestAucModel = mKey; }
            if (mVal.recall > bestRecall) { bestRecall = mVal.recall; bestRecallModel = mKey; }
        }

        document.getElementById('metricBestAcc').textContent = `${bestAcc}%`;
        document.getElementById('metricBestAccModel').textContent = bestAccModel.replace('_', ' ').toUpperCase();

        document.getElementById('metricBestAuc').textContent = `${bestAuc.toFixed(3)}`;
        document.getElementById('metricBestAucModel').textContent = bestAucModel.replace('_', ' ').toUpperCase();

        document.getElementById('metricBestRecall').textContent = `${bestRecall}%`;
        document.getElementById('metricBestRecallModel').textContent = bestRecallModel.replace('_', ' ').toUpperCase();

        const cvScores = Object.values(models).map(m => m.cv_roc_auc_mean).filter(Boolean);
        const avgCv = cvScores.length ? (cvScores.reduce((a, b) => a + b, 0) / cvScores.length).toFixed(3) : bestAuc.toFixed(3);
        document.getElementById('metricBestCv').textContent = avgCv;

        // Populate Table
        elements.benchmarkTableBody.innerHTML = '';
        const modelOrder = ['ensemble', 'random_forest', 'xgboost', 'svm', 'logistic_regression'];

        modelOrder.forEach(mKey => {
            const m = models[mKey];
            if (!m) return;
            const tr = document.createElement('tr');

            const cvText = m.cv_roc_auc_mean ? `${m.cv_roc_auc_mean}` : '--';
            tr.innerHTML = `
                <td><strong>${mKey.replace('_', ' ').toUpperCase()}</strong></td>
                <td><strong>${m.accuracy}%</strong></td>
                <td>${m.precision}%</td>
                <td>${m.recall}%</td>
                <td>${m.specificity}%</td>
                <td>${m.f1_score}%</td>
                <td><strong>${m.roc_auc}</strong></td>
                <td>${cvText}</td>
            `;
            elements.benchmarkTableBody.appendChild(tr);
        });

        // Charts
        window.PulseCharts.renderRocCurves('rocChartCanvas', models);
        const featImpModel = models['random_forest'] || models['xgboost'] || models['ensemble'];
        if (featImpModel && featImpModel.feature_importances) {
            window.PulseCharts.renderFeatureImportance('featureImportanceCanvas', featImpModel.feature_importances);
        }

        renderConfusionMatrix();
    }

    function renderConfusionMatrix() {
        const diseaseKey = elements.labDiseaseSelect.value;
        const modelKey = elements.cmModelSelect.value;
        if (!AppState.benchmarksData || !AppState.benchmarksData[diseaseKey]) return;

        const modelData = AppState.benchmarksData[diseaseKey].models[modelKey];
        if (!modelData || !modelData.confusion_matrix) return;

        const cm = modelData.confusion_matrix;
        const tn = cm[0][0];
        const fp = cm[0][1];
        const fn = cm[1][0];
        const tp = cm[1][1];

        document.getElementById('cmTN').querySelector('.cm-val').textContent = tn;
        document.getElementById('cmFP').querySelector('.cm-val').textContent = fp;
        document.getElementById('cmFN').querySelector('.cm-val').textContent = fn;
        document.getElementById('cmTP').querySelector('.cm-val').textContent = tp;

        const sensitivity = (tp + fn) > 0 ? ((tp / (tp + fn)) * 100).toFixed(1) : 0;
        const specificity = (tn + fp) > 0 ? ((tn / (tn + fp)) * 100).toFixed(1) : 0;
        const fdr = (tp + fp) > 0 ? ((fp / (tp + fp)) * 100).toFixed(1) : 0;

        document.getElementById('cmSensitivityVal').textContent = `${sensitivity}%`;
        document.getElementById('cmSpecificityVal').textContent = `${specificity}%`;
        document.getElementById('cmFdrVal').textContent = `${fdr}%`;
    }
});
