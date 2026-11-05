/**
 * batch.js
 * Cohort Population Screening & Bulk Triage Engine
 */

window.PulseBatch = (function () {
    let lastBatchResults = null;

    function initBatchModule() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('batchFileInput');
        const sampleBtn = document.getElementById('batchSampleRunBtn');
        const exportCsvBtn = document.getElementById('batchExportCsvBtn');
        const diseaseSelect = document.getElementById('batchDiseaseSelect');
        const sampleDownloadLink = document.getElementById('batchDownloadSampleLink');

        if (!dropZone || !fileInput) return;

        // Update download sample link when disease changes
        if (diseaseSelect && sampleDownloadLink) {
            diseaseSelect.addEventListener('change', () => {
                const disease = diseaseSelect.value;
                sampleDownloadLink.href = `/api/download-sample-batch/${disease}`;
            });
        }

        // Drag and drop events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleUpload(fileInput.files[0]);
            }
        });

        if (sampleBtn) {
            sampleBtn.addEventListener('click', () => {
                runSampleCohort();
            });
        }

        if (exportCsvBtn) {
            exportCsvBtn.addEventListener('click', () => {
                exportClassifiedCsv();
            });
        }
    }

    async function handleUpload(file) {
        const disease = document.getElementById('batchDiseaseSelect').value;
        const model = document.getElementById('batchModelSelect').value;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('disease', disease);
        formData.append('model', model);
        formData.append('use_sample', 'false');

        try {
            showLoading(true);
            const res = await fetch('/api/batch-predict', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                renderBatchResults(data.data);
            } else {
                alert(`Batch Screening Error: ${data.message}`);
            }
        } catch (err) {
            alert(`Error during batch processing: ${err.message}`);
        } finally {
            showLoading(false);
        }
    }

    async function runSampleCohort() {
        const disease = document.getElementById('batchDiseaseSelect').value;
        const model = document.getElementById('batchModelSelect').value;

        const formData = new FormData();
        formData.append('disease', disease);
        formData.append('model', model);
        formData.append('use_sample', 'true');

        try {
            showLoading(true);
            const res = await fetch('/api/batch-predict', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                renderBatchResults(data.data);
            } else {
                alert(`Sample Batch Error: ${data.message}`);
            }
        } catch (err) {
            alert(`Error running sample cohort: ${err.message}`);
        } finally {
            showLoading(false);
        }
    }

    function showLoading(isLoading) {
        const dropZone = document.getElementById('dropZone');
        if (dropZone) {
            dropZone.style.opacity = isLoading ? '0.5' : '1.0';
            dropZone.style.pointerEvents = isLoading ? 'none' : 'auto';
        }
    }

    function renderBatchResults(summary) {
        lastBatchResults = summary;
        const wrapper = document.getElementById('batchResultsWrapper');
        if (!wrapper) return;
        wrapper.style.display = 'block';

        // KPI updates
        document.getElementById('batchTotalCount').textContent = summary.total_patients;
        document.getElementById('batchHighCount').textContent = summary.high_risk_count;
        document.getElementById('batchHighPct').textContent = `${summary.high_risk_percentage}% of cohort`;
        document.getElementById('batchModCount').textContent = summary.moderate_risk_count;
        document.getElementById('batchLowCount').textContent = summary.low_risk_count;

        // Table population
        const tbody = document.getElementById('batchTableBody');
        tbody.innerHTML = '';

        summary.results.forEach(row => {
            const tr = document.createElement('tr');
            
            let tierBadgeClass = 'badge-neutral';
            if (row.tier === 'Elevated Risk') tierBadgeClass = 'badge-high';
            else if (row.tier === 'Moderate / Borderline') tierBadgeClass = 'badge-moderate';
            else if (row.tier === 'Low Risk') tierBadgeClass = 'badge-low';

            // Key observed biomarkers summary
            const observedKeys = Object.keys(row).filter(k => !['patient_id', 'risk_percentage', 'tier', 'action'].includes(k));
            const observedText = observedKeys.map(k => `${k}: <b>${row[k]}</b>`).join(', ');

            tr.innerHTML = `
                <td><strong>${row.patient_id}</strong></td>
                <td><span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.95rem;">${row.risk_percentage}%</span></td>
                <td><span class="clinical-badge ${tierBadgeClass}">${row.tier}</span></td>
                <td><strong>${row.action}</strong></td>
                <td><span style="font-size: 0.75rem; color: var(--text-muted);">${observedText}</span></td>
            `;
            tbody.appendChild(tr);
        });

        // Smooth scroll down to results
        wrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function exportClassifiedCsv() {
        if (!lastBatchResults || !lastBatchResults.results.length) {
            alert('No batch triage records available to export.');
            return;
        }

        const items = lastBatchResults.results;
        const replacer = (key, value) => value === null ? '' : value;
        const header = Object.keys(items[0]);
        const csv = [
            header.join(','),
            ...items.map(row => header.map(fieldName => JSON.stringify(row[fieldName], replacer)).join(','))
        ].join('\r\n');

        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `Clinical_Triage_Cohort_${Date.now()}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    return {
        init: initBatchModule
    };
})();

