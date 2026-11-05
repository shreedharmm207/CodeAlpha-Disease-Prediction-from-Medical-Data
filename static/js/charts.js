/**
 * charts.js
 * Visualization Engine for Model Laboratory & Diagnostic Explanations
 * Uses Chart.js (with canvas fallback) for ROC curves and Feature Importance.
 */

window.PulseCharts = (function () {
    let rocChartInstance = null;
    let featImpChartInstance = null;

    const MODEL_COLORS = {
        ensemble: '#0284c7', // Sky Blue
        xgboost: '#8b5cf6', // Violet
        random_forest: '#10b981', // Emerald
        svm: '#f59e0b', // Amber
        logistic_regression: '#ef4444' // Rose
    };

    function renderRocCurves(canvasId, modelsMetadata) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        if (rocChartInstance) {
            rocChartInstance.destroy();
        }

        const datasets = [];

        // Diagonal chance line (AUC = 0.5)
        datasets.push({
            label: 'Random Chance (AUC = 0.50)',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: '#94a3b8',
            borderWidth: 1.5,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
        });

        for (const [modelKey, modelData] of Object.entries(modelsMetadata)) {
            if (!modelData.roc_curve) continue;
            const color = MODEL_COLORS[modelKey] || '#64748b';
            const modelName = modelKey.replace('_', ' ').toUpperCase();
            
            datasets.push({
                label: `${modelName} (AUC: ${modelData.roc_auc})`,
                data: modelData.roc_curve,
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: modelKey === 'ensemble' ? 3 : 2,
                pointRadius: 0,
                tension: 0.1
            });
        }

        // If Chart.js is available
        if (typeof Chart !== 'undefined') {
            rocChartInstance = new Chart(ctx, {
                type: 'line',
                data: { datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 600 },
                    interaction: { mode: 'nearest', intersect: false },
                    scales: {
                        x: {
                            type: 'linear',
                            min: 0,
                            max: 1,
                            title: { display: true, text: 'False Positive Rate (1 - Specificity)', font: { size: 11, weight: 'bold' } },
                            grid: { color: '#e2e8f0' }
                        },
                        y: {
                            type: 'linear',
                            min: 0,
                            max: 1,
                            title: { display: true, text: 'True Positive Rate (Sensitivity)', font: { size: 11, weight: 'bold' } },
                            grid: { color: '#e2e8f0' }
                        }
                    },
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.dataset.label}: TPR=${context.parsed.y.toFixed(2)}, FPR=${context.parsed.x.toFixed(2)}`;
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    function renderFeatureImportance(canvasId, importancesDict, subtitleElementId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        if (featImpChartInstance) {
            featImpChartInstance.destroy();
        }

        // Top 8 features
        const sortedEntries = Object.entries(importancesDict).slice(0, 8);
        const labels = sortedEntries.map(e => e[0].replace('_', ' ').toUpperCase());
        const values = sortedEntries.map(e => e[1]);

        if (subtitleElementId) {
            const sub = document.getElementById(subtitleElementId);
            if (sub && sortedEntries.length > 0) {
                sub.textContent = `Top predictor: ${labels[0]} (${values[0]}% importance weight)`;
            }
        }

        if (typeof Chart !== 'undefined') {
            featImpChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Relative Predictive Weight (%)',
                        data: values,
                        backgroundColor: '#0284c7',
                        borderRadius: 4,
                        borderSkipped: false
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 600 },
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: Math.min(100, Math.ceil(Math.max(...values) * 1.25)),
                            title: { display: true, text: 'Relative Weight (%)', font: { size: 11, weight: 'bold' } },
                            grid: { color: '#e2e8f0' }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { font: { size: 11, weight: '600' } }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Predictive Weight: ${context.parsed.x}%`;
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    return {
        renderRocCurves: renderRocCurves,
        renderFeatureImportance: renderFeatureImportance
    };
})();

