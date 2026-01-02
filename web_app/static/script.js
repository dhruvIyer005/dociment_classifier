// static/script.js - BULK UPLOAD VERSION
let selectedFiles = [];

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Bulk PDF Classifier loaded');
    
    // File input event listener
    document.getElementById('fileInput').addEventListener('change', function(e) {
        if (this.files.length > 0) {
            handleFiles(Array.from(this.files));
        }
    });

    // Upload area click listener
    document.getElementById('uploadArea').addEventListener('click', function() {
        document.getElementById('fileInput').click();
    });

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.getElementById('uploadArea').addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        document.getElementById('uploadArea').addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        document.getElementById('uploadArea').addEventListener(eventName, unhighlight, false);
    });

    document.getElementById('uploadArea').addEventListener('drop', handleDrop, false);
    
    // Analyze button
    document.getElementById('analyzeFileBtn').addEventListener('click', analyzeFiles);
    
    // Clear button
    document.getElementById('clearFilesBtn').addEventListener('click', clearFiles);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    document.getElementById('uploadArea').classList.add('dragover');
}

function unhighlight() {
    document.getElementById('uploadArea').classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = Array.from(dt.files);
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        // Filter only PDF files
        const pdfFiles = files.filter(file => 
            file.type === 'application/pdf' || 
            file.name.toLowerCase().endsWith('.pdf')
        );
        
        if (pdfFiles.length === 0) {
            showError('Please select PDF files only.');
            return;
        }
        
        // Limit to 20 files
        if (pdfFiles.length > 20) {
            showError(`Maximum 20 files allowed. You selected ${pdfFiles.length} files.`);
            return;
        }
        
        // Add to selected files
        selectedFiles = [...selectedFiles, ...pdfFiles].slice(0, 20); // Keep max 20
        
        displayFileList();
    }
}

function displayFileList() {
    const fileList = document.getElementById('fileList');
    const fileInfo = document.getElementById('fileInfo');
    const uploadArea = document.getElementById('uploadArea');
    
    if (selectedFiles.length > 0) {
        // Show file list
        fileList.innerHTML = '';
        
        selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
                <button class="remove-file" onclick="removeFile(${index})">×</button>
            `;
            fileList.appendChild(fileItem);
        });
        
        // Show summary
        document.getElementById('fileCount').textContent = selectedFiles.length;
        document.getElementById('totalSize').textContent = formatFileSize(
            selectedFiles.reduce((total, file) => total + file.size, 0)
        );
        
        // Show/Hide elements
        fileInfo.classList.remove('hidden');
        uploadArea.style.display = 'none';
        
        // Update upload area text
        document.querySelector('.upload-text').textContent = `Add more files (${20 - selectedFiles.length} remaining)`;
        
    } else {
        // No files selected
        fileInfo.classList.add('hidden');
        uploadArea.style.display = 'block';
        document.querySelector('.upload-text').textContent = 'Drag & drop PDF files or click to browse';
    }
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    displayFileList();
}

function clearFiles() {
    selectedFiles = [];
    document.getElementById('fileInput').value = '';
    displayFileList();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Analysis functions
async function analyzeFiles() {
    if (selectedFiles.length === 0) {
        showError('Please select at least one PDF file to analyze.');
        return;
    }
    
    const formData = new FormData();
    
    // Add all files
    selectedFiles.forEach((file, index) => {
        formData.append('files[]', file);
    });
    
    await performBulkAnalysis(formData);
}

async function performBulkAnalysis(formData) {
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    
    // Clear previous results
    results.classList.add('hidden');
    error.classList.add('hidden');
    
    // Show loading
    loading.classList.remove('hidden');
    
    try {
        const response = await fetch('/bulk_upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayBulkResults(result);
        } else {
            showError(result.error || 'Analysis failed. Please try again.');
        }
        
    } catch (err) {
        console.error('Analysis error:', err);
        showError('Network error. Please check your connection and try again.');
    } finally {
        loading.classList.add('hidden');
    }
}

function displayBulkResults(data) {
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    
    error.classList.add('hidden');
    results.classList.remove('hidden');
    
    console.log('Bulk results:', data);
    
    // Display summary
    const summary = document.getElementById('summaryContent');
    if (data.summary) {
        summary.innerHTML = `
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-value">${data.summary.total_files}</span>
                    <span class="stat-label">Total Files</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value text-success">${data.summary.successful || 0}</span>
                    <span class="stat-label">Successful</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value text-danger">${data.summary.failed || 0}</span>
                    <span class="stat-label">Failed</span>
                </div>
            </div>
        `;
    }
    
    // Display results table
    const resultsBody = document.getElementById('resultsBody');
    resultsBody.innerHTML = '';
    
    if (data.results && data.results.length > 0) {
        data.results.forEach(result => {
            const row = document.createElement('tr');
            
            // Status badge
            let statusBadge = '';
            if (result.status === 'success') {
                statusBadge = `<span class="badge bg-success">Success</span>`;
            } else {
                statusBadge = `<span class="badge bg-danger">Failed</span>`;
            }
            
            // Confidence badge
            let confidenceBadge = '';
            if (result.confidence) {
                let badgeClass = 'bg-secondary';
                if (result.confidence >= 80) badgeClass = 'bg-success';
                else if (result.confidence >= 60) badgeClass = 'bg-warning';
                else badgeClass = 'bg-danger';
                
                confidenceBadge = `<span class="badge ${badgeClass}">${result.confidence}%</span>`;
            }
            
            row.innerHTML = `
                <td>${result.filename}</td>
                <td>${result.prediction || 'N/A'}</td>
                <td>${confidenceBadge}</td>
                <td>${statusBadge} ${result.error ? `<br><small class="text-danger">${result.error}</small>` : ''}</td>
            `;
            
            resultsBody.appendChild(row);
        });
    }
    
    // Show download button if results exist
    const downloadBtn = document.getElementById('downloadCsvBtn');
    if (data.results && data.results.length > 0) {
        downloadBtn.classList.remove('hidden');
        // Store results for download
        downloadBtn.dataset.results = JSON.stringify(data.results);
    } else {
        downloadBtn.classList.add('hidden');
    }
}

// Download CSV
document.getElementById('downloadCsvBtn').addEventListener('click', function() {
    const results = this.dataset.results;
    if (results) {
        // Create CSV content
        const data = JSON.parse(results);
        let csvContent = "data:text/csv;charset=utf-8,Filename,Document Type,Confidence %,Status,Error\n";
        
        data.forEach(result => {
            const row = [
                `"${result.filename}"`,
                `"${result.prediction || ''}"`,
                result.confidence || '',
                result.status || '',
                `"${result.error || ''}"`
            ];
            csvContent += row.join(',') + "\n";
        });
        
        // Create download link
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "classification_results.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});

function showError(message) {
    const error = document.getElementById('error');
    error.innerHTML = `
        <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle"></i> ${message}
        </div>
    `;
    error.classList.remove('hidden');
    
    document.getElementById('results').classList.add('hidden');
    document.getElementById('loading').classList.add('hidden');
}

function hideResults() {
    document.getElementById('results').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');
}