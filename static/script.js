document.addEventListener("DOMContentLoaded", function () {
  const bulkUploadForm = document.getElementById("bulkUploadForm");
  const bulkUploadBtn = document.getElementById("bulkUploadBtn");

  if (bulkUploadForm) {
    bulkUploadForm.addEventListener("submit", async function (e) {
      e.preventDefault();
      const files = document.getElementById("pdfFiles").files;
      if (!files || files.length === 0) {
        alert("Please select at least one PDF file");
        return;
      }
      if (files.length > 20) {
        alert("Maximum 20 files allowed. Please select fewer files.");
        return;
      }

      showProgressModal(files.length);
      const formData = new FormData();
      for (let i = 0; i < files.length; i++)
        formData.append("files[]", files[i]);

      try {
        bulkUploadBtn.disabled = true;
        const resp = await fetch("/bulk_upload", {
          method: "POST",
          body: formData,
        });
        const data = await resp.json();
        if (data.success) displayBulkResults(data.results, data.summary);
        else alert("Error: " + data.error);
      } catch (err) {
        alert("Upload failed: " + err.message);
      } finally {
        bulkUploadBtn.disabled = false;
        hideProgressModal();
      }
    });
  }

  const zipUploadForm = document.getElementById("zipUploadForm");
  if (zipUploadForm) {
    zipUploadForm.addEventListener("submit", async function (e) {
      e.preventDefault();
      const zipFile = document.getElementById("zipFile").files[0];
      if (!zipFile) {
        alert("Please select a ZIP file");
        return;
      }
      const formData = new FormData();
      formData.append("zip_file", zipFile);

      try {
        const resp = await fetch("/upload_zip", {
          method: "POST",
          body: formData,
        });
        const data = await resp.json();
        if (data.success) {
          const summary = {
            total_files: data.total_files,
            successful: data.results.filter((r) => r.status === "success")
              .length,
            failed: data.results.filter((r) => r.status === "failed").length,
          };
          displayBulkResults(data.results, summary);
        } else alert("Error: " + data.error);
      } catch (err) {
        alert("Upload failed: " + err.message);
      }
    });
  }

  function displayBulkResults(results, summary) {
    const resultsDiv = document.getElementById("bulkResults");
    const resultsBody = document.getElementById("resultsBody");
    const summaryTitle = document.getElementById("summaryTitle");
    const summaryContent = document.getElementById("summaryContent");

    summaryTitle.textContent = `Processed ${summary.total_files} files`;
    summaryContent.innerHTML = `
			<div class="row">
				<div class="col-md-4"><div class="small text-success"><strong>${summary.successful}</strong> successful</div></div>
				<div class="col-md-4"><div class="small text-danger"><strong>${summary.failed}</strong> failed</div></div>
				<div class="col-md-4"><button class="btn btn-sm btn-outline-dark" onclick="alert('Distribution feature not implemented')">View distribution</button></div>
			</div>
		`;

    resultsBody.innerHTML = "";
    results.forEach((result) => {
      const row = document.createElement("tr");
      let confidenceCell = "";
      if (result.status === "success") {
        confidenceCell = `<span class="${getConfidenceBadgeClass(
          result.confidence
        )} small px-2 py-1 rounded">${result.confidence}%</span>`;
      }
      row.innerHTML = `
				<td class="align-middle">${result.filename}</td>
				<td class="align-middle">${result.prediction || ""}</td>
				<td class="align-middle">${confidenceCell}</td>
				<td class="align-middle small text-muted">${result.status}${
        result.error
          ? `<div class="text-danger small">${result.error}</div>`
          : ""
      }</td>
			`;
      resultsBody.appendChild(row);
    });

    resultsDiv.style.display = "block";
    resultsDiv.scrollIntoView({ behavior: "smooth" });
  }

  function getConfidenceBadgeClass(confidence) {
    // subdued badge palette
    if (confidence >= 80) return "badge bg-dark text-white";
    if (confidence >= 60) return "badge bg-secondary text-white";
    return "badge bg-light text-dark border";
  }

  function showProgressModal(totalFiles) {
    const modalEl = document.getElementById("progressModal");
    if (!modalEl) return;
    const modal = new bootstrap.Modal(modalEl);
    modal.show();
    let processed = 0;
    const interval = setInterval(() => {
      processed++;
      const percent = Math.min((processed / totalFiles) * 100, 90);
      const bar = document.querySelector(".progress-bar");
      if (bar) bar.style.width = percent + "%";
      const txt = document.getElementById("progressText");
      if (txt)
        txt.textContent = `Processing ${processed} of ${totalFiles} files...`;
      if (processed >= totalFiles) clearInterval(interval);
    }, 500);
  }

  function hideProgressModal() {
    const modal = bootstrap.Modal.getInstance(
      document.getElementById("progressModal")
    );
    if (modal) modal.hide();
  }

  const downloadCsvBtn = document.getElementById("downloadCsvBtn");
  if (downloadCsvBtn) {
    downloadCsvBtn.addEventListener("click", function () {
      alert(
        "CSV download: implement server-side session or pass results as query to /download_results."
      );
    });
  }
});
