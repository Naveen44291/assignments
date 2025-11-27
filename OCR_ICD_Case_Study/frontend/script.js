
let currentDocId = null;
let ocrChunks = [];
let pageWidth = 800;
let pageHeight = 1000;

// FastAPI URL
let apiBase = "http://127.0.0.1:8000";

function log(msg) {
  const el = document.getElementById("log");
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}

async function uploadFile() {
  const input = document.getElementById("fileInput");
  if (!input.files || !input.files[0]) {
    alert("Choose a file first");
    return;
  }
  const file = input.files[0];
  const form = new FormData();
  form.append("file", file);
  log("Uploading and running OCR...");

  const resp = await fetch(apiBase + "/upload", {
    method: "POST",
    body: form,
  });
  const data = await resp.json();
  if (!resp.ok) {
    log("Error: " + JSON.stringify(data));
    return;
  }

  currentDocId = data.doc_id;
  log("Upload OK. doc_id = " + currentDocId);

  const docResp = await fetch(apiBase + "/doc/" + currentDocId);
  const docData = await docResp.json();
  if (docData.error) {
    log("Error: " + docData.error);
    return;
  }

  ocrChunks = docData.chunks || [];
  pageWidth = docData.page_width || 800;
  pageHeight = docData.page_height || 1000;

  if (docData.image_data_url) {
    const img = document.getElementById("page-image");
    img.onload = () => {
      const canvas = document.getElementById("overlay");
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      drawAllBoxes();
    };
    img.src = docData.image_data_url;
  } else {
    log("No image preview available.");
  }
}

function drawAllBoxes(highlightICDs = []) {
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const img = document.getElementById("page-image");

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!img.naturalWidth || !img.naturalHeight) return;

  const scaleX = canvas.width / pageWidth;
  const scaleY = canvas.height / pageHeight;

  // Grey boxes for all OCR lines
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(0,0,0,0.3)";
  ocrChunks.forEach(ch => {
    const [x1, y1, x2, y2] = ch.bbox;
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
  });

  // Red boxes for ICD supporting lines
  ctx.lineWidth = 2;
  ctx.strokeStyle = "red";
  highlightICDs.forEach(ch => {
    const [x1, y1, x2, y2] = ch.bbox;
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
  });
}

async function extractICD() {
  if (!currentDocId) {
    alert("Upload a document first");
    return;
  }
  log("Calling /extract-icd...");
  const resp = await fetch(apiBase + "/extract-icd", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: currentDocId }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    log("Error: " + JSON.stringify(data));
    return;
  }
  log("ICD extraction result: " + JSON.stringify(data, null, 2));

  log("Calling /view-report for grounded locations...");
  const repResp = await fetch(apiBase + "/view-report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: currentDocId, icd_codes: null }),
  });
  const repData = await repResp.json();
  if (!repResp.ok) {
    log("Error: " + JSON.stringify(repData));
    return;
  }
  log("Report with locations: " + JSON.stringify(repData, null, 2));

  const boxesToHighlight = [];
  repData.locations.forEach(loc => {
    const match = ocrChunks.find(ch => ch.text === loc.sentence);
    if (match) {
      boxesToHighlight.push(match);
    }
  });

  drawAllBoxes(boxesToHighlight);
}
