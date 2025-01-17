// Toggle between simulation & queueing
const modeSelect = document.getElementById("modeSelect");
const simulateParams = document.getElementById("simulateParams");
const queueingParams = document.getElementById("queueingParams");
function updateMode() {
  if (modeSelect.value === "simulate") {
    simulateParams.classList.remove("hidden");
    queueingParams.classList.add("hidden");
  } else {
    simulateParams.classList.add("hidden");
    queueingParams.classList.remove("hidden");
  }
}
modeSelect.addEventListener("change", updateMode);
updateMode();

// SIM ARRIVAL
const arrivalSelect = document.getElementById("arrival_type");
const arrivalPoissonParams = document.getElementById("arrivalPoissonParams");
const arrivalNormalParams = document.getElementById("arrivalNormalParams");
const arrivalUniformParams = document.getElementById("arrivalUniformParams");
const arrivalExpParams = document.getElementById("arrivalExpParams");
function updateArrivalSim() {
  arrivalPoissonParams.classList.add("hidden");
  arrivalNormalParams.classList.add("hidden");
  arrivalUniformParams.classList.add("hidden");
  arrivalExpParams.classList.add("hidden");
  if (arrivalSelect.value === "poisson") {
    arrivalPoissonParams.classList.remove("hidden");
  } else if (arrivalSelect.value === "normal") {
    arrivalNormalParams.classList.remove("hidden");
  } else if (arrivalSelect.value === "uniform") {
    arrivalUniformParams.classList.remove("hidden");
  } else if (arrivalSelect.value === "exponential") {
    arrivalExpParams.classList.remove("hidden");
  }
}
arrivalSelect.addEventListener("change", updateArrivalSim);
updateArrivalSim();

// SIM SERVICE
const serviceSelect = document.getElementById("service_type");
const serviceExponentialParams = document.getElementById("serviceExponentialParams");
const serviceNormalParams = document.getElementById("serviceNormalParams");
const serviceUniformParams = document.getElementById("serviceUniformParams");
const servicePoissonParams = document.getElementById("servicePoissonParams");
function updateServiceSim() {
  serviceExponentialParams.classList.add("hidden");
  serviceNormalParams.classList.add("hidden");
  serviceUniformParams.classList.add("hidden");
  servicePoissonParams.classList.add("hidden");
  if (serviceSelect.value === "exponential") {
    serviceExponentialParams.classList.remove("hidden");
  } else if (serviceSelect.value === "normal") {
    serviceNormalParams.classList.remove("hidden");
  } else if (serviceSelect.value === "uniform") {
    serviceUniformParams.classList.remove("hidden");
  } else if (serviceSelect.value === "poisson") {
    servicePoissonParams.classList.remove("hidden");
  }
}
serviceSelect.addEventListener("change", updateServiceSim);
updateServiceSim();

// Priority
const enablePriority = document.getElementById("enable_priority");
const priorityLevels = document.getElementById("priorityLevels");
enablePriority.addEventListener("change", () => {
  if (enablePriority.value === "yes") {
    priorityLevels.classList.remove("hidden");
  } else {
    priorityLevels.classList.add("hidden");
  }
});

// RateWise
const rateWiseSelect = document.getElementById("rate_wise");
const timeUnits = document.getElementById("timeUnits");
rateWiseSelect.addEventListener("change", () => {
  if (rateWiseSelect.value === "yes") {
    timeUnits.classList.remove("hidden");
  } else {
    timeUnits.classList.add("hidden");
  }
});

// QUEUEING ARRIVAL
const qArrivalSelect = document.getElementById("q_arrival_type");
const qArrivalNormalParams = document.getElementById("qArrivalNormalParams");
const qArrivalUniformParams = document.getElementById("qArrivalUniformParams");
const qArrivalExpParams = document.getElementById("qArrivalExpParams");
const qArrivalPoissonParams = document.getElementById("qArrivalPoissonParams");
function updateQArrival() {
  qArrivalNormalParams.classList.add("hidden");
  qArrivalUniformParams.classList.add("hidden");
  qArrivalExpParams.classList.add("hidden");
  qArrivalPoissonParams.classList.add("hidden");
  if (qArrivalSelect.value === "normal") {
    qArrivalNormalParams.classList.remove("hidden");
  } else if (qArrivalSelect.value === "uniform") {
    qArrivalUniformParams.classList.remove("hidden");
  } else if (qArrivalSelect.value === "exponential") {
    qArrivalExpParams.classList.remove("hidden");
  } else if (qArrivalSelect.value === "poisson") {
    qArrivalPoissonParams.classList.remove("hidden");
  }
}
qArrivalSelect.addEventListener("change", updateQArrival);
updateQArrival();

// QUEUEING SERVICE
const qServiceSelect = document.getElementById("q_service_type");
const qServiceNormalParams = document.getElementById("qServiceNormalParams");
const qServiceUniformParams = document.getElementById("qServiceUniformParams");
const qServiceExpParams = document.getElementById("qServiceExpParams");
const qServicePoissonParams = document.getElementById("qServicePoissonParams");
function updateQService() {
  qServiceNormalParams.classList.add("hidden");
  qServiceUniformParams.classList.add("hidden");
  qServiceExpParams.classList.add("hidden");
  qServicePoissonParams.classList.add("hidden");
  if (qServiceSelect.value === "normal") {
    qServiceNormalParams.classList.remove("hidden");
  } else if (qServiceSelect.value === "uniform") {
    qServiceUniformParams.classList.remove("hidden");
  } else if (qServiceSelect.value === "exponential") {
    qServiceExpParams.classList.remove("hidden");
  } else if (qServiceSelect.value === "poisson") {
    qServicePoissonParams.classList.remove("hidden");
  }
}
qServiceSelect.addEventListener("change", updateQService);
updateQService();

// SUBMIT
const mainForm = document.getElementById("mainForm");
const loader = document.getElementById("loader");
const resultsContainer = document.getElementById("resultsContainer");
const resultsOutput = document.getElementById("resultsOutput");
const ganttContainer = document.getElementById("ganttContainer");
const ganttImagesWrapper = document.getElementById("ganttImagesWrapper");
const barChartsContainer = document.getElementById("barChartsContainer");
const barChartsImage = document.getElementById("barChartsImage");
const queueingContainer = document.getElementById("queueingContainer");
const queueingOutput = document.getElementById("queueingOutput");

mainForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  loader.style.display = "block";

  resultsContainer.classList.add("hidden");
  ganttContainer.classList.add("hidden");
  barChartsContainer.classList.add("hidden");
  queueingContainer.classList.add("hidden");

  resultsOutput.textContent = "";
  ganttImagesWrapper.innerHTML = "";
  barChartsImage.src = "";
  queueingOutput.textContent = "";

  const formData = new FormData(mainForm);
  try {
    const resp = await fetch("/run_operation", {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) throw new Error("Server error: " + resp.statusText);
    const data = await resp.json();
    loader.style.display = "none";

    if (data.error) {
      // If server returns an error (e.g. invalid input)
      alert("Error: " + data.error);
      return;
    }

    if (data.mode === "simulate") {
      resultsContainer.classList.remove("hidden");
      resultsOutput.textContent = data.simulation_text || "No results";

      if (data.gantt_images && data.gantt_images.length > 0) {
        ganttContainer.classList.remove("hidden");
        data.gantt_images.forEach((b64, idx) => {
          const img = document.createElement("img");
          img.src = "data:image/png;base64," + b64;
          img.alt = `Gantt ${idx + 1}`;
          img.style.display = "block";
          img.style.marginBottom = "20px";
          ganttImagesWrapper.appendChild(img);
        });
      }
      if (data.bar_charts) {
        barChartsContainer.classList.remove("hidden");
        barChartsImage.src = "data:image/png;base64," + data.bar_charts;
      }
    } else if (data.mode === "queueing") {
      queueingContainer.classList.remove("hidden");
      queueingOutput.textContent =
        data.queueing_text || "No queueing output.";
    }
  } catch (err) {
    loader.style.display = "none";
    alert("Error: " + err);
  }
});
