/**
 * Smart Garage & Spare Parts Assistance System
 * Royal Enfield Thunderbird 350X - Interactive Frontend Helper
 */

document.addEventListener("DOMContentLoaded", () => {
  // Flash auto-dismiss after 6 seconds
  const flashAlerts = document.querySelectorAll(".flash-alert");
  flashAlerts.forEach((alert) => {
    setTimeout(() => {
      alert.style.transition = "opacity 0.4s ease";
      alert.style.opacity = "0";
      setTimeout(() => alert.remove(), 400);
    }, 6000);
  });

  // Handle quick prediction form on Predict page if present
  const predictForm = document.getElementById("prediction-form");
  if (predictForm) {
    predictForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const submitBtn = predictForm.querySelector("button[type='submit']");
      const originalText = submitBtn.innerText;
      submitBtn.disabled = true;
      submitBtn.innerText = "Computing Estimates...";

      const resultBox = document.getElementById("prediction-results");
      const errorBox = document.getElementById("prediction-error");
      if (resultBox) resultBox.style.display = "none";
      if (errorBox) errorBox.style.display = "none";

      const formData = new FormData(predictForm);
      const payload = {
        bike_model: formData.get("bike_model") || "Royal Enfield Thunderbird 350X",
        manufacturing_year: parseInt(formData.get("manufacturing_year")) || 2019,
        part_name: formData.get("part_name") || "",
        part_category: formData.get("part_category") || "Engine/Clutch",
        supplier_type: formData.get("supplier_type") || "Online Supplier",
        city: formData.get("city") || "Pune",
        garage_type: formData.get("garage_type") || "Royal Enfield Specialist",
        part_condition: formData.get("part_condition") || "New",
        service_priority: formData.get("service_priority") || "Medium",
      };

      try {
        const response = await fetch("/api/predict-part-details", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Prediction request failed.");
        }

        // Populate results
        document.getElementById("res-part-name").textContent = data.part_name || payload.part_name;
        document.getElementById("res-category").textContent = data.part_category || payload.part_category;
        
        const availBadge = document.getElementById("res-availability");
        availBadge.textContent = data.availability || "Unknown";
        availBadge.className = `badge badge-${(data.availability || "").toLowerCase()}`;

        document.getElementById("res-price").textContent = `₹${(data.estimated_price_inr || 0).toLocaleString("en-IN")}`;
        document.getElementById("res-wait").textContent = `${data.estimated_wait_days || 0} days`;
        
        if (data.confidence) {
          document.getElementById("res-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
        }

        // Update pre-fill link for service request
        const reqLink = document.getElementById("res-request-link");
        if (reqLink) {
          reqLink.href = `/service-request/new?part_name=${encodeURIComponent(payload.part_name)}&part_category=${encodeURIComponent(payload.part_category)}&city=${encodeURIComponent(payload.city)}&est_price=${data.estimated_price_inr}&wait_days=${data.estimated_wait_days}`;
        }

        if (resultBox) resultBox.style.display = "block";
      } catch (err) {
        if (errorBox) {
          errorBox.textContent = `Error: ${err.message}`;
          errorBox.style.display = "block";
        } else {
          alert(`Prediction Error: ${err.message}`);
        }
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = originalText;
      }
    });
  }
});
