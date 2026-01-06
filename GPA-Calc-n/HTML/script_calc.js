let currentStep = 1;
let userName = '';

function nextStep() {
  userName = document.getElementById("username").value.trim();
  if (!userName) {
    alert("Please enter your name");
    return;
  }

  const type = document.getElementById("calcType").value;
  document.getElementById("step1").classList.remove("active");
  document.getElementById(type === "sgpa" ? "step2-sgpa" : "step2-cgpa").classList.add("active");
  currentStep++;
}

function prevStep() {
  document.querySelectorAll('.step').forEach(s => s.classList.remove("active"));
  document.getElementById("step1").classList.add("active");
  currentStep--;
}

function createSGPAFields() {
  const num = parseInt(document.getElementById("numSubjects").value);
  if (isNaN(num) || num <= 0) {
    alert("Enter a valid number of subjects");
    return;
  }

  const container = document.getElementById("step3-sgpa");
  container.innerHTML = `<form name="sgpaForm">`;

  for (let i = 1; i <= num; i++) {
    container.innerHTML += `
      <label>Subject ${i} Credit:</label>
      <input type="number" name="credit${i}" min="1" required />
      <label>Subject ${i} Grade (O, A+, A, B+, B, C+, C, F):</label>
      <input type="text" name="grade${i}" placeholder="e.g. O, A+, A " required />
    `;
  }

  container.innerHTML += `
    <div class="button-row">
      <button type="button" class="btn" onclick="prevStepToSGPA()">Back</button>
      <button type="button" class="btn" onclick="calculateSGPA()">Calculate SGPA</button>
    </div>
  </form>`;

  document.querySelectorAll('.step').forEach(s => s.classList.remove("active"));
  container.classList.add("active");
  currentStep++;
}

function createCGPAFields() {
  const num = parseInt(document.getElementById("numSemesters").value);
  if (isNaN(num) || num <= 0) {
    alert("Enter a valid number of semesters");
    return;
  }

  const container = document.getElementById("step3-cgpa");
  container.innerHTML = `<form name="cgpaForm">`;

  for (let i = 1; i <= num; i++) {
    container.innerHTML += `
      <label>Semester ${i} SGPA:</label>
      <input type="number" name="sgpa${i}" min="0" max="10" step="0.01" required />
    `;
  }

  container.innerHTML += `
    <div class="button-row">
      <button type="button" class="btn" onclick="prevStepToCGPA()">Back</button>
      <button type="button" class="btn" onclick="calculateCGPA()">Calculate CGPA</button>
    </div>
  </form>`;

  document.querySelectorAll('.step').forEach(s => s.classList.remove("active"));
  container.classList.add("active");
  currentStep++;
}

function prevStepToSGPA() {
  document.querySelectorAll('.step').forEach(s => s.classList.remove("active"));
  document.getElementById("step2-sgpa").classList.add("active");
  currentStep = 2;
}

function prevStepToCGPA() {
  document.querySelectorAll('.step').forEach(s => s.classList.remove("active"));
  document.getElementById("step2-cgpa").classList.add("active");
  currentStep = 2;
}

function getGradePoint(grade) {
  switch (grade.toUpperCase()) {
    case 'O': return 10;
    case 'A+': return 9;
    case 'A': return 8;
    case 'B+': return 7;
    case 'B': return 6;
    case 'C+': return 5;
    case 'C': return 4;
    case 'F': return 0;
    default: return null;
  }
}

function calculateSGPA() {
  const creditInputs = document.querySelectorAll('[name^="credit"]');
  const gradeInputs = document.querySelectorAll('[name^="grade"]');

  if (creditInputs.length !== gradeInputs.length) {
    alert("Mismatch between credit and grade fields.");
    return;
  }

  let totalCredits = 0;
  let totalPoints = 0;

  for (let i = 0; i < creditInputs.length; i++) {
    const credit = parseFloat(creditInputs[i].value);
    const grade = gradeInputs[i].value.trim().toUpperCase();

    if (isNaN(credit) || credit <= 0) {
      alert(`Invalid credit for Subject ${i + 1}`);
      return;
    }

    const gradePoint = getGradePoint(grade);
    if (gradePoint === null) {
      alert(`Invalid grade "${grade}" for Subject ${i + 1}`);
      return;
    }

    totalCredits += credit;
    totalPoints += credit * gradePoint;
  }

  if (totalCredits === 0) {
    alert("Total credits cannot be zero.");
    return;
  }

  const sgpa = (totalPoints / totalCredits).toFixed(2);
  showResult(`🎉 Congrats ${userName}! Your SGPA is ${sgpa}`);
}

function calculateCGPA() {
  const sgpaInputs = document.querySelectorAll('[name^="sgpa"]');
  let total = 0;

  for (let i = 0; i < sgpaInputs.length; i++) {
    const sgpa = parseFloat(sgpaInputs[i].value);
    if (isNaN(sgpa) || sgpa < 0 || sgpa > 10) {
      alert(`Invalid SGPA for Semester ${i + 1}`);
      return;
    }
    total += sgpa;
  }

  const cgpa = (total / sgpaInputs.length).toFixed(2);
  showResult(`🎉 Congrats ${userName}! Your CGPA is ${cgpa}`);
}

function showResult(text) {
  document.querySelectorAll('.step').forEach(s => s.classList.remove("active"));
  document.getElementById("resultStep").classList.add("active");
  document.getElementById("resultText").innerText = text;
}

function resetForm() {
  location.reload();
}
