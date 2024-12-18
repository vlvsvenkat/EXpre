// Store user answers and correct answers
const userAnswers = [];
let correctAnswers = [];

// Dynamically render quiz questions
function renderQuiz(quizQuestions) {
    const quizContainer = document.getElementById('quiz-container');

    quizQuestions.forEach((questionObj, index) => {
        const questionBlock = document.createElement('div');
        questionBlock.className = 'question-block';

        // Question text
        const questionText = document.createElement('p');
        questionText.innerText = `${index + 1}. ${questionObj.question}`;
        questionBlock.appendChild(questionText);

        // Options
        questionObj.options.forEach((option, optionIndex) => {
            const label = document.createElement('label');
            label.innerHTML = `
                <input type="radio" name="question_${index}" value="${option}">
                ${option}
            `;
            questionBlock.appendChild(label);
        });

        quizContainer.appendChild(questionBlock);
        correctAnswers[index] = questionObj.answer; // Store correct answer
    });

    // Display Submit Button
    const submitButton = document.getElementById('submit-quiz');
    submitButton.style.display = 'block';
    submitButton.addEventListener('click', submitQuiz);
}

// Submit the quiz and calculate results
function submitQuiz() {
    userAnswers.length = 0; // Reset user answers
    correctAnswers.forEach((_, index) => {
        const selectedOption = document.querySelector(`input[name="question_${index}"]:checked`);
        userAnswers[index] = selectedOption ? selectedOption.value : null;
    });

    // Check if all questions are answered
    if (userAnswers.includes(null)) {
        alert("Please answer all the questions before submitting.");
        return;
    }

    // Calculate results
    let correctCount = 0;
    const feedback = [];
    correctAnswers.forEach((answer, index) => {
        if (userAnswers[index] === answer) {
            correctCount++;
            feedback.push(`Question ${index + 1}: Correct`);
        } else {
            feedback.push(`Question ${index + 1}: Incorrect (Correct Answer: ${answer})`);
        }
    });

    // Send feedback to the server
    fetch('/quiz-results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ correct: correctCount, total: correctAnswers.length, feedback })
    }).then(response => response.json()).then(data => {
        if (data.redirect) {
            window.location.href = data.redirect;
        }
    }).catch(err => console.error('Error submitting quiz:', err));
}

// Initialize quiz on page load
document.addEventListener('DOMContentLoaded', () => {
    renderQuiz(quizQuestions);
});
