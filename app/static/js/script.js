// Prediction form handling
document.addEventListener('DOMContentLoaded', () => {
    const predictionForm = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('results-container');
    const initialInfo = document.getElementById('initial-info');
    let chart = null;

    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Get form values
            const age = document.getElementById('age').value;
            const unitsEnrolled1 = document.getElementById('units_enrolled_1').value;
            const unitsApproved1 = document.getElementById('units_approved_1').value;
            const unitsEnrolled2 = document.getElementById('units_enrolled_2').value;
            const unitsApproved2 = document.getElementById('units_approved_2').value;
            const unemployment = document.getElementById('unemployment').value;

            // Validate form (make sure approved units are not more than enrolled units)
            if (parseInt(unitsApproved1) > parseInt(unitsEnrolled1)) {
                alert('Approved units in 1st semester cannot be more than enrolled units');
                return;
            }

            if (parseInt(unitsApproved2) > parseInt(unitsEnrolled2)) {
                alert('Approved units in 2nd semester cannot be more than enrolled units');
                return;
            }

            try {
                // Prepare request body
                const requestBody = {
                    age_at_enrollment: parseInt(age),
                    curricular_units_1st_sem_enrolled: parseInt(unitsEnrolled1),
                    curricular_units_1st_sem_approved: parseInt(unitsApproved1),
                    curricular_units_2nd_sem_enrolled: parseInt(unitsEnrolled2),
                    curricular_units_2nd_sem_approved: parseInt(unitsApproved2),
                    unemployment_rate: parseFloat(unemployment)
                };

                // Send prediction request
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    throw new Error('Prediction failed');
                }

                const result = await response.json();
                displayResults(result);

                // Hide initial info and show results
                if (initialInfo) initialInfo.classList.add('d-none');
                if (resultsContainer) resultsContainer.classList.remove('d-none');

            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while making the prediction. Please try again.');
            }
        });
    }

    function displayResults(result) {
        const predictionBadge = document.getElementById('prediction-badge');
        const dropoutProb = document.getElementById('dropout-prob');
        const graduateProb = document.getElementById('graduate-prob');

        // Display prediction result
        if (predictionBadge) {
            predictionBadge.textContent = result.prediction;
            predictionBadge.className = '';
            predictionBadge.classList.add(result.prediction.toLowerCase());
        }

        // Display probabilities
        if (dropoutProb) {
            dropoutProb.textContent = `${(result.dropout_probability * 100).toFixed(1)}%`;
        }

        if (graduateProb) {
            graduateProb.textContent = `${(result.graduate_probability * 100).toFixed(1)}%`;
        }

        // Create/update chart
        updateChart(result);
    }

    function updateChart(result) {
        const ctx = document.getElementById('probability-chart');

        if (ctx) {
            // Destroy existing chart if it exists
            if (chart) {
                chart.destroy();
            }

            // Create new chart
            chart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Dropout', 'Graduate'],
                    datasets: [{
                        data: [
                            result.dropout_probability * 100,
                            result.graduate_probability * 100
                        ],
                        backgroundColor: [
                            '#dc3545',
                            '#198754'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom',
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.label}: ${context.raw.toFixed(1)}%`;
                                }
                            }
                        }
                    }
                }
            });
        }
    }
});
