// Function to display uploaded image preview
document.addEventListener('DOMContentLoaded', function() {
    // For disease detection page
    const imageInput = document.getElementById('image');
    if (imageInput) {
        imageInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                }
                reader.readAsDataURL(file);
            }
        });
    }
    
    // For weather forecast chart
    if (typeof forecastData !== 'undefined') {
        // Initialize Plotly chart
        // This would be implemented if Plotly.js is included
        // For simplicity, we're not including the full implementation here
        console.log('Forecast data available for charting:', forecastData);
    }
});