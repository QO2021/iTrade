// iTrade.com JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add animation to cards on scroll
    const observeCards = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    });

    document.querySelectorAll('.card').forEach(card => {
        observeCards.observe(card);
    });

    // Real-time clock update
    function updateClock() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour12: true,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        const clockElements = document.querySelectorAll('.live-clock');
        clockElements.forEach(element => {
            element.textContent = timeString;
        });
    }

    // Update clock every second
    setInterval(updateClock, 1000);
    updateClock(); // Initial call

    // Market status simulation
    function updateMarketStatus() {
        const now = new Date();
        const hour = now.getHours();
        const day = now.getDay();
        
        // Simplified market hours: weekdays 9:30 AM - 4:00 PM EST
        const isWeekday = day >= 1 && day <= 5;
        const isMarketHours = hour >= 9 && hour < 16;
        const isOpen = isWeekday && isMarketHours;
        
        const statusElements = document.querySelectorAll('.market-status');
        statusElements.forEach(element => {
            element.textContent = isOpen ? 'OPEN' : 'CLOSED';
            element.className = isOpen ? 'market-open' : 'market-closed';
        });
    }

    updateMarketStatus();
    setInterval(updateMarketStatus, 60000); // Update every minute

    // Form validation enhancements
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });

    // Stock symbol formatting
    const stockInputs = document.querySelectorAll('input[name="symbol"]');
    stockInputs.forEach(input => {
        input.addEventListener('input', function() {
            this.value = this.value.toUpperCase();
        });
    });

    // Price formatting
    function formatPrice(price) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2
        }).format(price);
    }

    // Percentage formatting
    function formatPercentage(value) {
        const sign = value >= 0 ? '+' : '';
        return sign + value.toFixed(2) + '%';
    }

    // Copy to clipboard functionality
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(function() {
            // Show success message
            const toast = document.createElement('div');
            toast.className = 'toast align-items-center text-white bg-success border-0';
            toast.innerHTML = `
                <div class="d-flex">
                    <div class="toast-body">
                        Copied to clipboard!
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            `;
            document.body.appendChild(toast);
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();
            
            // Remove toast after it's hidden
            toast.addEventListener('hidden.bs.toast', function() {
                toast.remove();
            });
        });
    }

    // Add copy functionality to elements with copy-text class
    document.querySelectorAll('.copy-text').forEach(element => {
        element.addEventListener('click', function() {
            copyToClipboard(this.textContent);
        });
    });

    // Loading state management
    function showLoading(button) {
        const originalText = button.innerHTML;
        button.innerHTML = '<span class="spinner"></span> Loading...';
        button.disabled = true;
        
        return function hideLoading() {
            button.innerHTML = originalText;
            button.disabled = false;
        };
    }

    // Add loading states to form submissions
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            const submitButton = form.querySelector('[type="submit"]');
            if (submitButton) {
                showLoading(submitButton);
            }
        });
    });

    // Theme toggle (if implemented)
    const themeToggle = document.querySelector('#theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-theme');
            localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
        });

        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
        }
    }

    // Search functionality enhancements
    const searchInputs = document.querySelectorAll('.stock-search');
    searchInputs.forEach(input => {
        let searchTimeout;
        
        input.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.trim();
            
            if (query.length > 0) {
                searchTimeout = setTimeout(() => {
                    // Simulate API call delay
                    performStockSearch(query, this);
                }, 300);
            }
        });
    });

    function performStockSearch(query, inputElement) {
        // This would be replaced with actual API call in production
        const mockResults = [
            { symbol: 'AAPL', name: 'Apple Inc.' },
            { symbol: 'GOOGL', name: 'Alphabet Inc.' },
            { symbol: 'MSFT', name: 'Microsoft Corporation' },
            { symbol: 'TSLA', name: 'Tesla Inc.' },
            { symbol: 'AMZN', name: 'Amazon.com Inc.' }
        ];

        const filteredResults = mockResults.filter(stock => 
            stock.symbol.includes(query.toUpperCase()) || 
            stock.name.toLowerCase().includes(query.toLowerCase())
        );

        displaySearchResults(filteredResults, inputElement);
    }

    function displaySearchResults(results, inputElement) {
        // Remove existing results
        const existingResults = document.querySelector('.search-results');
        if (existingResults) {
            existingResults.remove();
        }

        if (results.length === 0) return;

        // Create results container
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'search-results position-absolute bg-white border rounded shadow-sm w-100';
        resultsContainer.style.zIndex = '1000';
        resultsContainer.style.top = '100%';
        resultsContainer.style.left = '0';

        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'p-2 border-bottom search-result-item';
            resultItem.style.cursor = 'pointer';
            resultItem.innerHTML = `
                <div class="fw-bold">${result.symbol}</div>
                <small class="text-muted">${result.name}</small>
            `;
            
            resultItem.addEventListener('click', function() {
                inputElement.value = result.symbol;
                resultsContainer.remove();
            });

            resultItem.addEventListener('mouseenter', function() {
                this.style.backgroundColor = '#f8f9fa';
            });

            resultItem.addEventListener('mouseleave', function() {
                this.style.backgroundColor = 'transparent';
            });

            resultsContainer.appendChild(resultItem);
        });

        // Position relative to input
        const inputRect = inputElement.getBoundingClientRect();
        inputElement.parentNode.style.position = 'relative';
        inputElement.parentNode.appendChild(resultsContainer);

        // Close results when clicking outside
        document.addEventListener('click', function closeResults(e) {
            if (!resultsContainer.contains(e.target) && e.target !== inputElement) {
                resultsContainer.remove();
                document.removeEventListener('click', closeResults);
            }
        });
    }

    // Initialize any chart animations
    const chartElements = document.querySelectorAll('[id$="-chart"]');
    chartElements.forEach(chart => {
        chart.classList.add('slide-up');
    });

    console.log('iTrade.com JavaScript initialized successfully');
});