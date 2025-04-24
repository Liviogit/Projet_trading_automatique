// Function to highlight the active navigation link
document.addEventListener('DOMContentLoaded', function() {
    // Function to update active link
    function updateActiveLink() {
        const currentPath = window.location.pathname;
        
        // Remove active class from all links
        document.querySelectorAll('.nav-link, .mobile-nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        // Add active class to current path links
        document.querySelectorAll(`.nav-link[href="${currentPath}"], .mobile-nav-link[href="${currentPath}"]`).forEach(link => {
            link.classList.add('active');
        });
    }
    
    // Initial call
    updateActiveLink();
    
    // Set up a MutationObserver to detect URL changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                updateActiveLink();
            }
        });
    });
    
    // Start observing the page-content div for changes
    const targetNode = document.getElementById('page-content');
    if (targetNode) {
        observer.observe(targetNode, { childList: true });
    }
    
    // Close mobile menu when a link is clicked
    document.querySelectorAll('.mobile-nav-link').forEach(link => {
        link.addEventListener('click', function() {
            document.getElementById('mobile-nav').style.display = 'none';
        });
    });
});