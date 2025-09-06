// Material UI Timeline initialization script
// Provides interactivity for timeline elements matching MUI Timeline behavior

document.addEventListener("nav", function() {
  // Initialize all timeline elements
  const timelines = document.querySelectorAll('.timeline');
  
  timelines.forEach(timeline => {
    const timelineItems = timeline.querySelectorAll('.timeline-item');
    
    timelineItems.forEach((item) => {
      const dot = item.querySelector('.timeline-dot');
      const content = item.querySelector('.timeline-content');
      
      if (dot && content) {
        // Add click handler for timeline dots (MUI style interaction)
        const clickHandler = function() {
          // Smooth scroll to content
          content.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          });
          
          // Add temporary highlight effect
          const contentEl = content as HTMLElement;
          const originalBg = getComputedStyle(contentEl).backgroundColor;
          contentEl.style.transition = 'background-color 0.3s ease';
          contentEl.style.backgroundColor = '#BBDEFB'; // Light blue highlight
          
          setTimeout(() => {
            contentEl.style.backgroundColor = originalBg;
          }, 1000);
        };
        
        dot.addEventListener('click', clickHandler);
        
        // Add hover effects to dots
        const dotEl = dot as HTMLElement;
        dotEl.style.cursor = 'pointer';
        
        // Add ripple effect on click (Material UI style)
        const rippleHandler = function() {
          const ripple = document.createElement('div');
          ripple.classList.add('timeline-ripple');
          ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.5);
            transform: scale(0);
            animation: timeline-ripple 0.6s linear;
            width: 24px;
            height: 24px;
            left: -6px;
            top: -6px;
            pointer-events: none;
          `;
          
          dotEl.style.position = 'relative';
          dotEl.appendChild(ripple);
          
          setTimeout(() => {
            ripple.remove();
          }, 600);
        };
        
        dot.addEventListener('click', rippleHandler);
        
        // Cleanup event listeners on navigation
        window.addCleanup(() => {
          dot.removeEventListener('click', clickHandler);
          dot.removeEventListener('click', rippleHandler);
        });
      }
    });
  });
  
  // Add CSS animation for ripple effect (only once)
  if (!document.querySelector('#timeline-ripple-styles')) {
    const style = document.createElement('style');
    style.id = 'timeline-ripple-styles';
    style.textContent = `
      @keyframes timeline-ripple {
        to {
          transform: scale(2);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);
  }
});
