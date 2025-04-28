// assets/custom.js
document.addEventListener('DOMContentLoaded',function(){
    // Active sidebar link
    function updateActive(){
        const p=window.location.pathname;
        document.querySelectorAll('.sidebar-link').forEach(l=>l.classList.remove('active'));
        document.querySelectorAll(`.sidebar-link[href="${p}"]`).forEach(l=>l.classList.add('active'));
    }
    updateActive();
    new MutationObserver(updateActive).observe(document.getElementById('page-content'),{childList:true});

    // Tilt effect
    function tiltMove(e){
        if(window.matchMedia('(prefers-reduced-motion:reduce)').matches) return;
        const el=this.getBoundingClientRect(), x=e.clientX-el.left, y=e.clientY-el.top;
        const rx=((y-el.height/2)/(el.height/2))*-4, ry=((x-el.width/2)/(el.width/2))*4;
        this.style.transform=`perspective(1000px) rotateX(${rx}deg) rotateY(${ry}deg)`;
    }
    function tiltReset(){ this.style.transform='perspective(1000px) rotateX(0) rotateY(0)'; }
    function initTilt(){
        document.querySelectorAll('.tiltable').forEach(e=>{
            if(!e.dataset.tilt){ e.dataset.tilt=1;
                e.addEventListener('mousemove',tiltMove);
                e.addEventListener('mouseleave',tiltReset);
            }
        });
    }
    initTilt();
    new MutationObserver(initTilt).observe(document.body,{childList:true,subtree:true});

    // KPI pulse
    document.querySelectorAll('.kpi-card').forEach(c=>{
        c.classList.add('pulse');
        c.addEventListener('animationend',()=>c.classList.remove('pulse'));
    });
});
