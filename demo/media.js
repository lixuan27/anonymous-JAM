(() => {
  const players = [...document.querySelectorAll('.video-card video')];
  const objectURLs = new Set();

  for (const video of players) {
    const source = video.querySelector('source');
    if (!source) continue;
    const url = source.src;
    const caption = video.closest('figure').querySelector('figcaption');
    const task = video.closest('.real-task')?.querySelector('h3')?.textContent;
    const description = caption ? [...caption.children].map(item => item.textContent).join(': ') : '';
    const label = [task, description].filter(Boolean).join(': ');
    video.setAttribute('aria-label', label);
    video.controls = false;
    video.setAttribute('aria-hidden', 'true');
    source.remove();
    video.load();

    const wrapper = document.createElement('div');
    wrapper.className = 'video-player';
    video.before(wrapper);
    wrapper.append(video);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'video-play';
    button.textContent = 'Play video';
    button.setAttribute('aria-label', `Play ${label}`);
    wrapper.append(button);
    const status = document.createElement('span');
    status.className = 'video-status';
    status.setAttribute('role', 'status');
    wrapper.append(status);
    let loaded = false;
    let busy = false;

    button.addEventListener('click', async () => {
      if (busy) return;
      busy = true;
      button.disabled = true;
      button.textContent = 'Loading…';
      status.textContent = '';
      try {
        if (!loaded) {
          if (location.protocol === 'file:') {
            video.src = url;
          } else {
            // A single download also supports hosts without byte-range responses.
            const response = await fetch(url, { credentials: 'omit', referrerPolicy: 'no-referrer' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.blob();
            if (!data.size || !data.type.startsWith('video/')) throw new Error('Invalid video response');
            const objectURL = URL.createObjectURL(data);
            objectURLs.add(objectURL);
            video.src = objectURL;
          }
          loaded = true;
        }
        video.controls = true;
        video.removeAttribute('aria-hidden');
        await video.play();
        button.hidden = true;
      } catch (error) {
        video.controls = loaded;
        button.textContent = 'Retry playback';
        status.textContent = 'Video unavailable. Please try again shortly.';
      } finally {
        busy = false;
        button.disabled = false;
      }
    });
    video.addEventListener('play', () => {
      for (const other of players) if (other !== video) other.pause();
    });
  }

  window.addEventListener('pagehide', (event) => {
    if (!event.persisted) for (const url of objectURLs) URL.revokeObjectURL(url);
  });
})();
