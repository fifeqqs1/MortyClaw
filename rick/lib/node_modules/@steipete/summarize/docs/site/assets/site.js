const canonicalHost = 'summarize.sh'
const redirectHostnames = new Set(['summarize.is', 'www.summarize.is'])

const maybeRedirect = () => {
  try {
    const { hostname, pathname, search, hash } = window.location
    if (!redirectHostnames.has(hostname)) return
    const target = `https://${canonicalHost}${pathname}${search}${hash}`
    window.location.replace(target)
  } catch {
    // ignore
  }
}

const highlightNav = () => {
  const path = window.location.pathname
  const isDocs = path.includes('/docs/')
  const navDocs = document.querySelector('a[data-nav="docs"]')
  const navHome = document.querySelector('a[data-nav="home"]')
  if (navDocs && isDocs) navDocs.setAttribute('aria-current', 'page')
  if (navHome && !isDocs) navHome.setAttribute('aria-current', 'page')

  const sideLinks = document.querySelectorAll('.side a[href]')
  for (const a of sideLinks) {
    const href = a.getAttribute('href') ?? ''
    if (!href) continue
    const normalized = href.replace(/^\.\//, '')
    if (path.endsWith(normalized)) a.setAttribute('aria-current', 'page')
  }
}

const wireCopyButtons = () => {
  const buttons = document.querySelectorAll('[data-copy]')
  for (const button of buttons) {
    button.addEventListener('click', async () => {
      const selector = button.getAttribute('data-copy')
      const target = selector ? document.querySelector(selector) : null
      const text = target?.textContent?.trim() ?? ''
      if (!text) return
      try {
        await navigator.clipboard.writeText(text)
        const prev = button.textContent ?? ''
        button.textContent = 'Copied'
        button.setAttribute('data-copied', '1')
        window.setTimeout(() => {
          button.textContent = prev
          button.removeAttribute('data-copied')
        }, 900)
      } catch {
        // ignore
      }
    })
  }
}

const reveal = () => {
  const items = document.querySelectorAll('.reveal')
  let i = 0
  for (const el of items) {
    const delay = Math.min(380, i * 70)
    window.setTimeout(() => el.classList.add('is-on'), delay)
    i++
  }
}

maybeRedirect()
highlightNav()
wireCopyButtons()
reveal()
