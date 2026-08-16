/** Design reminder: Morrow Field editorial field notes — warm paper, mineral green, cartographic lines, asymmetric folio layout. Never use Nexuss-auth branding in this page. */
import { createAuth, type NexAuthUser } from "nexuss-auth";
import { ArrowUpRight, Check, Github, LockKeyhole, LogOut, MapPin, Menu, Sparkles, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const AUTH_URL = import.meta.env.VITE_NEXUSS_AUTH_URL ?? "https://nexuss-auth.vercel.app";
const PROJECT_ID = import.meta.env.VITE_NEXUSS_AUTH_PROJECT_ID ?? "morrow-field-demo";

const projects = [
  {
    index: "01",
    title: "Common Ground",
    kind: "Civic archive",
    year: "2024",
    description: "A patient way to explore oral histories, neighborhood maps, and the people who keep a city legible.",
    tone: "sage",
  },
  {
    index: "02",
    title: "Tide Table",
    kind: "Climate journal",
    year: "2023",
    description: "A field guide for coastal change, built to make slow-moving data feel close enough to touch.",
    tone: "clay",
  },
  {
    index: "03",
    title: "Afterlight",
    kind: "Arts platform",
    year: "2022",
    description: "An editorial home for artists working between memory, material, and the edge of the day.",
    tone: "ochre",
  },
];

function StatusMark({ active }: { active: boolean }) {
  return <span className={`status-mark ${active ? "status-mark-active" : ""}`} aria-hidden="true" />;
}

export default function Portfolio() {
  const [user, setUser] = useState<NexAuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<"google" | "github" | "logout" | null>(null);
  const [notice, setNotice] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  const auth = useMemo(() => {
    if (!PROJECT_ID) return null;
    try {
      return createAuth({ projectId: PROJECT_ID, authUrl: AUTH_URL });
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    if (!auth) {
      setLoading(false);
      return;
    }
    void auth.getUser().then((nextUser) => {
      if (mounted) setUser(nextUser);
    }).catch(() => {
      if (mounted) setNotice("The field station could not be reached. Try again in a moment.");
    }).finally(() => {
      if (mounted) setLoading(false);
    });
    return () => { mounted = false; };
  }, [auth]);

  const signIn = (provider: "google" | "github") => {
    if (!auth) {
      setNotice("The field station is not connected to its registered project yet.");
      return;
    }
    setBusy(provider);
    setNotice("");
    try {
      auth.signIn(provider, { redirectUri: window.location.href });
    } catch {
      setBusy(null);
      setNotice("This sign-in route is not ready yet. Check the registered callback address.");
    }
  };

  const logout = async () => {
    if (!auth) return;
    setBusy("logout");
    try {
      await auth.logout();
      setUser(null);
      setNotice("You have left the field station.");
    } catch {
      setNotice("We could not close the session. Please try again.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="morrow-site">
      <div className="contour contour-one" aria-hidden="true" />
      <div className="contour contour-two" aria-hidden="true" />
      <header className="morrow-header">
        <a className="morrow-wordmark" href="#top" aria-label="Morrow Field home">
          <span className="morrow-mark" aria-hidden="true"><span /></span>
          <span>Morrow Field</span>
        </a>
        <nav className={menuOpen ? "morrow-nav morrow-nav-open" : "morrow-nav"} aria-label="Primary navigation">
          <a href="#work" onClick={() => setMenuOpen(false)}>Selected work</a>
          <a href="#approach" onClick={() => setMenuOpen(false)}>Approach</a>
          <a href="#contact" onClick={() => setMenuOpen(false)}>Contact</a>
        </nav>
        <button className="morrow-menu" onClick={() => setMenuOpen((open) => !open)} aria-label={menuOpen ? "Close menu" : "Open menu"}>
          {menuOpen ? <X size={18} /> : <Menu size={18} />}
        </button>
        <div className="session-slot">
          {loading ? <span className="session-loading">Checking station…</span> : user ? (
            <div className="session-user">
              {user.avatarUrl ? <img src={user.avatarUrl} alt="" /> : <span className="session-initial">{(user.name ?? user.email ?? "M").slice(0, 1).toUpperCase()}</span>}
              <span>{user.name ?? user.email ?? "Signed in"}</span>
              <button onClick={() => void logout()} disabled={busy === "logout"} aria-label="Sign out"><LogOut size={14} /></button>
            </div>
          ) : <a className="session-link" href="#access"><LockKeyhole size={14} /> Client access</a>}
        </div>
      </header>

      <main id="top">
        <section className="morrow-hero">
          <div className="hero-kicker"><span className="folio-line" /> Independent digital practice / 2018—24</div>
          <div className="hero-grid">
            <div className="hero-copy">
              <p className="eyebrow">Designing for the spaces between people</p>
              <h1>Interfaces with <em>somewhere</em> to go.</h1>
              <p className="hero-lede">Morrow Field is the independent practice of Mara Vale, building digital places for culture, climate, and the curious public.</p>
              <a className="text-link" href="#work">Open the field notes <ArrowUpRight size={16} /></a>
            </div>
            <div className="hero-plate" aria-label="Abstract topographic field illustration">
              <div className="plate-stamp">MF / 24</div>
              <div className="plate-sun" />
              <div className="plate-path path-a" />
              <div className="plate-path path-b" />
              <div className="plate-path path-c" />
              <span className="plate-caption">A study in<br />soft systems</span>
            </div>
          </div>
          <div className="hero-footer"><span><MapPin size={14} /> Working between coast &amp; city</span><span>Scroll to wander <span className="scroll-arrow">↓</span></span></div>
        </section>

        <section className="morrow-section work-section" id="work">
          <div className="section-rail"><span>01</span><span>Selected work</span></div>
          <div className="section-body">
            <div className="section-intro"><p className="eyebrow">A small archive of ongoing questions</p><h2>Work that leaves<br /><em>a little room.</em></h2><p>Each engagement begins with listening, then moves toward a system that makes people feel more capable in the world.</p></div>
            <div className="project-list">{projects.map((project) => <article className="project-plate" key={project.index}><div className={`project-art project-art-${project.tone}`}><span>{project.index}</span><div className="project-orbit" /></div><div className="project-meta"><div><p className="eyebrow">{project.kind} / {project.year}</p><h3>{project.title}</h3></div><p>{project.description}</p><a className="arrow-link" href="#contact" aria-label={`Read about ${project.title}`}><ArrowUpRight size={18} /></a></div></article>)}</div>
          </div>
        </section>

        <section className="morrow-section approach-section" id="approach">
          <div className="section-rail"><span>02</span><span>Approach</span></div>
          <div className="approach-body"><div><p className="eyebrow">The useful kind of beautiful</p><h2>Make the complex<br /><em>feel held.</em></h2></div><div className="approach-notes"><p>Good digital work does not call attention to its cleverness. It gives people a clear next step, a sense of belonging, and enough quiet to notice what matters.</p><div className="note-row"><span>01</span><strong>Listen for the real shape</strong></div><div className="note-row"><span>02</span><strong>Build a generous system</strong></div><div className="note-row"><span>03</span><strong>Leave a clear trace</strong></div></div></div>
        </section>

        <section className="access-section" id="access">
          <div className="access-card"><div className="access-mark"><Sparkles size={18} /></div><div><p className="eyebrow">Private client room</p><h2>Keep the work<br /><em>in the room.</em></h2><p>Access project notes and shared drafts through the secure client station.</p></div><div className="access-actions"><button className="provider-button provider-google" onClick={() => signIn("google")} disabled={busy !== null}><span className="provider-symbol google-g">G</span><span className="provider-copy"><strong>{busy === "google" ? "Opening Google…" : "Google"}</strong><small>Trusted client access</small></span><ArrowUpRight size={15} /></button><button className="provider-button provider-github" onClick={() => signIn("github")} disabled={busy !== null}><span className="provider-symbol"><Github size={16} /></span><span className="provider-copy"><strong>{busy === "github" ? "Opening GitHub…" : "GitHub"}</strong><small>Trusted client access</small></span><ArrowUpRight size={15} /></button><p className="access-note"><StatusMark active={Boolean(auth)} /> {auth ? "Connected to the client station" : "Project connection required"}</p>{notice && <p className="access-notice" role="status">{notice}</p>}{user && <p className="signed-note"><Check size={14} /> Verified for {user.email ?? user.name ?? "this account"}</p>}</div></div>
        </section>
      </main>

      <footer className="morrow-footer" id="contact"><div><span className="morrow-wordmark"><span className="morrow-mark" aria-hidden="true"><span /></span><span>Morrow Field</span></span><p>Digital places for curious people.</p></div><div className="footer-contact"><span>Available for a few 2025 collaborations</span><a href="mailto:hello@morrowfield.studio">hello@morrowfield.studio <ArrowUpRight size={14} /></a></div><span className="footer-index">MF / 24—25</span></footer>
    </div>
  );
}
