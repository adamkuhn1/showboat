// Embed mode (`?embed=1`): the portfolio site shows Showboat inside an iframe.
//
// Messages follow the portfolio's existing postMessage contract
// (portfolio src/lib/embedProtocol.ts, `EmbedInbound`): every message is
// `{ source: "portfolio-embed", type, id }`, and the protocol already defines
// the height report as `type: "resize"` with `height` in CSS px. Nothing here
// adds a message type the host doesn't know.

export const EMBED_SOURCE = "portfolio-embed" as const;
export const EMBED_ID = "showboat";

export const isEmbedMode = (): boolean =>
  new URLSearchParams(window.location.search).get("embed") === "1";

const inFrame = (): boolean => window.parent !== window;

type Inbound =
  | { source: typeof EMBED_SOURCE; type: "ready"; id: string }
  | { source: typeof EMBED_SOURCE; type: "resize"; id: string; height: number };

const post = (msg: Inbound): void => {
  if (!inFrame()) return;
  window.parent.postMessage(msg, "*");
};

export const postReady = (): void =>
  post({ source: EMBED_SOURCE, type: "ready", id: EMBED_ID });

const contentHeight = (el: HTMLElement): number =>
  Math.ceil(el.getBoundingClientRect().height);

let lastHeight = -1;
// Report `el`'s rendered height to the host. Deduplicated, so the host only
// hears about real changes; `force` re-sends the current value (used together
// with the ready message so the host can size the frame immediately).
export const postHeight = (el: HTMLElement, force = false): void => {
  const height = contentHeight(el);
  if (!force && height === lastHeight) return;
  lastHeight = height;
  post({ source: EMBED_SOURCE, type: "resize", id: EMBED_ID, height });
};

// Report the content height after every layout change. The observed element
// must size to its content (embed CSS drops the 100%-height chain), otherwise
// a host that grows the frame would grow the reported height with it and it
// could never shrink again.
export const observeHeight = (el: HTMLElement): (() => void) => {
  const ro = new ResizeObserver(() => postHeight(el));
  ro.observe(el);
  postHeight(el);
  return () => ro.disconnect();
};
