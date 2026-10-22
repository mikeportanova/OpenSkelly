"""Push-to-talk window: hold SPACE to talk to Skelly; the screen shows what he is doing."""

from __future__ import annotations

import asyncio
import tkinter as tk

STATES = {
    "idle": ("Hold  SPACE  to talk to Skelly", "#1b1b2f", "#d8d8e8"),
    "listening": ("Skelly is listening…", "#0f6b2f", "#ffffff"),
    "thinking": ("Skelly is thinking…", "#8a6d00", "#ffffff"),
    "talking": ("Skelly is talking", "#8b1e1e", "#ffffff"),
}


class TalkWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Skelly")
        self.root.geometry("900x520")
        self.root.configure(bg=STATES["idle"][1])
        self.status = tk.Label(self.root, text="", font=("Helvetica", 44, "bold"), wraplength=840)
        self.status.pack(expand=True, fill="both", padx=30, pady=(40, 10))
        self.heard = tk.Label(self.root, text="", font=("Helvetica", 22), wraplength=840, justify="center")
        self.heard.pack(fill="x", padx=30)
        self.said = tk.Label(self.root, text="", font=("Helvetica", 22, "italic"), wraplength=840, justify="center")
        self.said.pack(fill="x", padx=30, pady=(10, 40))
        self.on_clear = lambda: None  # talk.run points this at Brain.reset
        # takefocus=0 so the button never owns the keyboard: SPACE must stay push-to-talk, not "press button"
        self.clear_button = tk.Button(self.root, text="Clear memory", font=("Helvetica", 14), takefocus=0,
                                      highlightthickness=0, command=self._clear)
        self.clear_button.pack(side="bottom", anchor="e", padx=20, pady=16)
        self.down = False
        self.pending_release: str | None = None
        self.pressed = asyncio.Event()
        self.released = asyncio.Event()
        self.closed = asyncio.Event()
        self.root.bind("<KeyPress-space>", self._press)
        self.root.bind("<KeyRelease-space>", self._release)
        self.root.bind("<Escape>", lambda _e: self.root.attributes("-fullscreen", False))
        self.root.bind("<F>", lambda _e: self.root.attributes("-fullscreen", True))
        self.root.protocol("WM_DELETE_WINDOW", self.closed.set)
        self.set_state("idle")
        self.root.lift()
        self.root.focus_force()

    def _press(self, _event: tk.Event) -> None:
        if self.pending_release is not None:  # key auto-repeat: release+press pairs while held
            self.root.after_cancel(self.pending_release)
            self.pending_release = None
        if not self.down:
            self.down = True
            self.released.clear()
            self.pressed.set()

    def _release(self, _event: tk.Event) -> None:
        if self.pending_release is None:
            self.pending_release = self.root.after(80, self._confirm_release)

    def _confirm_release(self) -> None:
        self.pending_release = None
        if self.down:
            self.down = False
            self.pressed.clear()
            self.released.set()

    def _clear(self) -> None:
        self.on_clear()
        self.heard.configure(text="Memory cleared. Skelly forgot everything.")
        self.said.configure(text="")
        self.root.focus_force()

    def set_state(self, state: str, heard: str | None = None, said: str | None = None) -> None:
        text, bg, fg = STATES[state]
        self.status.configure(text=text, bg=bg, fg=fg)
        self.root.configure(bg=bg)
        for label, value in ((self.heard, heard), (self.said, said)):
            label.configure(bg=bg, fg=fg)
            if value is not None:
                label.configure(text=value)

    async def pump(self) -> None:
        """Drive Tk from the asyncio loop until the window closes."""
        while not self.closed.is_set():
            try:
                self.root.update()
            except tk.TclError:
                break
            await asyncio.sleep(0.02)
        self.closed.set()
