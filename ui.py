"""Push-to-talk window: hold SPACE to talk to Skelly; the screen shows what he is doing."""

from __future__ import annotations

import asyncio
import tkinter as tk

STATES = {
    "idle": ("Hold  SPACE  to talk to Skelly", "#1b1b2f", "#d8d8e8"),
    "listening": ("Skelly is listening…", "#0f6b2f", "#ffffff"),
    "thinking": ("Skelly is thinking…", "#8a6d00", "#ffffff"),
    "talking": ("Skelly is talking", "#8b1e1e", "#ffffff"),
    "waiting": ("Skelly is listening for voices…", "#1f4a6b", "#ffffff"),
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
        self.level_source = lambda: (0.0, 1.0)  # talk.run: () -> (mic rms, speech threshold)
        self.handsfree = False
        self.mode_changed = asyncio.Event()
        bottom = tk.Frame(self.root, highlightthickness=0)
        bottom.pack(side="bottom", fill="x", padx=20, pady=16)
        self.bottom = bottom
        self.meter = tk.Canvas(bottom, width=260, height=14, highlightthickness=0)
        self.meter.pack(side="left")
        self.mode_button = tk.Button(bottom, text="Hands-free: off", font=("Helvetica", 14), takefocus=0,
                                     highlightthickness=0, command=self._toggle_mode)
        self.mode_button.pack(side="left", padx=16)
        self.on_model = lambda model_id: None  # talk.run: switch the chat model live
        self.models: dict[str, str] = {}
        self.model_var = tk.StringVar(value="")
        self.model_menu = tk.OptionMenu(bottom, self.model_var, "")
        self.model_menu.configure(takefocus=0, highlightthickness=0, font=("Helvetica", 14))
        self.model_menu.pack(side="left", padx=16)
        # takefocus=0 so the button never owns the keyboard: SPACE must stay push-to-talk, not "press button"
        self.clear_button = tk.Button(bottom, text="Clear memory", font=("Helvetica", 14), takefocus=0,
                                      highlightthickness=0, command=self._clear)
        self.clear_button.pack(side="right")
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

    def set_models(self, models: dict[str, str], current_id: str) -> None:
        """Offer the chat models by label; selecting one calls on_model with its id."""
        self.models = models
        menu = self.model_menu["menu"]
        menu.delete(0, "end")
        for label in models:
            menu.add_command(label=label, command=lambda l=label: self._choose_model(l))
        self.model_var.set(next((l for l, m in models.items() if m == current_id), next(iter(models))))

    def _choose_model(self, label: str) -> None:
        self.model_var.set(label)
        self.on_model(self.models[label])
        self.root.focus_force()

    def _toggle_mode(self) -> None:
        self.handsfree = not self.handsfree
        self.mode_button.configure(text=f"Hands-free: {'on' if self.handsfree else 'off'}")
        self.mode_changed.set()
        self.root.focus_force()

    def _draw_meter(self) -> None:
        level, threshold = self.level_source()
        fill = min(1.0, level / (threshold * 2.5))  # threshold sits at 40% of the bar
        self.meter.delete("all")
        self.meter.create_rectangle(0, 0, 260, 14, fill="#000000", outline="")
        self.meter.create_rectangle(0, 0, int(260 * fill), 14, fill="#3ddc84" if level >= threshold else "#5a5a7a", outline="")
        self.meter.create_line(104, 0, 104, 14, fill="#ffffff")

    def _clear(self) -> None:
        self.on_clear()
        self.heard.configure(text="Memory cleared. Skelly forgot everything.")
        self.said.configure(text="")
        self.root.focus_force()

    def set_state(self, state: str, heard: str | None = None, said: str | None = None) -> None:
        text, bg, fg = STATES[state]
        self.status.configure(text=text, bg=bg, fg=fg)
        self.root.configure(bg=bg)
        self.bottom.configure(bg=bg)
        for label, value in ((self.heard, heard), (self.said, said)):
            label.configure(bg=bg, fg=fg)
            if value is not None:
                label.configure(text=value)

    async def pump(self) -> None:
        """Drive Tk from the asyncio loop until the window closes."""
        while not self.closed.is_set():
            try:
                self._draw_meter()
                self.root.update()
            except tk.TclError:
                break
            await asyncio.sleep(0.02)
        self.closed.set()
