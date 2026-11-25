"""
camera_gui.py: Modern Tkinter GUI for Selecting Camera & Input Source with Live Preview.

Allows users to visually scan connected webcams, preview live feeds (with Mirror toggle),
or browse video / image / dataset folders before launching the BISINDO pipeline.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Union
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


def probe_cameras(max_to_test: int = 6) -> List[Dict[str, any]]:
    """
    Scans indices 0..max_to_test-1 on Windows using DirectShow and returns
    a list of working cameras with their index, resolution, and display label.
    """
    found: List[Dict[str, any]] = []
    for idx in range(max_to_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            found.append({
                "index": idx,
                "width": w,
                "height": h,
                "label": f"Kamera {idx} ({w}x{h})"
            })
        cap.release()
    return found


class InputSelectorApp:
    PREVIEW_W = 460
    PREVIEW_H = 258

    def __init__(
        self,
        root: tk.Tk,
        allow_files: bool = True,
        title: str = "BISINDO Translator - Pilih Input Kamera",
        action_text: str = "Mulai Penerjemah BISINDO"
    ):
        self.root = root
        self.allow_files = allow_files
        self.action_text = action_text
        self.result: Optional[Dict[str, any]] = None

        self.cameras: List[Dict[str, any]] = []
        self.preview_cap: Optional[cv2.VideoCapture] = None
        self.preview_job: Optional[str] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self.mode_var = tk.StringVar(value="webcam")
        self.cam_var = tk.StringVar(value="")
        self.mirror_var = tk.BooleanVar(value=False)
        self.video_path_var = tk.StringVar(value="")
        self.image_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Memindai kamera...")

        self._setup_window(title)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Defer initial camera scan slightly so window renders immediately
        self.root.after(100, self._refresh_cameras)

    def _setup_window(self, title: str):
        self.root.title(title)
        win_w, win_h = 540, 620 if self.allow_files else 550
        self.root.configure(bg="#18181b")
        self.root.resizable(False, False)

        # Center on screen
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        pos_x = max(0, (scr_w - win_w) // 2)
        pos_y = max(0, (scr_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Dark.TCombobox",
            fieldbackground="#27272a",
            background="#3f3f46",
            foreground="#f4f4f5",
            arrowcolor="#10b981",
            bordercolor="#3f3f46"
        )

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#09090b", pady=12, padx=18)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Penerjemah BISINDO Real-Time",
            font=("Segoe UI", 14, "bold"),
            fg="#10b981",
            bg="#09090b"
        ).pack(anchor="w")
        tk.Label(
            header,
            text="MediaPipe 42 Keypoints + YOLOv11n-cls + Landmark MLP",
            font=("Segoe UI", 9),
            fg="#a1a1aa",
            bg="#09090b"
        ).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self.root, bg="#18181b", padx=18, pady=12)
        body.pack(fill="both", expand=True)

        # Mode Selector (Tabs)
        if self.allow_files:
            mode_card = tk.Frame(body, bg="#27272a", padx=10, pady=8)
            mode_card.pack(fill="x", pady=(0, 10))
            tk.Label(
                mode_card, text="Jenis Input:", font=("Segoe UI", 9, "bold"),
                fg="#e4e4e7", bg="#27272a"
            ).pack(side="left", padx=(0, 10))

            for val, txt in [("webcam", "Kamera Webcam"), ("video", "File Video"), ("image", "Citra / Folder")]:
                rb = tk.Radiobutton(
                    mode_card, text=txt, variable=self.mode_var, value=val,
                    command=self._on_mode_change,
                    font=("Segoe UI", 9), fg="#f4f4f5", bg="#27272a",
                    selectcolor="#09090b", activebackground="#27272a",
                    activeforeground="#10b981", cursor="hand2"
                )
                rb.pack(side="left", padx=6)

        # Dynamic Controls Frame
        self.controls_card = tk.Frame(body, bg="#27272a", padx=12, pady=10)
        self.controls_card.pack(fill="x", pady=(0, 10))

        # --- Webcam Controls ---
        self.webcam_frame = tk.Frame(self.controls_card, bg="#27272a")
        row_cam = tk.Frame(self.webcam_frame, bg="#27272a")
        row_cam.pack(fill="x")
        tk.Label(
            row_cam, text="Pilih Kamera:", font=("Segoe UI", 9, "bold"),
            fg="#e4e4e7", bg="#27272a"
        ).pack(side="left", padx=(0, 8))

        self.cam_combo = ttk.Combobox(
            row_cam, textvariable=self.cam_var, state="readonly",
            style="Dark.TCombobox", width=28, font=("Segoe UI", 9)
        )
        self.cam_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.cam_combo.bind("<<ComboboxSelected>>", lambda _e: self._start_camera_preview())

        tk.Button(
            row_cam, text="Scan Ulang", command=self._refresh_cameras,
            font=("Segoe UI", 9, "bold"), bg="#3f3f46", fg="#f4f4f5",
            activebackground="#52525b", activeforeground="#ffffff",
            relief="flat", padx=10, pady=4, cursor="hand2"
        ).pack(side="right")

        tk.Checkbutton(
            self.webcam_frame,
            text="Mirror Kamera (Flip Horizontal - disarankan jika kamera terbalik)",
            variable=self.mirror_var,
            font=("Segoe UI", 9), fg="#d4d4d8", bg="#27272a",
            selectcolor="#09090b", activebackground="#27272a",
            activeforeground="#10b981", cursor="hand2"
        ).pack(anchor="w", pady=(8, 0))

        # --- Video Controls ---
        self.video_frame = tk.Frame(self.controls_card, bg="#27272a")
        tk.Label(
            self.video_frame, text="Path File Video (.mp4 / .avi / .mov):",
            font=("Segoe UI", 9, "bold"), fg="#e4e4e7", bg="#27272a"
        ).pack(anchor="w", pady=(0, 5))
        v_row = tk.Frame(self.video_frame, bg="#27272a")
        v_row.pack(fill="x")
        tk.Entry(
            v_row, textvariable=self.video_path_var, font=("Segoe UI", 9),
            bg="#18181b", fg="#f4f4f5", insertbackground="#10b981", relief="flat"
        ).pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 8))
        tk.Button(
            v_row, text="Browse Video...", command=self._browse_video,
            font=("Segoe UI", 9, "bold"), bg="#3f3f46", fg="#f4f4f5",
            relief="flat", padx=10, pady=4, cursor="hand2"
        ).pack(side="right")

        # --- Image / Folder Controls ---
        self.image_frame = tk.Frame(self.controls_card, bg="#27272a")
        tk.Label(
            self.image_frame, text="Path Citra Tunggal atau Folder Dataset:",
            font=("Segoe UI", 9, "bold"), fg="#e4e4e7", bg="#27272a"
        ).pack(anchor="w", pady=(0, 5))
        i_row = tk.Frame(self.image_frame, bg="#27272a")
        i_row.pack(fill="x")
        tk.Entry(
            i_row, textvariable=self.image_path_var, font=("Segoe UI", 9),
            bg="#18181b", fg="#f4f4f5", insertbackground="#10b981", relief="flat"
        ).pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 6))
        tk.Button(
            i_row, text="File...", command=self._browse_image_file,
            font=("Segoe UI", 9, "bold"), bg="#3f3f46", fg="#f4f4f5",
            relief="flat", padx=8, pady=4, cursor="hand2"
        ).pack(side="left", padx=(0, 4))
        tk.Button(
            i_row, text="Folder...", command=self._browse_image_dir,
            font=("Segoe UI", 9, "bold"), bg="#3f3f46", fg="#f4f4f5",
            relief="flat", padx=8, pady=4, cursor="hand2"
        ).pack(side="right")

        self.webcam_frame.pack(fill="x")

        # Preview Box
        preview_card = tk.Frame(body, bg="#09090b", padx=4, pady=4)
        preview_card.pack(fill="both", expand=True, pady=(0, 8))

        self.preview_label = tk.Label(
            preview_card,
            text="Memuat Preview...",
            font=("Segoe UI", 10),
            fg="#71717a",
            bg="#09090b",
            width=self.PREVIEW_W,
            height=self.PREVIEW_H
        )
        self.preview_label.pack(expand=True)

        tk.Label(
            body, textvariable=self.status_var, font=("Segoe UI", 8),
            fg="#a1a1aa", bg="#18181b"
        ).pack(anchor="w", pady=(0, 8))

        # Action Buttons
        btn_row = tk.Frame(body, bg="#18181b")
        btn_row.pack(fill="x")

        tk.Button(
            btn_row, text="Batal", command=self._on_cancel,
            font=("Segoe UI", 10, "bold"), bg="#3f3f46", fg="#f4f4f5",
            activebackground="#52525b", relief="flat", padx=16, pady=9, cursor="hand2"
        ).pack(side="right", padx=(8, 0))

        tk.Button(
            btn_row, text=self.action_text, command=self._on_confirm,
            font=("Segoe UI", 10, "bold"), bg="#10b981", fg="#09090b",
            activebackground="#059669", relief="flat", padx=18, pady=9, cursor="hand2"
        ).pack(side="right", fill="x", expand=True)

    def _stop_preview(self):
        if self.preview_job is not None:
            try:
                self.root.after_cancel(self.preview_job)
            except tk.TclError:
                pass
            self.preview_job = None
        if self.preview_cap is not None:
            self.preview_cap.release()
            self.preview_cap = None

    def _display_bgr_frame(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        scale = min(self.PREVIEW_W / max(1, w), self.PREVIEW_H / max(1, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.current_photo = img_tk
        self.preview_label.configure(image=img_tk, text="")

    def _clear_preview_image(self, msg: str):
        self.current_photo = None
        self.preview_label.configure(image="", text=msg)

    def _refresh_cameras(self):
        self._stop_preview()
        self.status_var.set("Memindai kamera yang tersedia (0-5)...")
        self.root.update_idletasks()

        self.cameras = probe_cameras(max_to_test=6)
        if self.cameras:
            labels = [c["label"] for c in self.cameras]
            self.cam_combo["values"] = labels
            self.cam_combo.current(0)
            self.status_var.set(f"Ditemukan {len(self.cameras)} kamera aktif.")
            if self.mode_var.get() == "webcam":
                self._start_camera_preview()
        else:
            self.cam_combo["values"] = ["Tidak ada kamera terdeteksi (Fallback: 0)"]
            self.cam_combo.current(0)
            self._clear_preview_image("Tidak ada kamera aktif ditemukan.\nCek koneksi webcam lalu klik 'Scan Ulang'.")
            self.status_var.set("Tidak ada kamera terdeteksi.")

    def _get_selected_camera_index(self) -> int:
        if not self.cameras:
            return 0
        sel = self.cam_combo.current()
        if 0 <= sel < len(self.cameras):
            return int(self.cameras[sel]["index"])
        return int(self.cameras[0]["index"])

    def _start_camera_preview(self):
        self._stop_preview()
        if not self.cameras:
            return
        cam_idx = self._get_selected_camera_index()
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._clear_preview_image(f"Gagal membuka preview Kamera {cam_idx}")
            self.status_var.set(f"Error membuka Kamera {cam_idx}.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.preview_cap = cap
        self.status_var.set(f"Preview Aktif: {self.cam_var.get()}")
        self._update_camera_preview()

    def _update_camera_preview(self):
        if self.preview_cap is None or self.mode_var.get() != "webcam":
            return
        ret, frame = self.preview_cap.read()
        if ret and frame is not None:
            if self.mirror_var.get():
                frame = cv2.flip(frame, 1)
            self._display_bgr_frame(frame)
        self.preview_job = self.root.after(33, self._update_camera_preview)

    def _on_mode_change(self):
        self._stop_preview()
        for f in (self.webcam_frame, self.video_frame, self.image_frame):
            f.pack_forget()

        mode = self.mode_var.get()
        if mode == "webcam":
            self.webcam_frame.pack(fill="x")
            self._start_camera_preview()
        elif mode == "video":
            self.video_frame.pack(fill="x")
            self._preview_file_thumbnail(self.video_path_var.get(), is_video=True)
        elif mode == "image":
            self.image_frame.pack(fill="x")
            self._preview_file_thumbnail(self.image_path_var.get(), is_video=False)

    def _preview_file_thumbnail(self, path_str: str, is_video: bool):
        p = Path(path_str) if path_str else None
        if p is None or not p.exists():
            self._clear_preview_image("Pilih file atau folder untuk melihat pratinjau.")
            return

        if is_video:
            cap = cv2.VideoCapture(str(p))
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                self._display_bgr_frame(frame)
                self.status_var.set(f"Video siap: {p.name}")
            else:
                self._clear_preview_image("Tidak dapat membaca frame video.")
        else:
            img_path = p
            if p.is_dir():
                imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
                if not imgs:
                    self._clear_preview_image("Folder tidak berisi file citra.")
                    return
                img_path = imgs[0]
                self.status_var.set(f"Folder siap ({len(imgs)} citra): {p.name}")
            else:
                self.status_var.set(f"Citra siap: {p.name}")

            img = cv2.imread(str(img_path))
            if img is not None:
                self._display_bgr_frame(img)

    def _browse_video(self):
        init_dir = r"E:\Semester 5\C&V" if os.path.exists(r"E:\Semester 5\C&V") else os.getcwd()
        chosen = filedialog.askopenfilename(
            title="Pilih File Video BISINDO",
            initialdir=init_dir,
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if chosen:
            self.video_path_var.set(chosen)
            self._preview_file_thumbnail(chosen, is_video=True)

    def _browse_image_file(self):
        init_dir = r"E:\Semester 5\C&V\Dataset\data 1\Citra BISINDO"
        if not os.path.exists(init_dir):
            init_dir = os.getcwd()
        chosen = filedialog.askopenfilename(
            title="Pilih File Citra BISINDO",
            initialdir=init_dir,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if chosen:
            self.image_path_var.set(chosen)
            self._preview_file_thumbnail(chosen, is_video=False)

    def _browse_image_dir(self):
        init_dir = r"E:\Semester 5\C&V\Dataset\data 1\Citra BISINDO"
        if not os.path.exists(init_dir):
            init_dir = os.getcwd()
        chosen = filedialog.askdirectory(title="Pilih Folder Citra BISINDO", initialdir=init_dir)
        if chosen:
            self.image_path_var.set(chosen)
            self._preview_file_thumbnail(chosen, is_video=False)

    def _on_confirm(self):
        mode = self.mode_var.get()
        if mode == "webcam":
            cam_idx = self._get_selected_camera_index()
            self._stop_preview()
            self.result = {"source": cam_idx, "mirror": bool(self.mirror_var.get())}
            self.root.destroy()
        elif mode == "video":
            path = self.video_path_var.get().strip().strip('"')
            if not path or not os.path.exists(path):
                messagebox.showwarning("File Tidak Ditemukan", "Silakan pilih file video yang valid terlebih dahulu.")
                return
            self._stop_preview()
            self.result = {"source": path, "mirror": False}
            self.root.destroy()
        elif mode == "image":
            path = self.image_path_var.get().strip().strip('"')
            if not path or not os.path.exists(path):
                messagebox.showwarning("Path Tidak Ditemukan", "Silakan pilih file citra atau folder citra yang valid.")
                return
            self._stop_preview()
            self.result = {"source": path, "mirror": False}
            self.root.destroy()

    def _on_cancel(self):
        self._stop_preview()
        self.result = None
        self.root.destroy()


def launch_input_selector_gui(
    allow_files: bool = True,
    title: str = "BISINDO Translator - Pilih Input Kamera",
    action_text: str = "Mulai Penerjemah BISINDO"
) -> Optional[Dict[str, any]]:
    """
    Opens the Tkinter input selector GUI and returns:
        {'source': Union[int, str], 'mirror': bool}
    or None if the user cancelled / closed the window.
    """
    root = tk.Tk()
    app = InputSelectorApp(root, allow_files=allow_files, title=title, action_text=action_text)
    root.mainloop()
    return app.result
