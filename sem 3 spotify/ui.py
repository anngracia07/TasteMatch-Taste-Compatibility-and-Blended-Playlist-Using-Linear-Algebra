import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_loader import FEATURES, DISPLAY_FEATURES, load_dataset
from taste_analysis import analyze_taste
from recommender import generate_recommendations


# ============================================================
# TASTEMATCH UI
# Subspace-Based Geometric Framework for Music Taste
# ============================================================


class TasteMatchApp:
    def __init__(self, root):
        self.root = root

        self.root.title("TasteMatch — Subspace-Based Music Compatibility")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 750)
        self.root.configure(bg="#0f1117")

        self.colors = {
            "bg": "#0f1117",
            "panel": "#171a23",
            "panel2": "#1e222d",
            "text": "#f3f4f6",
            "muted": "#a7adbb",
            "accent": "#8b5cf6",
            "accent2": "#22c55e",
            "blue": "#38bdf8",
            "orange": "#fb923c",
            "red": "#ef4444",
            "border": "#303542",
        }

        self.df = None
        self.scaler = None

        self.selected_you = []
        self.selected_them = []

        self.song_tree = None
        self.your_tree = None
        self.their_tree = None

        self.search_var = tk.StringVar()

        self.results_canvas = None
        self.results_scroll_frame = None

        try:
            self.df = load_dataset()
        except Exception as exc:
            messagebox.showerror("Dataset error", str(exc))
            self.root.destroy()
            return
        self.configure_styles()
        self.show_selection_screen()

    def configure_styles(self):
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Treeview",
            background=self.colors["panel2"],
            foreground=self.colors["text"],
            fieldbackground=self.colors["panel2"],
            rowheight=30,
            borderwidth=0,
            font=("Segoe UI", 10),
        )

        style.configure(
            "Treeview.Heading",
            background="#272c38",
            foreground=self.colors["text"],
            relief="flat",
            font=("Segoe UI Semibold", 10),
        )

        style.map(
            "Treeview",
            background=[
                ("selected", "#5b3ba5")
            ],
            foreground=[
                ("selected", "white")
            ]
        )

        style.configure(
            "TNotebook",
            background=self.colors["bg"],
            borderwidth=0,
        )

        style.configure(
            "TNotebook.Tab",
            background=self.colors["panel2"],
            foreground=self.colors["muted"],
            padding=(18, 10),
            font=("Segoe UI Semibold", 10),
        )

        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", self.colors["accent"])
            ],
            foreground=[
                ("selected", "white")
            ]
        )

        style.configure(
            "Vertical.TScrollbar",
            background="#272c38",
            troughcolor=self.colors["bg"],
            bordercolor=self.colors["bg"],
            arrowcolor=self.colors["muted"],
        )

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_button(
        self,
        parent,
        text,
        command,
        bg=None,
        fg="white",
        font=("Segoe UI Semibold", 10),
        padx=16,
        pady=9,
    ):
        if bg is None:
            bg = self.colors["accent"]

        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            font=font,
            padx=padx,
            pady=pady,
        )

        return button

    def create_card(self, parent, **kwargs):
        return tk.Frame(
            parent,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            **kwargs
        )

    def show_selection_screen(self):
        self.clear_root()

        header = tk.Frame(
            self.root,
            bg=self.colors["bg"],
            height=90
        )
        header.pack(fill="x", padx=30, pady=(20, 5))
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="♫  TasteMatch",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=("Segoe UI", 27, "bold"),
        )
        title.pack(side="left", anchor="w")

        subtitle = tk.Label(
            header,
            text="Subspace-Based Geometric Framework for Music Taste Compatibility",
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            font=("Segoe UI", 11),
        )
        subtitle.pack(side="left", padx=20, pady=(15, 0))

        info = tk.Label(
            self.root,
            text=(
                "Select at least 3 songs for each person. "
                "The program uses standardized 7D music vectors, PCA, "
                "principal angles and projection-based recommendations."
            ),
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            font=("Segoe UI", 10),
            anchor="w",
        )
        info.pack(fill="x", padx=32, pady=(0, 15))

        main = tk.Frame(
            self.root,
            bg=self.colors["bg"]
        )
        main.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=5
        )

        # ----------------------------------------------------
        # Library
        # ----------------------------------------------------

        library_card = self.create_card(main)
        library_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 8)
        )

        tk.Label(
            library_card,
            text="SONG LIBRARY",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12),
        ).pack(
            anchor="w",
            padx=16,
            pady=(14, 8)
        )

        search_frame = tk.Frame(
            library_card,
            bg=self.colors["panel"]
        )
        search_frame.pack(
            fill="x",
            padx=16,
            pady=(0, 10)
        )

        search_entry = tk.Entry(
            search_frame,
            textvariable=self.search_var,
            bg=self.colors["panel2"],
            fg=self.colors["text"],
            insertbackground="white",
            relief="flat",
            font=("Segoe UI", 10),
        )
        search_entry.pack(
            side="left",
            fill="x",
            expand=True,
            ipady=8,
            padx=(0, 7)
        )

        search_entry.bind(
            "<KeyRelease>",
            lambda event: self.update_song_library()
        )

        clear_button = self.create_button(
            search_frame,
            "Clear",
            self.clear_search,
            bg="#343947",
            padx=10,
            pady=6,
        )
        clear_button.pack(side="right")

        library_table_frame = tk.Frame(
            library_card,
            bg=self.colors["panel"]
        )
        library_table_frame.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=(0, 10)
        )

        columns = (
            "track",
            "artist",
            "genre",
        )

        self.song_tree = ttk.Treeview(
            library_table_frame,
            columns=columns,
            show="headings",
            selectmode="extended"
        )

        self.song_tree.heading(
            "track",
            text="Track"
        )
        self.song_tree.heading(
            "artist",
            text="Artist"
        )
        self.song_tree.heading(
            "genre",
            text="Genre"
        )

        self.song_tree.column(
            "track",
            width=260,
            anchor="w"
        )
        self.song_tree.column(
            "artist",
            width=180,
            anchor="w"
        )
        self.song_tree.column(
            "genre",
            width=120,
            anchor="w"
        )

        scroll = ttk.Scrollbar(
            library_table_frame,
            orient="vertical",
            command=self.song_tree.yview,
            style="Vertical.TScrollbar"
        )
        self.song_tree.configure(
            yscrollcommand=scroll.set
        )

        self.song_tree.pack(
            side="left",
            fill="both",
            expand=True
        )
        scroll.pack(
            side="right",
            fill="y"
        )

        library_buttons = tk.Frame(
            library_card,
            bg=self.colors["panel"]
        )
        library_buttons.pack(
            fill="x",
            padx=16,
            pady=(0, 16)
        )

        self.create_button(
            library_buttons,
            "＋ ADD TO YOU",
            lambda: self.add_selected("you"),
            bg="#4c2c87",
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 5)
        )

        self.create_button(
            library_buttons,
            "＋ ADD TO THEM",
            lambda: self.add_selected("them"),
            bg="#185f7a",
        ).pack(
            side="right",
            fill="x",
            expand=True,
            padx=(5, 0)
        )

        # ----------------------------------------------------
        # You
        # ----------------------------------------------------

        your_card = self.create_card(main)
        your_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=8
        )

        tk.Label(
            your_card,
            text="YOUR MUSIC",
            bg=self.colors["panel"],
            fg=self.colors["accent"],
            font=("Segoe UI Semibold", 12),
        ).pack(
            anchor="w",
            padx=16,
            pady=(14, 3)
        )

        self.your_count_label = tk.Label(
            your_card,
            text="0 songs selected",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9),
        )
        self.your_count_label.pack(
            anchor="w",
            padx=16,
            pady=(0, 10)
        )

        your_table_frame = tk.Frame(
            your_card,
            bg=self.colors["panel"]
        )
        your_table_frame.pack(
            fill="both",
            expand=True,
            padx=12
        )

        self.your_tree = ttk.Treeview(
            your_table_frame,
            columns=("track", "artist"),
            show="headings",
            selectmode="extended"
        )

        self.your_tree.heading(
            "track",
            text="Track"
        )
        self.your_tree.heading(
            "artist",
            text="Artist"
        )

        self.your_tree.column(
            "track",
            width=220,
            anchor="w"
        )
        self.your_tree.column(
            "artist",
            width=150,
            anchor="w"
        )

        your_scroll = ttk.Scrollbar(
            your_table_frame,
            orient="vertical",
            command=self.your_tree.yview,
            style="Vertical.TScrollbar"
        )
        self.your_tree.configure(
            yscrollcommand=your_scroll.set
        )

        self.your_tree.pack(
            side="left",
            fill="both",
            expand=True
        )
        your_scroll.pack(
            side="right",
            fill="y"
        )

        self.create_button(
            your_card,
            "− REMOVE SELECTED",
            lambda: self.remove_selected("you"),
            bg="#343947",
        ).pack(
            fill="x",
            padx=16,
            pady=16
        )

        # ----------------------------------------------------
        # Them
        # ----------------------------------------------------

        them_card = self.create_card(main)
        them_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(8, 0)
        )

        tk.Label(
            them_card,
            text="THEIR MUSIC",
            bg=self.colors["panel"],
            fg="#38bdf8",
            font=("Segoe UI Semibold", 12),
        ).pack(
            anchor="w",
            padx=16,
            pady=(14, 3)
        )

        self.their_count_label = tk.Label(
            them_card,
            text="0 songs selected",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9),
        )
        self.their_count_label.pack(
            anchor="w",
            padx=16,
            pady=(0, 10)
        )

        their_table_frame = tk.Frame(
            them_card,
            bg=self.colors["panel"]
        )
        their_table_frame.pack(
            fill="both",
            expand=True,
            padx=12
        )

        self.their_tree = ttk.Treeview(
            their_table_frame,
            columns=("track", "artist"),
            show="headings",
            selectmode="extended"
        )

        self.their_tree.heading(
            "track",
            text="Track"
        )
        self.their_tree.heading(
            "artist",
            text="Artist"
        )

        self.their_tree.column(
            "track",
            width=220,
            anchor="w"
        )
        self.their_tree.column(
            "artist",
            width=150,
            anchor="w"
        )

        their_scroll = ttk.Scrollbar(
            their_table_frame,
            orient="vertical",
            command=self.their_tree.yview,
            style="Vertical.TScrollbar"
        )
        self.their_tree.configure(
            yscrollcommand=their_scroll.set
        )

        self.their_tree.pack(
            side="left",
            fill="both",
            expand=True
        )
        their_scroll.pack(
            side="right",
            fill="y"
        )

        self.create_button(
            them_card,
            "− REMOVE SELECTED",
            lambda: self.remove_selected("them"),
            bg="#343947",
        ).pack(
            fill="x",
            padx=16,
            pady=16
        )

        self.update_song_library()
        self.update_selection_lists()

        # ----------------------------------------------------
        # Compare footer
        # ----------------------------------------------------

        footer = tk.Frame(
            self.root,
            bg=self.colors["bg"]
        )
        footer.pack(
            fill="x",
            padx=30,
            pady=(8, 22)
        )

        self.compare_status = tk.Label(
            footer,
            text="Select at least 3 songs for each person.",
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            font=("Segoe UI", 10)
        )
        self.compare_status.pack(
            side="left"
        )

        compare_button = self.create_button(
            footer,
            "COMPARE TASTES  →",
            self.run_analysis,
            bg=self.colors["accent"],
            font=("Segoe UI Semibold", 13),
            padx=30,
            pady=12,
        )
        compare_button.pack(
            side="right"
        )

    def clear_search(self):
        self.search_var.set("")
        self.update_song_library()

    def update_song_library(self):
        if self.song_tree is None:
            return

        for item in self.song_tree.get_children():
            self.song_tree.delete(item)

        query = self.search_var.get().strip().lower()

        if query:
            mask = (
                self.df["track_name"].astype(str).str.lower().str.contains(
                    query,
                    na=False
                )
                |
                self.df["artists"].astype(str).str.lower().str.contains(
                    query,
                    na=False
                )
                |
                self.df["track_genre"].astype(str).str.lower().str.contains(
                    query,
                    na=False
                )
            )
            display_df = self.df[mask]
        else:
            display_df = self.df

        # Show at most all available rows; Treeview remains scrollable.
        for idx, row in display_df.iterrows():
            self.song_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    str(row["track_name"]),
                    str(row["artists"]),
                    str(row["track_genre"]),
                )
            )

    def add_selected(self, person):
        if self.song_tree is None:
            return

        selected_items = self.song_tree.selection()

        if not selected_items:
            messagebox.showinfo(
                "Select songs",
                "Select one or more songs from the Song Library first."
            )
            return

        target = (
            self.selected_you
            if person == "you"
            else self.selected_them
        )

        other = (
            self.selected_them
            if person == "you"
            else self.selected_you
        )

        for item in selected_items:
            idx = int(item)

            if idx in target:
                continue

            if idx in other:
                messagebox.showwarning(
                    "Song already used",
                    "A song cannot be selected for both people."
                )
                continue

            target.append(idx)

        self.update_selection_lists()
        self.update_compare_status()

    def remove_selected(self, person):
        tree = (
            self.your_tree
            if person == "you"
            else self.their_tree
        )

        selected_items = tree.selection()

        if not selected_items:
            return

        target = (
            self.selected_you
            if person == "you"
            else self.selected_them
        )

        indices_to_remove = {
            int(item)
            for item in selected_items
        }

        target[:] = [
            idx for idx in target
            if idx not in indices_to_remove
        ]

        self.update_selection_lists()
        self.update_compare_status()

    def update_selection_lists(self):
        if self.your_tree is None or self.their_tree is None:
            return

        for tree in [self.your_tree, self.their_tree]:
            for item in tree.get_children():
                tree.delete(item)

        for idx in self.selected_you:
            row = self.df.loc[idx]
            self.your_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    str(row["track_name"]),
                    str(row["artists"]),
                )
            )

        for idx in self.selected_them:
            row = self.df.loc[idx]
            self.their_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    str(row["track_name"]),
                    str(row["artists"]),
                )
            )

        self.your_count_label.config(
            text=f"{len(self.selected_you)} songs selected"
        )

        self.their_count_label.config(
            text=f"{len(self.selected_them)} songs selected"
        )

    def update_compare_status(self):
        a = len(self.selected_you)
        b = len(self.selected_them)

        if a >= 3 and b >= 3:
            self.compare_status.config(
                text="Ready — both taste collections have at least 3 songs.",
                fg=self.colors["accent2"]
            )
        else:
            self.compare_status.config(
                text=f"Need at least 3 songs each. You: {a} | Them: {b}",
                fg=self.colors["muted"]
            )

    def show_results_screen(self, results):
        self.clear_root()

        header = tk.Frame(
            self.root,
            bg=self.colors["bg"],
            height=72
        )
        header.pack(
            fill="x",
            padx=25,
            pady=(15, 5)
        )
        header.pack_propagate(False)

        self.create_button(
            header,
            "← BACK TO SONGS",
            self.show_selection_screen,
            bg="#343947",
            padx=14,
            pady=7
        ).pack(
            side="left",
            pady=5
        )

        tk.Label(
            header,
            text="TasteMatch Results",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=("Segoe UI", 23, "bold")
        ).pack(
            side="left",
            padx=18
        )

        tk.Label(
            header,
            text="PCA • Principal Angles • Projection",
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            font=("Segoe UI", 10)
        ).pack(
            side="left",
            pady=(8, 0)
        )

        # Main scrollable dashboard.
        outer = tk.Frame(
            self.root,
            bg=self.colors["bg"]
        )
        outer.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=(5, 20)
        )

        self.results_canvas = tk.Canvas(
            outer,
            bg=self.colors["bg"],
            highlightthickness=0
        )

        scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=self.results_canvas.yview,
            style="Vertical.TScrollbar"
        )

        self.results_scroll_frame = tk.Frame(
            self.results_canvas,
            bg=self.colors["bg"]
        )

        self.results_window = self.results_canvas.create_window(
            (0, 0),
            window=self.results_scroll_frame,
            anchor="nw"
        )

        self.results_canvas.configure(
            yscrollcommand=scrollbar.set
        )

        self.results_scroll_frame.bind(
            "<Configure>",
            lambda event: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )

        self.results_canvas.bind(
            "<Configure>",
            lambda event: self.results_canvas.itemconfig(
                self.results_window,
                width=event.width
            )
        )

        self.results_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )
        scrollbar.pack(
            side="right",
            fill="y"
        )

        # Mouse wheel.
        self.results_canvas.bind_all(
            "<MouseWheel>",
            self.on_mousewheel
        )

        self.build_results_dashboard(
            self.results_scroll_frame,
            results
        )

    def on_mousewheel(self, event):
        if self.results_canvas is not None:
            self.results_canvas.yview_scroll(
                int(-1 * (event.delta / 120)),
                "units"
            )

    def build_results_dashboard(self, parent, r):
        # ----------------------------------------------------
        # Hero
        # ----------------------------------------------------

        hero = self.create_card(parent)
        hero.pack(
            fill="x",
            padx=8,
            pady=(5, 12)
        )

        hero_left = tk.Frame(
            hero,
            bg=self.colors["panel"]
        )
        hero_left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=25,
            pady=20
        )

        tk.Label(
            hero_left,
            text="SUBSPACE COMPATIBILITY",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI Semibold", 10)
        ).pack(
            anchor="w"
        )

        tk.Label(
            hero_left,
            text=f"{r['subspace_compatibility']:.2f}%",
            bg=self.colors["panel"],
            fg=self.colors["accent"],
            font=("Segoe UI", 38, "bold")
        ).pack(
            anchor="w"
        )

        tk.Label(
            hero_left,
            text=r["verdict"],
            bg=self.colors["panel"],
            fg=self.colors["accent2"],
            font=("Segoe UI Semibold", 13)
        ).pack(
            anchor="w"
        )

        tk.Label(
            hero_left,
            text=(
                "Mean cosine of the principal-angle cosines. "
                "This measures geometric alignment between the "
                "two 3D taste subspaces."
            ),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 10),
            wraplength=620,
            justify="left"
        ).pack(
            anchor="w",
            pady=(8, 0)
        )

        # Summary cards.
        stats = tk.Frame(
            hero,
            bg=self.colors["panel"]
        )
        stats.pack(
            side="right",
            padx=25,
            pady=22
        )

        self.add_stat(
            stats,
            "Your songs",
            str(len(self.selected_you))
        )

        self.add_stat(
            stats,
            "Their songs",
            str(len(self.selected_them))
        )

        self.add_stat(
            stats,
            "Your rank",
            str(r["rank_A"])
        )

        self.add_stat(
            stats,
            "Their rank",
            str(r["rank_B"])
        )

        # ----------------------------------------------------
        # Comparison card
        # ----------------------------------------------------

        comparison = self.create_card(parent)
        comparison.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            comparison,
            text="COMPATIBILITY COMPARISON",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 3)
        )

        tk.Label(
            comparison,
            text=(
                "The cosine score is a baseline using one average taste vector. "
                "The principal-angle score compares entire 3D taste subspaces."
            ),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 8)
        )

        fig = Figure(
            figsize=(9, 3.2),
            dpi=100,
            facecolor=self.colors["panel"]
        )

        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors["panel"])

        methods = [
            "Average-vector cosine",
            "Principal-angle\nsubspace"
        ]

        values = [
            r["cosine_percent"],
            r["subspace_compatibility"]
        ]

        bars = ax.bar(
            methods,
            values
        )

        ax.set_ylim(0, 100)
        ax.set_ylabel("Compatibility (%)")
        ax.set_title(
            "Baseline vs proposed geometric method",
            color=self.colors["text"]
        )

        ax.tick_params(
            colors=self.colors["muted"]
        )

        ax.yaxis.label.set_color(
            self.colors["muted"]
        )

        for spine in ax.spines.values():
            spine.set_color(self.colors["border"])

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 2, 97),
                f"{value:.1f}%",
                ha="center",
                color=self.colors["text"],
                fontweight="bold"
            )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=comparison
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # PCA explained variance
        # ----------------------------------------------------

        variance_card = self.create_card(parent)
        variance_card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            variance_card,
            text="PCA EXPLAINED VARIANCE",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 3)
        )

        tk.Label(
            variance_card,
            text=(
                "PCA finds the dominant directions of variation in each "
                "person's music feature vectors. The first three components "
                "form the 3D taste subspace."
            ),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 8)
        )

        fig = Figure(
            figsize=(9, 3.5),
            dpi=100,
            facecolor=self.colors["panel"]
        )

        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors["panel"])

        x = np.arange(3)
        width = 0.36

        vals_A = (
            r["pca_A"].explained_variance_ratio_
            * 100
        )

        vals_B = (
            r["pca_B"].explained_variance_ratio_
            * 100
        )

        ax.bar(
            x - width / 2,
            vals_A,
            width,
            label="You"
        )

        ax.bar(
            x + width / 2,
            vals_B,
            width,
            label="Them"
        )

        ax.set_xticks(
            x,
            ["PC1", "PC2", "PC3"]
        )

        ax.set_ylabel("Explained variance (%)")
        ax.set_title(
            "Variance captured by the three taste directions"
        )

        ax.legend()

        ax.tick_params(
            colors=self.colors["muted"]
        )

        for spine in ax.spines.values():
            spine.set_color(self.colors["border"])

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=variance_card
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        totals = (
            np.sum(vals_A),
            np.sum(vals_B)
        )

        tk.Label(
            variance_card,
            text=(
                f"Your first 3 PCs explain {totals[0]:.2f}% of variance.   "
                f"Their first 3 PCs explain {totals[1]:.2f}% of variance."
            ),
            bg=self.colors["panel"],
            fg=self.colors["accent2"],
            font=("Segoe UI Semibold", 10)
        ).pack(
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # Principal angles
        # ----------------------------------------------------

        angle_card = self.create_card(parent)
        angle_card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            angle_card,
            text="PRINCIPAL ANGLES",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 3)
        )

        tk.Label(
            angle_card,
            text=(
                "Smaller angles mean the corresponding dominant taste "
                "directions are more closely aligned."
            ),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 8)
        )

        fig = Figure(
            figsize=(9, 3.2),
            dpi=100,
            facecolor=self.colors["panel"]
        )

        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors["panel"])

        labels = [
            "Angle 1",
            "Angle 2",
            "Angle 3"
        ]

        bars = ax.bar(
            labels,
            r["angles"]
        )

        ax.set_ylabel("Angle (degrees)")
        ax.set_title(
            "Principal angles between the two 3D taste subspaces"
        )

        ax.tick_params(
            colors=self.colors["muted"]
        )

        for spine in ax.spines.values():
            spine.set_color(self.colors["border"])

        for bar, value in zip(
            bars,
            r["angles"]
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.2f}°",
                ha="center",
                color=self.colors["text"],
                fontweight="bold"
            )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=angle_card
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # Taste space 2D
        # ----------------------------------------------------

        taste_card = self.create_card(parent)
        taste_card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            taste_card,
            text="2D PCA TASTE SPACE",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 3)
        )

        tk.Label(
            taste_card,
            text=(
                "Each point is a selected song. The x and y axes are "
                "the first two PCA directions of the common feature space."
            ),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 8)
        )

        fig = Figure(
            figsize=(9, 5),
            dpi=100,
            facecolor=self.colors["panel"]
        )

        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors["panel"])

        ax.scatter(
            r["A_pca"][:, 0],
            r["A_pca"][:, 1],
            s=70,
            label="You"
        )

        ax.scatter(
            r["B_pca"][:, 0],
            r["B_pca"][:, 1],
            s=70,
            marker="^",
            label="Them"
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            "Selected songs in the common PCA space"
        )

        ax.legend()

        ax.tick_params(
            colors=self.colors["muted"]
        )

        for spine in ax.spines.values():
            spine.set_color(self.colors["border"])

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=taste_card
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # Recommended songs in PCA space
        # ----------------------------------------------------

        rec_space_card = self.create_card(parent)
        rec_space_card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            rec_space_card,
            text="RECOMMENDED SONGS IN PCA SPACE",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 3)
        )

        tk.Label(
            rec_space_card,
            text=(
                "Recommended songs are candidates outside both selected "
                "collections and are ranked by their projection fit to "
                "the shared taste space."
            ),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 8)
        )

        fig = Figure(
            figsize=(9, 5),
            dpi=100,
            facecolor=self.colors["panel"]
        )

        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors["panel"])

        ax.scatter(
            r["A_pca"][:, 0],
            r["A_pca"][:, 1],
            s=60,
            label="You"
        )

        ax.scatter(
            r["B_pca"][:, 0],
            r["B_pca"][:, 1],
            s=60,
            marker="^",
            label="Them"
        )

        if len(r["recommended_pca"]) > 0:
            ax.scatter(
                r["recommended_pca"][:, 0],
                r["recommended_pca"][:, 1],
                s=100,
                marker="*",
                label="Recommended"
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            "Selected songs and projection-based recommendations"
        )

        ax.legend()

        ax.tick_params(
            colors=self.colors["muted"]
        )

        for spine in ax.spines.values():
            spine.set_color(self.colors["border"])

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=rec_space_card
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # Feature comparison
        # ----------------------------------------------------

        feature_card = self.create_card(parent)
        feature_card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            feature_card,
            text="MUSICAL FEATURE COMPARISON",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 3)
        )

        tk.Label(
            feature_card,
            text=(
                "Mean values of the original seven numerical music "
                "features for the two selected collections."
            ),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 8)
        )

        fig = Figure(
            figsize=(10, 4.2),
            dpi=100,
            facecolor=self.colors["panel"]
        )

        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors["panel"])

        x = np.arange(len(FEATURES))
        width = 0.35

        # Normalize only for display so tempo does not dominate
        # the visual scale. The actual calculations use the
        # standardized vectors above.
        feature_scale = np.maximum(
            np.maximum(
                np.abs(r["feature_mean_A"]),
                np.abs(r["feature_mean_B"])
            ),
            1e-12
        )

        # Keep bounded-looking display values.
        display_A = (
            r["feature_mean_A"]
            / np.maximum(
                self.df[FEATURES].max().to_numpy(),
                1e-12
            )
        )

        display_B = (
            r["feature_mean_B"]
            / np.maximum(
                self.df[FEATURES].max().to_numpy(),
                1e-12
            )
        )

        ax.bar(
            x - width / 2,
            display_A,
            width,
            label="You"
        )

        ax.bar(
            x + width / 2,
            display_B,
            width,
            label="Them"
        )

        ax.set_xticks(
            x,
            DISPLAY_FEATURES,
            rotation=25,
            ha="right"
        )

        ax.set_ylabel("Relative feature level")
        ax.set_title(
            "Average musical feature profile"
        )

        ax.legend()

        ax.tick_params(
            colors=self.colors["muted"]
        )

        for spine in ax.spines.values():
            spine.set_color(self.colors["border"])

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=feature_card
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # Mathematical summary
        # ----------------------------------------------------

        math_card = self.create_card(parent)
        math_card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            math_card,
            text="MATHEMATICAL SUMMARY",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 10)
        )

        math_text = (
            "1. Song vector:\n"
            "   x = [danceability, energy, valence, acousticness, "
            "speechiness, instrumentalness, tempo] ∈ R⁷\n\n"

            "2. Standardization:\n"
            "   z = (x − μ) / σ\n\n"

            "3. PCA taste basis:\n"
            "   Q_A, Q_B ∈ R⁷ˣ³\n"
            "   Columns of Q are the three dominant orthonormal taste directions.\n\n"

            "4. Principal-angle matrix:\n"
            "   M = Q_Aᵀ Q_B\n\n"

            "5. SVD:\n"
            "   M = UΣVᵀ\n"
            "   σᵢ = cos(θᵢ)\n"
            "   θᵢ = arccos(σᵢ)\n\n"

            "6. Project compatibility:\n"
            "   Compatibility = mean(cos(θᵢ)) × 100\n\n"

            "7. Baseline:\n"
            "   cosine(mean_A, mean_B)\n\n"

            "8. Recommendation projection:\n"
            "   x̂ = QQᵀx\n"
            "   projection error = ||x − x̂||\n\n"

            "The recommendation system therefore ranks songs by how well "
            "their feature vectors fit the shared low-dimensional taste space "
            "rather than simply choosing songs closest to one average vector."
        )

        tk.Label(
            math_card,
            text=math_text,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Consolas", 10),
            justify="left",
            anchor="w",
            wraplength=1100
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 18)
        )

        # ----------------------------------------------------
        # RREF / rank / Gram-Schmidt
        # ----------------------------------------------------

        linear_card = self.create_card(parent)
        linear_card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            linear_card,
            text="LINEAR-ALGEBRA SUPPORTING ANALYSIS",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 10)
        )

        summary = (
            f"Your matrix shape: {r['A_matrix'].shape}   |   "
            f"rank = {r['rank_A']}   |   "
            f"Gram-Schmidt basis size = {r['GS_A'].shape[1]}\n"
            f"Their matrix shape: {r['B_matrix'].shape}   |   "
            f"rank = {r['rank_B']}   |   "
            f"Gram-Schmidt basis size = {r['GS_B'].shape[1]}\n\n"
            "RREF is included as a supporting rank/independence check. "
            "PCA and SVD are used for the main numerical geometry because "
            "they are more suitable for floating-point data."
        )

        tk.Label(
            linear_card,
            text=summary,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 10),
            justify="left"
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 15)
        )

        rref_frame = tk.Frame(
            linear_card,
            bg=self.colors["panel"]
        )
        rref_frame.pack(
            fill="x",
            padx=20,
            pady=(0, 20)
        )

        left_text = (
            "YOUR RREF\n\n"
            + self.format_matrix(r["rref_A"])
        )

        right_text = (
            "THEIR RREF\n\n"
            + self.format_matrix(r["rref_B"])
        )

        tk.Label(
            rref_frame,
            text=left_text,
            bg=self.colors["panel2"],
            fg=self.colors["muted"],
            font=("Consolas", 8),
            justify="left",
            anchor="nw",
            padx=12,
            pady=12
        ).pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 5)
        )

        tk.Label(
            rref_frame,
            text=right_text,
            bg=self.colors["panel2"],
            fg=self.colors["muted"],
            font=("Consolas", 8),
            justify="left",
            anchor="nw",
            padx=12,
            pady=12
        ).pack(
            side="right",
            fill="both",
            expand=True,
            padx=(5, 0)
        )

        # ----------------------------------------------------
        # Principal angle numerical table
        # ----------------------------------------------------

        numerical_card = self.create_card(parent)
        numerical_card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        tk.Label(
            numerical_card,
            text="PRINCIPAL-ANGLE NUMERICAL RESULTS",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 12)
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 10)
        )

        table_frame = tk.Frame(
            numerical_card,
            bg=self.colors["panel"]
        )
        table_frame.pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        columns = (
            "direction",
            "cosine",
            "angle"
        )

        tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=3
        )

        tree.heading(
            "direction",
            text="Principal direction"
        )
        tree.heading(
            "cosine",
            text="σᵢ = cos(θᵢ)"
        )
        tree.heading(
            "angle",
            text="θᵢ (degrees)"
        )

        tree.column(
            "direction",
            width=200
        )
        tree.column(
            "cosine",
            width=200
        )
        tree.column(
            "angle",
            width=200
        )

        for i, (s, angle) in enumerate(
            zip(
                r["singular_values"],
                r["angles"]
            ),
            start=1
        ):
            tree.insert(
                "",
                "end",
                values=(
                    f"Direction {i}",
                    f"{s:.6f}",
                    f"{angle:.4f}°"
                )
            )

        tree.pack(
            fill="x"
        )

        # ----------------------------------------------------
        # Suggested playlist
        # ----------------------------------------------------

        playlist_card = self.create_card(parent)
        playlist_card.pack(
            fill="x",
            padx=8,
            pady=(8, 25)
        )

        tk.Label(
            playlist_card,
            text="♫  SUGGESTED 10-SONG PLAYLIST",
            bg=self.colors["panel"],
            fg=self.colors["accent"],
            font=("Segoe UI Semibold", 15)
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 3)
        )

        tk.Label(
            playlist_card,
            text=(
                "These songs were selected from the unselected candidates. "
                "The ranking is based on projection fit to the shared taste "
                "space plus fit to both individual taste spaces."
            ),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 12)
        )

        rec_df = r["recommendation_df"]

        playlist_frame = tk.Frame(
            playlist_card,
            bg=self.colors["panel"]
        )
        playlist_frame.pack(
            fill="x",
            padx=15,
            pady=(0, 20)
        )

        columns = (
            "rank",
            "track",
            "artist",
            "genre",
            "score",
            "error",
            "you",
            "them",
        )

        playlist_tree = ttk.Treeview(
            playlist_frame,
            columns=columns,
            show="headings",
            height=10
        )

        headings = {
            "rank": "#",
            "track": "Track",
            "artist": "Artist",
            "genre": "Genre",
            "score": "Score",
            "error": "Projection error",
            "you": "Fit to you",
            "them": "Fit to them",
        }

        widths = {
            "rank": 45,
            "track": 250,
            "artist": 190,
            "genre": 120,
            "score": 90,
            "error": 110,
            "you": 90,
            "them": 90,
        }

        for col in columns:
            playlist_tree.heading(
                col,
                text=headings[col]
            )
            playlist_tree.column(
                col,
                width=widths[col],
                anchor="center" if col not in ["track", "artist"] else "w"
            )

        for i, row in rec_df.iterrows():
            playlist_tree.insert(
                "",
                "end",
                values=(
                    i + 1,
                    str(row["track_name"]),
                    str(row["artists"]),
                    str(row["track_genre"]),
                    f"{row['recommendation_score']:.2f}",
                    f"{row['projection_error']:.3f}",
                    f"{row['fit_to_you']:.1f}%",
                    f"{row['fit_to_them']:.1f}%",
                )
            )

        playlist_tree.pack(
            fill="x"
        )

        tk.Label(
            playlist_card,
            text=(
                "Recommendation formula: "
                "70% shared-subspace closeness + "
                "30% average closeness to the two personal subspaces."
            ),
            bg=self.colors["panel"],
            fg=self.colors["accent2"],
            font=("Segoe UI Semibold", 9)
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 15)
        )

    def add_stat(self, parent, label, value):
        card = tk.Frame(
            parent,
            bg=self.colors["panel2"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )

        card.pack(
            side="left",
            padx=5
        )

        tk.Label(
            card,
            text=label,
            bg=self.colors["panel2"],
            fg=self.colors["muted"],
            font=("Segoe UI", 8)
        ).pack(
            padx=15,
            pady=(8, 0)
        )

        tk.Label(
            card,
            text=value,
            bg=self.colors["panel2"],
            fg=self.colors["text"],
            font=("Segoe UI", 17, "bold")
        ).pack(
            padx=15,
            pady=(0, 8)
        )

    def format_matrix(self, matrix):
        matrix = np.asarray(matrix)

        lines = []

        for row in matrix:
            values = [
                f"{float(x):7.3f}"
                for x in row
            ]

            lines.append(
                "[ " + "  ".join(values) + " ]"
            )

        return "\n".join(lines)

    # ========================================================
    # Analysis
    # ========================================================

    def run_analysis(self):
        if len(self.selected_you) < 3:
            messagebox.showwarning(
                "Not enough songs",
                "Select at least 3 songs for YOUR MUSIC."
            )
            return

        if len(self.selected_them) < 3:
            messagebox.showwarning(
                "Not enough songs",
                "Select at least 3 songs for THEIR MUSIC."
            )
            return

        try:
            analysis = analyze_taste(
                self.df,
                self.selected_you,
                self.selected_them
            )

            self.scaler = analysis["scaler"]

            recommendations = generate_recommendations(
                self.df,
                self.selected_you,
                self.selected_them,
                analysis
            )

            results = {
                **analysis,
                **recommendations
            }

            self.show_results_screen(results)

        except Exception as exc:
            messagebox.showerror(
                "Analysis error",
                f"Something went wrong during analysis:\n\n{exc}"
            )


# ============================================================
# Program entry point
# ============================================================


def main():
    root = tk.Tk()
    app = TasteMatchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
