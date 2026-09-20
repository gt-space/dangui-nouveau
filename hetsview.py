import os
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
#ttkbootstrap pretty much adds more functionality to tkinter
import ttkbootstrap as ttk
from ttkbootstrap.style import Style, ThemeDefinition
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

bgColor = "#2b2b2b"
fgColor = "#e8e8e8"
accents = "#d9d9d9"
subColor = "#6e6e6e"
darkBgColor = "#1c1c1c"

GRAY_CYCLE = ["#f2f2f2", "#bdbdbd", "#8c8c8c", "#5c5c5c", "#3a3a3a"]

#ttkbootstrap theme definition for color palette
# gave it a grayscale look with the 'monkeytype' font
hetsviewTheme = ThemeDefinition(
    name="hetsview",
    themetype="dark",
    colors={
        "primary": accents,
        "secondary": subColor,
        "success": "#9e9e9e",
        "info": "#bdbdbd",
        "warning": "#d9d9d9",
        "danger": "#8c8c8c",
        "dark": darkBgColor,
        "light": fgColor,
        "bg": bgColor,
        "fg": fgColor,
        "selectbg": accents,
        "selectfg": bgColor,
        "border": subColor,
        "inputfg": fgColor,
        "inputbg": darkBgColor,
        "active": accents,
    }
)

#main app class
class hetsview:
    def __init__(self, root):
        self.root = root
        self.root.title("hetsview")
        self.root.geometry("1150x760")
        self.root.minsize(900, 600)

        #apply the theme to the GUI globally
        # also allows each new plot cycle through the GRAY_CYCLE list
        ttk.Style().configure('.', font=('Consolas', 10))
        plt.style.use('dark_background')
        plt.rcParams.update({
            "figure.facecolor": darkBgColor,
            "axes.facecolor": darkBgColor,
            "axes.edgecolor": subColor,
            "axes.labelcolor": fgColor,
            "text.color": fgColor,
            "xtick.color": subColor,
            "ytick.color": subColor,
            "grid.color": subColor,
            "font.family": "monospace",
            "axes.prop_cycle": plt.cycler(color=GRAY_CYCLE),
        })

        #initializing the app state
        #.lines maps the filename to matplotlib
        self.lines = {}

        #.notepads allows for correct naming on the side
        self.notepads = {}

        #makes it so each new plotted item starts out as a different color
        self.color_index = 0

        #tracks whats being editted and prevents control-change callbacks
        self.selected = tk.StringVar()
        self._updating_controls = False
        self.current_view = "main"

        #tracks if canvas is expanded to full view
        self.canvas_fullscreen = False

        #general axis labeling/storing
        self.base_xlabel = ""
        self.base_ylabel = ""
        self.extra_xlabels = []
        self.extra_ylabels = []
        self.extra_text_artists = []

        #actually builds the GUI
        self._build_layout()

        #allowing scrollwheel usage
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)

    #building the actual layout
    def _build_layout(self):
        #setting up grid layout of the widgets
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0) #title bar
        self.root.rowconfigure(1, weight=4) #middle section
        self.root.rowconfigure(2, weight=1) #control panel

        self._build_title_bar()
        self._build_middle_section()
        self._build_control_panel()

    #detailing the title bar
    def _build_title_bar(self):
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        self.top_frame.columnconfigure(0, weight=1) #title size
        self.top_frame.columnconfigure(1, weight=0) #buttons don't change size

        title_lbl = tk.Label(self.top_frame, text="hetsview", font=("Consolas", 26, "bold"),
                             bg=bgColor, fg=fgColor, anchor="w")
        title_lbl.grid(row=0, column=0, sticky="w")

        btn_frame = ttk.Frame(self.top_frame)
        btn_frame.grid(row=0, column=1, sticky="e")

        #import data opens file selector and plots
        self.btn_import = ttk.Button(btn_frame, text="Import Data", command=self.import_data,
                                     bootstyle="light")
        self.btn_import.pack(side=tk.LEFT, padx=(0, 8))

        #resets canvas
        self.btn_clear = ttk.Button(btn_frame, text="Clear Data", command=self.clear_data,
                                    bootstyle="secondary-outline")
        self.btn_clear.pack(side=tk.LEFT)

    def _build_middle_section(self):
        self.middle_frame = ttk.Frame(self.root)
        self.middle_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))
        self.middle_frame.columnconfigure(0, weight=4)
        self.middle_frame.columnconfigure(1, weight=1)
        self.middle_frame.rowconfigure(0, weight=1)

        self._build_canvas(self.middle_frame)
        self._build_notes(self.middle_frame)

    #building the actual canvas + layout
    def _build_canvas(self, parent):
        canvas_container = ttk.Frame(parent)
        canvas_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        canvas_container.rowconfigure(2, weight=1)
        canvas_container.columnconfigure(0, weight=1)

        self.toolbar_frame = ttk.Frame(canvas_container)
        self.toolbar_frame.grid(row=0, column=0, sticky="ew")

        self.view_tabs = ttk.Notebook(canvas_container)
        self.view_tabs.grid(row=1, column=0, sticky="ew")
        self.view_tabs.bind("<<NotebookTabChanged>>", self._on_view_changed)

        main_tab = ttk.Frame(self.view_tabs)
        self.view_tabs.add(main_tab, text="main")

        self.canvas_frame = ttk.Frame(canvas_container)
        self.canvas_frame.grid(row=2, column=0, sticky="nsew")

        self.fig, self.ax = plt.subplots(dpi=100)
        self.fig.subplots_adjust(bottom=0.25, left=0.2)

        self.ax.set_title("No Data To Display", fontsize=12, color=subColor)
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.config(bg=bgColor, bd=0, highlightthickness=0)
        self.toolbar.update()

        #button to recenter plot view next to toolbar items
        self.btn_reset_view = ttk.Button(self.toolbar, text="Reset View", command=self.reset_view,
                                        bootstyle="secondary-outline")
        self.btn_reset_view.pack(side=tk.LEFT, padx=(10, 4), pady=2)

        #button to expand canvas to fill window
        self.btn_fullscreen = ttk.Button(self.toolbar, text="Full Canvas", command=self.toggle_canvas_fullscreen,
                                        bootstyle="secondary-outline")
        self.btn_fullscreen.pack(side=tk.LEFT, padx=(4, 4), pady=2)

        #collapses back to normal layout
        self.btn_exit_fullscreen = tk.Button(self.toolbar, text="X", bg="#5c5c5c", fg="white", bd=0,
                                             padx=8, pady=2, font=("Consolas", 9, "bold"),
                                             command=self.toggle_canvas_fullscreen)

        self.canvas.mpl_connect("pick_event", self._on_pick)

    #toggles between full canvas view and normal layout
    def toggle_canvas_fullscreen(self):
        self.canvas_fullscreen = not self.canvas_fullscreen
        if self.canvas_fullscreen:
            self.top_frame.grid_remove()
            self.bottom_frame.grid_remove()
            self.notes_outer.grid_remove()
            self.middle_frame.grid_configure(padx=5, pady=5)
            self.middle_frame.columnconfigure(1, weight=0)
            self.btn_exit_fullscreen.pack(side=tk.RIGHT, padx=6, pady=2)
        else:
            self.top_frame.grid()
            self.bottom_frame.grid()
            self.notes_outer.grid()
            self.middle_frame.grid_configure(padx=15, pady=(0, 10))
            self._update_sidebar_state()
            self.btn_exit_fullscreen.pack_forget()
        self.canvas.draw_idle()

    #recenters view and fits data back onto the canvas
    def reset_view(self):
        self.toolbar.home()
        self.ax.relim()
        self.ax.autoscale()
        self.canvas.draw_idle()

    #switching tabs to filter single or overlaid plots
    def _on_view_changed(self, event=None):
        tab_id = self.view_tabs.select()
        if not tab_id:
            return
        tab_text = self.view_tabs.tab(tab_id, "text")
        self.current_view = tab_text
        if tab_text in self.lines:
            self.selected.set(tab_text)
            self._on_selection_change()
        self._apply_view()
        self._refresh_notepads()

    #adds a tab for each imported plot
    def _add_plot_tab(self, filename):
        for tab_id in self.view_tabs.tabs():
            if self.view_tabs.tab(tab_id, "text") == filename:
                return
        tab_frame = ttk.Frame(self.view_tabs)
        self.view_tabs.add(tab_frame, text=filename)

    #shows only the active tab's plot or shows all on main
    def _apply_view(self):
        for name, data in self.lines.items():
            if self.current_view == "main":
                is_visible = data.get("visible", True)
            else:
                is_visible = (name == self.current_view) and data.get("visible", True)

            data["line"].set_visible(is_visible)
            for artist in data.get("minmax_artists", []):
                artist.set_visible(is_visible and data.get("minmax", False))

        self._refresh_legend()
        self.canvas.draw_idle()

    #making the notepad widget (coolest part imo)
    def _build_notes(self, parent):
        self.notes_outer = tk.Frame(parent, bg=darkBgColor)
        self.notes_outer.grid(row=0, column=1, sticky="nsew")
        self.notes_outer.columnconfigure(0, weight=1)
        self.notes_outer.rowconfigure(0, weight=1)

        #making the notepad bar scrollable
        self.notes_canvas = tk.Canvas(self.notes_outer, bg=darkBgColor, highlightthickness=0, width=250)
        self.notes_canvas.grid(row=0, column=0, sticky="nsew")

        self.notes_scrollbar = ttk.Scrollbar(self.notes_outer, orient="vertical", command=self.notes_canvas.yview)
        self.notes_scrollbar.grid(row=0, column=1, sticky="ns")

        self.notes_canvas.configure(yscrollcommand=self.notes_scrollbar.set)

        #creating the frame of the notepad widget
        self.notes_container = tk.Frame(self.notes_canvas, bg=darkBgColor)
        self.notes_canvas_window = self.notes_canvas.create_window((0, 0), window=self.notes_container, anchor="nw")

        #adjusting for resizes
        self.notes_container.bind("<Configure>", lambda e: self.notes_canvas.configure(scrollregion=self.notes_canvas.bbox("all")))
        self.notes_canvas.bind("<Configure>", lambda e: self.notes_canvas.itemconfig(self.notes_canvas_window, width=e.width))

    #essentially adding scrolling to the entire widget
    def _on_mousewheel(self, event):
        x, y = self.root.winfo_pointerxy()
        widget = self.root.winfo_containing(x, y)
        if widget and str(widget).startswith(str(self.notes_outer)):
            if event.num == 4 or event.delta > 0:
                self.notes_canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                self.notes_canvas.yview_scroll(1, "units")

    #creating a notepad named to the corresponding plot whenever a plot is imported
    def add_notepad(self, name, title_text, notes_text):
        pad_frame = tk.Frame(self.notes_container, bg="#1c1c1c", bd=1, relief="solid",
                             highlightbackground="#3a3a3a", highlightthickness=1)
        pad_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=4)

        title_bar = tk.Frame(pad_frame, bg="#3a3a3a", height=28)
        title_bar.pack(side=tk.TOP, fill=tk.X)
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text=title_text, bg="#3a3a3a", fg="#e8e8e8",
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=6)

        content_frame = tk.Frame(pad_frame)
        content_frame.pack(side=tk.TOP, fill=tk.X)

        #making the frame of the notepad user-adjustable + syncing keystrokes
        text_frame = tk.Frame(content_frame, height=180)
        text_frame.pack_propagate(False)
        text_frame.pack(side=tk.TOP, fill=tk.X)

        text_area = tk.Text(text_frame, bg="#323437", fg="#d1d0c5", insertbackground="#d1d0c5",
                            font=("Consolas", 14), wrap="word", bd=0)
        text_area.insert("1.0", notes_text)
        text_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        text_area.bind("<KeyRelease>", lambda e, n=name, t=text_area: self._on_notes_edited_specific(n, t))

        resizer = tk.Frame(content_frame, height=6, bg="#4a4a4a", cursor="sb_v_double_arrow")
        resizer.pack(side=tk.BOTTOM, fill=tk.X)
        resizer.bind("<B1-Motion>", lambda e, f=text_frame: self._resize_notepad(e, f))

        #collaspable 'X' button
        btn = tk.Button(title_bar, text="X", bg="#5c5c5c", fg="white", bd=0,
                        command=lambda: self.toggle_notepad(content_frame))
        btn.pack(side=tk.RIGHT, padx=4, pady=4)

        #save button to write notes to file
        btn_save = tk.Button(title_bar, text="Save", bg="#4a4a4a", fg="white", bd=0,
                             font=("Consolas", 8), command=lambda n=name: self.save_notes_to_file(n))
        btn_save.pack(side=tk.RIGHT, padx=(0, 2), pady=4)

        #updating .notes to update it
        self.notepads[name] = {
            "frame": pad_frame,
            "text": text_area,
            "content": content_frame
        }

        self._refresh_notepads()
        self._update_sidebar_state()

    #saves notes back into the original plt file
    def save_notes_to_file(self, name):
        if name not in self.lines:
            return
        data = self.lines[name]
        file_path = data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", f"File for {name} could not be located.")
            return

        new_notes = self.notepads[name]["text"].get("1.0", tk.END).strip()
        data["notes"] = new_notes

        try:
            with open(file_path, 'r') as f:
                lines = f.read().splitlines()

            if len(lines) < 5:
                raise ValueError("File does not contain the 5 metadata lines.")

            #flattens line breaks to maintain 5 line metadata structure
            flat_notes = " ".join(new_notes.splitlines())
            had_quotes = lines[-1].strip().startswith("'") and lines[-1].strip().endswith("'")
            lines[-1] = f"'{flat_notes}'" if had_quotes else flat_notes

            with open(file_path, 'w') as f:
                f.write("\n".join(lines) + "\n")

            messagebox.showinfo("Saved", f"Notes successfully saved to {name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save notes: {e}")

    #only shows notepads for the active plot tab
    def _refresh_notepads(self):
        for name, pad in self.notepads.items():
            if self.current_view == "main" or name == self.current_view:
                pad["frame"].pack(side=tk.TOP, fill=tk.X, padx=2, pady=4)
            else:
                pad["frame"].pack_forget()
        self._update_sidebar_state()

    #prevents notepad from collasping when nothing is actively adjusting it
    def _resize_notepad(self, event, frame):
        new_height = event.y_root - frame.winfo_rooty()
        if new_height > 30:
            frame.config(height=new_height)
            self.notes_container.update_idletasks()
            self.notes_canvas.configure(scrollregion=self.notes_canvas.bbox("all"))

    #hides the notepad when 'X' is clicked
    def toggle_notepad(self, content_frame):
        if content_frame.winfo_manager():
            content_frame.pack_forget()
        else:
            content_frame.pack(side=tk.TOP, fill=tk.X)
        self._update_sidebar_state()

    #controls the notepad sidebar collasping when all notepads are collasped
    def _update_sidebar_state(self):
        any_open = False
        for data in self.notepads.values():
            if data["content"].winfo_manager():
                any_open = True
                break

        if any_open or not self.notepads:
            self.middle_frame.columnconfigure(1, weight=1)
            self.notes_canvas.config(width=250)
        else:
            self.middle_frame.columnconfigure(1, weight=0)
            max_w = 120
            for data in self.notepads.values():
                title_bar = data["frame"].winfo_children()[0]
                children = title_bar.winfo_children()
                w = sum(c.winfo_reqwidth() for c in children) + 40
                if w > max_w:
                    max_w = w
            self.notes_canvas.config(width=max_w)

    def _on_notes_edited_specific(self, name, text_area):
        if name in self.lines:
            self.lines[name]["notes"] = text_area.get("1.0", tk.END).strip()

    #building control panel for plot editting
    def _build_control_panel(self):
        self.bottom_frame = ttk.LabelFrame(self.root, text="Graph Controls")
        self.bottom_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 15))
        for c in range(8):
            self.bottom_frame.columnconfigure(c, weight=1)
        self.bottom_frame.rowconfigure(0, weight=1)
        self.bottom_frame.rowconfigure(1, weight=1)

        #plot selector, color picker, opacity slider
        ttk.Label(self.bottom_frame, text="Editing:").grid(row=0, column=0, sticky="w", padx=(10, 5), pady=(8, 4))
        self.plot_selector = ttk.Combobox(self.bottom_frame, textvariable=self.selected, state="readonly", width=18)
        self.plot_selector.grid(row=0, column=1, sticky="w", pady=(8, 4))
        self.plot_selector.bind("<<ComboboxSelected>>", self._on_selection_change)

        #plot color pickers
        ttk.Label(self.bottom_frame, text="Color:").grid(row=0, column=2, sticky="w", padx=(20, 5), pady=(8, 4))
        self.color_swatch = tk.Button(self.bottom_frame, text="   ", command=self._pick_color,
                                      bg=subColor, relief="flat", width=4)
        self.color_swatch.grid(row=0, column=3, sticky="w", pady=(8, 4))

        ttk.Label(self.bottom_frame, text="Opacity:").grid(row=0, column=4, sticky="w", padx=(20, 5), pady=(8, 4))
        self.opacity_var = tk.DoubleVar(value=1.0)
        self.opacity_scale = ttk.Scale(self.bottom_frame, from_=0.05, to=1.0, variable=self.opacity_var,
                                       orient="horizontal", command=self._on_opacity_change,
                                       bootstyle="light")
        self.opacity_scale.grid(row=0, column=5, sticky="ew", padx=(0, 10), pady=(8, 4))

        #line width slider
        ttk.Label(self.bottom_frame, text="Width:").grid(row=0, column=6, sticky="w", padx=(20, 5), pady=(8, 4))
        self.line_width_var = tk.DoubleVar(value=1.5)
        self.line_width_scale = ttk.Scale(self.bottom_frame, from_=0.5, to=5.0, variable=self.line_width_var,
                                          orient="horizontal", command=self._on_line_width_change,
                                          bootstyle="light")
        self.line_width_scale.grid(row=0, column=7, sticky="ew", padx=(0, 10), pady=(8, 4))

        #toggle switches for markers + min/max
        self.show_points_var = tk.BooleanVar(value=False)
        self.chk_points = ttk.Checkbutton(self.bottom_frame, text="Show Data Points", variable=self.show_points_var,
                                          command=self._on_points_toggle, bootstyle="round-toggle")
        self.chk_points.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 8))

        self.show_minmax_var = tk.BooleanVar(value=False)
        self.chk_minmax = ttk.Checkbutton(self.bottom_frame, text="Show Min/Max", variable=self.show_minmax_var,
                                          command=self._on_minmax_toggle, bootstyle="round-toggle")
        self.chk_minmax.grid(row=1, column=2, columnspan=2, sticky="w", padx=10, pady=(4, 8))

        #plot visibility toggle
        self.show_plot_var = tk.BooleanVar(value=True)
        self.chk_plot = ttk.Checkbutton(self.bottom_frame, text="Show Plot", variable=self.show_plot_var,
                                        command=self._on_plot_visibility_toggle, bootstyle="round-toggle")
        self.chk_plot.grid(row=1, column=4, columnspan=2, sticky="w", padx=10, pady=(4, 8))

        #if no data, no plot controls
        self._set_controls_state("disabled")

    def import_data(self):
        file_paths = filedialog.askopenfilenames(
            title="Select PLT files",
            filetypes=[
                ("PLT files", "*.plt"),
                ("All files", "*.*")
            ]
        )

        #skips multiple redraws when importing a list of files
        for file_path in file_paths:
            self.process_and_plot(file_path, redraw=False)

        if file_paths:
            self.view_tabs.select(0)
            self.current_view = "main"
            self._apply_view()
            self._refresh_notepads()
            self.canvas.draw_idle()

    #parse data file based on hetsview format.txt
    def process_and_plot(self, file_path, redraw=True):
        x, y = [], []
        try:
            with open(file_path, 'r') as file:
                raw_lines = file.read().splitlines()

            if len(raw_lines) < 5:
                raise ValueError("File must contain 5 lines of metadata at the end.")

            #strip meta data
            meta = [m.replace("'", "").strip() for m in raw_lines[-5:]]
            title, xlabel, ylabel, linestyle, notes = meta

            if linestyle not in ['-', '--', '-.', ':']:
                linestyle = '-'

            #faster comma separation on data rows
            for line in raw_lines[:-5]:
                line = line.strip()
                if not line:
                    continue
                comma = line.find(',')
                if comma != -1:
                    try:
                        x.append(float(line[:comma]))
                        y.append(float(line[comma + 1:]))
                    except ValueError:
                        continue

            if not x:
                raise ValueError("No valid data found.")

            filename = os.path.basename(file_path)

            #axis labeling when multiple plots/data are overlaid
            # includes warnings when axis labels aren't the same
            if len(self.lines) > 0:
                if xlabel and self.base_xlabel and xlabel != self.base_xlabel:
                    ans = messagebox.askyesno("Overlay Warning",
                                              "These plots will not have the same x-axis. Are you sure you want to overlay?")
                    if not ans:
                        return
                    if xlabel not in self.extra_xlabels:
                        self.extra_xlabels.append(xlabel)
                if ylabel and self.base_ylabel and ylabel != self.base_ylabel:
                    if ylabel not in self.extra_ylabels:
                        self.extra_ylabels.append(ylabel)
            else:
                self.base_xlabel = xlabel
                self.base_ylabel = ylabel
                self.ax.set_title(title if title else filename, fontsize=12, color=fgColor)

            self._update_axis_labels()

            color = GRAY_CYCLE[self.color_index % len(GRAY_CYCLE)]
            self.color_index += 1

            line, = self.ax.plot(
                x,
                y,
                marker="None",
                linestyle=linestyle,
                color=color,
                alpha=1.0,
                linewidth=1.5,
                label=title if title else filename,
                picker=5
            )

            #precalculating min and max to keep toggle quick
            min_y = min(y)
            max_y = max(y)
            min_idx = y.index(min_y)
            max_idx = y.index(max_y)

            #storing file path so notes can be saved back to disk
            self.lines[filename] = {
                "line": line,
                "x": x,
                "y": y,
                "color": color,
                "alpha": 1.0,
                "linewidth": 1.5,
                "visible": True,
                "markers": False,
                "minmax": False,
                "minmax_artists": [],
                "notes": notes,
                "file_path": file_path,
                "min_idx": min_idx,
                "max_idx": max_idx
            }

            self._add_plot_tab(filename)
            self.add_notepad(filename, title if title else filename, notes)
            self._refresh_legend()

            if redraw:
                self.canvas.draw_idle()

            self.plot_selector["values"] = list(self.lines.keys())
            self.selected.set(filename)
            self._on_selection_change()
            self._set_controls_state("normal")
            self._update_sidebar_state()

        #catch all for malformed files
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the file: {e}")

    #rewriting axis labels when overlapping plots are shown
    def _update_axis_labels(self):
        for artist in self.extra_text_artists:
            artist.remove()
        self.extra_text_artists.clear()

        self.ax.set_xlabel(self.base_xlabel)
        self.ax.set_ylabel(self.base_ylabel)

        y_offset = -0.18
        for ex in self.extra_xlabels:
            txt = self.ax.text(0.5, y_offset, ex, transform=self.ax.transAxes,
                               ha='center', va='top', fontsize=9, color=fgColor)
            self.extra_text_artists.append(txt)
            y_offset -= 0.08

        x_offset = -0.18
        for ey in self.extra_ylabels:
            txt = self.ax.text(x_offset, 0.5, ey, transform=self.ax.transAxes,
                               ha='right', va='center', rotation=90, fontsize=9, color=fgColor)
            self.extra_text_artists.append(txt)
            x_offset -= 0.08

    #huge clear button that just resets the whole GUI
    def clear_data(self):
        self.ax.clear()
        self.ax.set_title("No Data To Display", fontsize=12, color=subColor)
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        self.canvas.draw_idle()

        self.lines.clear()
        self.notepads.clear()

        for widget in self.notes_container.winfo_children():
            widget.destroy()

        #resets tabs back to only main
        for tab_id in self.view_tabs.tabs()[1:]:
            self.view_tabs.forget(tab_id)
        self.view_tabs.select(0)
        self.current_view = "main"

        self.color_index = 0
        self.plot_selector["values"] = []
        self.selected.set("")

        self.base_xlabel = ""
        self.base_ylabel = ""
        self.extra_xlabels.clear()
        self.extra_ylabels.clear()

        for artist in self.extra_text_artists:
            artist.remove()
        self.extra_text_artists.clear()

        self._set_controls_state("disabled")
        self._update_sidebar_state()

    #allows for clickable plots lines to edit
    def _on_pick(self, event):
        artist = event.artist
        for name, data in self.lines.items():
            if data["line"] == artist:
                self.selected.set(name)
                self._on_selection_change()
                break

    def _current(self):
        name = self.selected.get()
        return self.lines.get(name)

    def _on_selection_change(self, event=None):
        data = self._current()
        if not data:
            return

        self._updating_controls = True
        self.color_swatch.config(bg=data["color"])
        self.opacity_var.set(data["alpha"])
        self.line_width_var.set(data.get("linewidth", 1.5))
        self.show_points_var.set(data["markers"])
        self.show_minmax_var.set(data["minmax"])
        self.show_plot_var.set(data.get("visible", True))
        self._updating_controls = False

    def _pick_color(self):
        data = self._current()
        if not data:
            return
        rgb, hex_color = colorchooser.askcolor(initialcolor=data["color"])
        if hex_color:
            data["color"] = hex_color
            data["line"].set_color(hex_color)
            self.color_swatch.config(bg=hex_color)
            self._refresh_legend()
            self.canvas.draw_idle()

    def _on_opacity_change(self, value):
        if self._updating_controls:
            return
        data = self._current()
        if not data:
            return
        alpha = float(value)
        data["alpha"] = alpha
        data["line"].set_alpha(alpha)
        self.canvas.draw_idle()

    def _on_points_toggle(self):
        data = self._current()
        if not data:
            return
        show = self.show_points_var.get()
        data["markers"] = show
        data["line"].set_marker('o' if show else 'None')
        self.canvas.draw_idle()

    def _on_minmax_toggle(self):
        data = self._current()
        if not data:
            return
        show = self.show_minmax_var.get()
        data["minmax"] = show

        for artist in data["minmax_artists"]:
            artist.remove()
        data["minmax_artists"] = []

        if show:
            x, y = data["x"], data["y"]
            i_max, i_min = data["max_idx"], data["min_idx"]
            for i, label in ((i_max, "max"), (i_min, "min")):
                ann = self.ax.annotate(
                    f"{label}: {y[i]:.2f}", xy=(x[i], y[i]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color=fgColor,
                    arrowprops=dict(arrowstyle="->", color=subColor, lw=0.8),
                )
                marker, = self.ax.plot(x[i], y[i], marker='x', color=data["color"],
                                       markersize=8, linestyle='None')
                data["minmax_artists"].extend([ann, marker])

        self.canvas.draw_idle()

    def _refresh_legend(self):
        visible_handles_labels = [
            (h, l) for h, l in zip(*self.ax.get_legend_handles_labels()) if h.get_visible()
        ]
        if visible_handles_labels:
            handles, labels = zip(*visible_handles_labels)
            self.ax.legend(handles, labels, facecolor=darkBgColor, edgecolor=subColor, labelcolor=fgColor)
        elif self.ax.get_legend():
            self.ax.get_legend().remove()

    def _set_controls_state(self, state):
        self.plot_selector.config(
            state="readonly" if self.lines else "disabled"
        )
        self.color_swatch.config(state=state)
        self.opacity_scale.config(state=state)
        self.line_width_scale.config(state=state)
        self.chk_points.config(state=state)
        self.chk_minmax.config(state=state)
        self.chk_plot.config(state=state)

    def _on_line_width_change(self, value):
        if self._updating_controls:
            return

        data = self._current()
        if not data:
            return

        width = float(value)
        data["linewidth"] = width
        data["line"].set_linewidth(width)
        self.canvas.draw_idle()

    def _on_plot_visibility_toggle(self):
        data = self._current()
        if not data:
            return

        data["visible"] = self.show_plot_var.get()
        self._apply_view()


if __name__ == "__main__":
    root = tk.Tk()
    style = Style()
    style.register_theme(hetsviewTheme)
    style.theme_use("hetsview")
    app = hetsview(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
