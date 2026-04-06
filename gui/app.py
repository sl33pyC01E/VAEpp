#!/usr/bin/env python3
"""SynthyperVAE GUI — thin assembler.

Imports tab groups from gui_data, gui_models, gui_compress and
assembles them into a nested notebook layout:
  Data (Static Gen, Video Gen)
  Models (Static Train, Static Inf, Convert, Video Train, Video Inf)
  Compress (Flatten, Flatten Inf, Flatten Video, Flatten Vid Inf, FSQ...)
"""

import tkinter as tk
from tkinter import ttk

from gui.common import BG, BG_PANEL, FG, ACCENT, FONT_BOLD
from gui.data_tabs import GeneratorTab, VideoGenTab
from gui.models_tabs import (
    TrainingTab, InferenceTab, ConvertTab, VideoTrainTab, VideoInferenceTab,
)
from gui.compress_tabs import (
    FlattenTab, FlattenInferenceTab, FlattenVideoTab, FlattenVideoInferenceTab,
    FSQConvertTab, FSQInferenceTab, FlattenFSQTab, FlattenVideoFSQTab,
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SynthyperVAE")
        self.geometry("1100x750")
        self.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=FG,
                        padding=[12, 4], font=FONT_BOLD)
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=5, pady=5)

        # -- Data --
        data_frame = tk.Frame(nb, bg=BG)
        data_nb = ttk.Notebook(data_frame)
        data_nb.pack(fill="both", expand=True)
        data_nb.add(GeneratorTab(data_nb), text="Static Gen")
        data_nb.add(VideoGenTab(data_nb), text="Video Gen")
        nb.add(data_frame, text="Data")

        # -- Models --
        models_frame = tk.Frame(nb, bg=BG)
        models_nb = ttk.Notebook(models_frame)
        models_nb.pack(fill="both", expand=True)
        models_nb.add(TrainingTab(models_nb), text="Static Train")
        models_nb.add(InferenceTab(models_nb), text="Static Inf")
        models_nb.add(ConvertTab(models_nb), text="Convert")
        models_nb.add(VideoTrainTab(models_nb), text="Video Train")
        models_nb.add(VideoInferenceTab(models_nb), text="Video Inf")
        nb.add(models_frame, text="Models")

        # -- Compress --
        compress_frame = tk.Frame(nb, bg=BG)
        compress_nb = ttk.Notebook(compress_frame)
        compress_nb.pack(fill="both", expand=True)
        compress_nb.add(FSQConvertTab(compress_nb), text="FSQ Convert")
        compress_nb.add(FSQInferenceTab(compress_nb), text="FSQ Inf")
        compress_nb.add(FlattenTab(compress_nb), text="Flatten")
        compress_nb.add(FlattenInferenceTab(compress_nb), text="Flatten Inf")
        compress_nb.add(FlattenVideoTab(compress_nb), text="Flatten Video")
        compress_nb.add(FlattenVideoInferenceTab(compress_nb), text="Flatten Vid Inf")
        compress_nb.add(FlattenFSQTab(compress_nb), text="Flatten FSQ")
        compress_nb.add(FlattenVideoFSQTab(compress_nb), text="Flatten Vid FSQ")
        nb.add(compress_frame, text="Compress")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
