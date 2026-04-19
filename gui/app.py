#!/usr/bin/env python3
"""VAEpp0r GUI — thin assembler.

Imports tab groups from gui_data, gui_models, gui_compress and
assembles them into a nested notebook layout:
  Data (Static Gen, Video Gen)
  Models (Static Train, Static Inf, Convert, Video Train, Video Train 3D, Video Inf)
  Compress (Flatten, Flatten Inf, Flatten Video, Flatten Vid Inf)
"""

import tkinter as tk
from tkinter import ttk

from gui.common import BG, BG_PANEL, FG, ACCENT, FONT_BOLD
from gui.data_tabs import GeneratorTab, VideoGenTab, PrefillBankTab
from gui.models_tabs import (
    TrainingTab, InferenceTab, ConvertTab, VideoTrainTab, VideoTrain3DTab,
    VideoInferenceTab,
)
from gui.compress_tabs import (
    FlattenTab, FlattenInferenceTab, FlattenVideoTab, FlattenVideoInferenceTab,
)
from gui.tokenizer_tabs import TokenizerTrainTab, TokenizerInfTab
from gui.elastictok_tabs import ElasticTokTrainTab, ElasticTokInfTab


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VAEpp0r")
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
        data_nb.add(PrefillBankTab(data_nb), text="Prefill Bank")
        nb.add(data_frame, text="Data")

        # -- Models --
        models_frame = tk.Frame(nb, bg=BG)
        models_nb = ttk.Notebook(models_frame)
        models_nb.pack(fill="both", expand=True)
        models_nb.add(TrainingTab(models_nb), text="Static Train")
        models_nb.add(InferenceTab(models_nb), text="Static Inf")
        models_nb.add(ConvertTab(models_nb), text="Convert")
        models_nb.add(VideoTrainTab(models_nb), text="Video Train")
        models_nb.add(VideoTrain3DTab(models_nb), text="Video Train 3D")
        models_nb.add(VideoInferenceTab(models_nb), text="Video Inf")
        nb.add(models_frame, text="Models")

        # -- Tokenizer --
        tok_frame = tk.Frame(nb, bg=BG)
        tok_nb = ttk.Notebook(tok_frame)
        tok_nb.pack(fill="both", expand=True)
        tok_nb.add(TokenizerTrainTab(tok_nb), text="Tokenizer Train")
        tok_nb.add(TokenizerInfTab(tok_nb), text="Tokenizer Inf")
        tok_nb.add(ElasticTokTrainTab(tok_nb), text="ElasticTok Train")
        tok_nb.add(ElasticTokInfTab(tok_nb), text="ElasticTok Inf")
        nb.add(tok_frame, text="Tokenizer")

        # -- Compress --
        compress_frame = tk.Frame(nb, bg=BG)
        compress_nb = ttk.Notebook(compress_frame)
        compress_nb.pack(fill="both", expand=True)
        compress_nb.add(FlattenTab(compress_nb), text="Flatten")
        compress_nb.add(FlattenInferenceTab(compress_nb), text="Flatten Inf")
        compress_nb.add(FlattenVideoTab(compress_nb), text="Flatten Video")
        compress_nb.add(FlattenVideoInferenceTab(compress_nb), text="Flatten Vid Inf")
        nb.add(compress_frame, text="Compress")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
