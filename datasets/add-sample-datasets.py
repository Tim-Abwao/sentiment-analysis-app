from pathlib import Path
from tkinter import (
    Button,
    Frame,
    Label,
    StringVar,
    Tk,
    filedialog,
    messagebox,
    simpledialog,
)

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_SIZE = 1000
FEATURES = ["star_rating", "review_body", "review_headline"]
SEED = 7654  # for reproducability


class SamplingApp(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Dataset Sampling App")
        self.master.geometry("400x240")
        self.master.resizable(False, False)  # Fix window size
        self.current_action = StringVar()
        self._create_content()
        self.pack()

    def _create_content(self):
        """Prepare and display a brief introduction."""
        self.intro_title = Label(
            self,
            text="Amazon Reviews Data Sampler",
            font=("Courier", 21, "bold"),
            justify="center",
            wraplength=320,
        )
        self.intro_title.pack(pady=(30, 10))

        self.intro_body = Label(
            self,
            text=(
                "Extract sentiment analysis samples from the Amazon Customer"
                " Reviews datasets."
            ),
            font=("Times", 14, "italic"),
            wraplength=360,
        )
        self.intro_body.pack()

        self.button = Button(
            self,
            text="Select Files",
            bg="#cf5",
            font=("Courier", 12),
            width=90,
            height=25,
            command=self._fetch_files,
        )
        self.button.pack(padx=115, pady=(10, 50))

    def _fetch_files(self):
        """Open a filedialog to select data files to process."""
        self.current_action.set("Fetching files...")
        self.files = filedialog.askopenfilenames(
            initialdir=Path.home() / "Downloads",
            filetypes=[("compressed tsv", "*.tsv.gz")],
        )
        self._get_samples()

    def _get_samples(self):
        """Extract and save samples of the selected files."""
        self.current_action.set("Extracting samples...")
        if self.files:
            SAMPLE_SIZE = simpledialog.askinteger(
                title="Sample size", prompt="Enter sample size: "
            )
            process_files(
                files=self.files,
                sample_size=SAMPLE_SIZE or DEFAULT_SAMPLE_SIZE,
            )
            messagebox.showinfo(
                message="Done! Results saved to current folder."
            )
        else:
            self.destroy()


def process_files(*, files: tuple, sample_size: int) -> None:
    """Prepare sentiment analysis samples of the specified size from the
    supplied files.

    The files should be one of the Amazon Customer Reviews datasets available
    at https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt.

    Ratings less than 3 are considered negative, while ratings of 3 to 5 are
    assumed to be positive.

    Args:
        files (tuple): Paths to files.
        sample_size (int): Desired sample size.
    """
    for file in files:
        filepath = Path(file)
        data = pd.read_csv(filepath, sep="\t", usecols=FEATURES).dropna()
        data["text"] = data["review_headline"] + " " + data["review_body"]

        # Set [1, 2] as 0 (-ve) and [3, 4, 5] as 1 (+ve)
        data["sentiment"] = np.where(data["star_rating"] < 3, 0, 1)
        data.drop(columns=FEATURES, inplace=True)

        positive_sample = data.query("sentiment == 1").sample(
            sample_size // 2 + 1, random_state=SEED
        )
        negative_sample = data.query("sentiment == 0").sample(
            sample_size // 2 + 1, random_state=SEED
        )

        source_info = filepath.name.rstrip(".tsv.gz").split("_")[3:]
        name = "-".join(source_info).lower()
        pd.concat([positive_sample, negative_sample]).sample(
            sample_size, random_state=SEED
        ).to_csv(f"{name}-reviews-sample.csv.xz", index=False)


if __name__ == "__main__":
    app = SamplingApp(master=Tk())
    app.mainloop()
